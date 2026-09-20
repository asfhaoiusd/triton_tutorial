"""FlashAttention-2 fwd（教学版，单头）

论文双循环（每个 batch）:

    for i in Q 行块:                 # BLOCK_M
        加载 Q_i，初始化 (m, l, acc)
        for j in K/V 列块:           # BLOCK_N
            S_ij = Q_i @ K_j^T / sqrt(D)
            online softmax 更新 (m, l)，rescale acc
            acc += P_ij @ V_j
        O_i = acc / l

工程上常把外层 i 映射成 program_id，只留下内层 for；
本文件按论文写成双重 for，方便对照。grid 只需 (B,)。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _prune_flash_configs(configs, named_args, **_kwargs):
    """教学核把整段 D 放进 SRAM。D=512 时 32x32 就要 ~300KB，本机上限约 99KB。"""
    block_d = triton.next_power_of_2(int(named_args["D"]))
    limit = 90_000
    kept = []
    for cfg in configs:
        bm, bn = cfg.kwargs["BLOCK_M"], cfg.kwargs["BLOCK_N"]
        # Q + acc + K + V，fp32，不算 pipeline 余量
        est = 4 * (2 * bm * block_d + 2 * bn * block_d)
        if est < limit:
            kept.append(cfg)
    return kept


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8),
    ],
    key=["S", "D"],
    prune_configs_by={"early_config_prune": _prune_flash_configs},
)
@triton.jit
def flash_atten_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    B,
    S,
    D,
    stride_qb,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_km,
    stride_kd,
    stride_vb,
    stride_vm,
    stride_vd,
    stride_ob,
    stride_om,
    stride_od,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    offs_d = tl.arange(0, BLOCK_D)

    q_batch = q_ptr + pid_b * stride_qb
    k_batch = k_ptr + pid_b * stride_kb
    v_batch = v_ptr + pid_b * stride_vb
    o_batch = o_ptr + pid_b * stride_ob

    # 外循环：Q 行块 i
    for start_m in range(0, S, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)

        q = tl.load(
            q_batch + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=q_mask,
            other=0.0,
        )

        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        # 内循环：K/V 列块 j（不能拆成 program_id，否则 (m,l) 无法在线合并）
        for start_n in range(0, S, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_mask = (offs_n[:, None] < S) & (offs_d[None, :] < D)

            k = tl.load(
                k_batch + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
                mask=kv_mask,
                other=0.0,
            )
            v = tl.load(
                v_batch + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
                mask=kv_mask,
                other=0.0,
            )

            qk = tl.dot(q, tl.trans(k)) * scale
            qk = tl.where(
                (offs_m[:, None] < S) & (offs_n[None, :] < S),
                qk,
                -float("inf"),
            )

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            p = tl.where(
                (offs_m[:, None] < S) & (offs_n[None, :] < S),
                p,
                0.0,
            )

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        tl.store(
            o_batch + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
            acc,
            mask=q_mask,
        )


def flash_atten_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape and q.ndim == 3

    B, S, D = q.shape
    o = torch.empty_like(q)
    BLOCK_D = triton.next_power_of_2(D)
    # 配置里最小 BLOCK 是 32；教学核整段 D 在 SRAM
    min_sram = 4 * (2 * 32 * BLOCK_D + 2 * 32 * BLOCK_D)
    if min_sram >= 90_000:
        raise RuntimeError(
            f"教学 flash 跑不了 D={D}（BLOCK_D={BLOCK_D}）："
            f"最小 32×32 估计 SRAM {min_sram}B，本机上限约 99KB。"
            "对照三种实现请用 D=64。"
        )
    scale = D**-0.5

    grid = lambda meta: (B,)
    flash_atten_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        B,
        S,
        D,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        scale,
        BLOCK_D=BLOCK_D,
    )
    return o


def attention_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) * (d**-0.5)
    attn = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    B, S, D = 4, 256, 64
    q = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, S, D, device="cuda", dtype=torch.float32)

    o_triton = flash_atten_fwd(q, k, v)
    o_torch = attention_torch(q, k, v)

    print(f"allclose: {torch.allclose(o_torch, o_triton, rtol=1e-2, atol=1e-2)}")
    print(f"max abs err: {(o_torch - o_triton).abs().max().item():.3e}")
    print(f"shape: QKV{(B, S, D)}")
    print(f"torch : {triton.testing.do_bench(lambda: attention_torch(q, k, v)):.3f} ms")
    print(f"flash : {triton.testing.do_bench(lambda: flash_atten_fwd(q, k, v)):.3f} ms")


if __name__ == "__main__":
    main()
