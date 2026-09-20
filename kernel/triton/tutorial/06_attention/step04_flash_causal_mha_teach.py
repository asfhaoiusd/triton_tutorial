"""FlashAttention-2 教学版：causal + 多头（对照论文双循环）

布局: Q,K,V,O = (B, H, S, D)  # D 是 head dim

每个 program 负责一个 (batch, head)，内部仍是论文双循环:
    for i in Q 行块:
        for j in K/V 列块:          # 沿序列 S，不是沿 D
            causal: 只看 offs_n <= offs_m
            online softmax + acc
        写回 O_i

头之间互不干扰，用 pid_h 并行，不要 Python 里 for h 多次 launch。

训练 / 加速版在同目录 `step04_flash_causal_mha.py`（Q 块并行、GQA GROUP）。
本文件只为把公式和数据流看清楚。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8),
    ],
    key=["S", "D"],
)
@triton.jit
def flash_attention_fwd_causal_and_multihead_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    B,
    H,
    S,
    D,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_km,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vm,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lb,
    stride_lh,
    stride_ls,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    offs_d = tl.arange(0, BLOCK_D)

    q_bh = q_ptr + pid_b * stride_qb + pid_h * stride_qh
    k_bh = k_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_bh = v_ptr + pid_b * stride_vb + pid_h * stride_vh
    o_bh = o_ptr + pid_b * stride_ob + pid_h * stride_oh
    lse_bh = lse_ptr + pid_b * stride_lb + pid_h * stride_lh

    for start_m in range(0, S, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)

        q = tl.load(
            q_bh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=q_mask,
            other=0.0,
        )

        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for start_n in range(0, S, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_mask = (offs_n[:, None] < S) & (offs_d[None, :] < D)

            k = tl.load(
                k_bh + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
                mask=kv_mask,
                other=0.0,
            )
            v = tl.load(
                v_bh + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
                mask=kv_mask,
                other=0.0,
            )

            qk = tl.dot(q, tl.trans(k)) * scale
            # 边界 + causal：query 只能看 <= 自己位置的 key
            keep = (
                (offs_m[:, None] < S)
                & (offs_n[None, :] < S)
                & (offs_n[None, :] <= offs_m[:, None])
            )
            qk = tl.where(keep, qk, -float("inf"))

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            p = tl.where(keep, p, 0.0)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        tl.store(
            o_bh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
            acc,
            mask=q_mask,
        )
        # LSE_i = m_i + log(l_i)，反向用它还原 P_ij = exp(S_ij - LSE_i)
        tl.store(lse_bh + offs_m * stride_ls, m_i + tl.log(l_i), mask=offs_m < S)


def flash_attention_fwd_causal_mha(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """q,k,v: (B, H, S, D) → (B, H, S, D)，causal。"""
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape and q.ndim == 4

    B, H, S, D = q.shape
    o = torch.empty_like(q)
    lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(D)
    scale = D**-0.5

    grid = lambda meta: (B, H)
    flash_attention_fwd_causal_and_multihead_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        B,
        H,
        S,
        D,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        scale,
        BLOCK_D=BLOCK_D,
    )
    return o, lse


@triton.jit
def flash_attention_bwd_causal_mha_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    LSE_ptr,
    dO_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    B,
    H,
    S,
    D,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_km,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vm,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    stride_dkb,
    stride_dkh,
    stride_dkm,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvm,
    stride_dvd,
    scale,
    BLOCK_Br: tl.constexpr,
    BLOCK_Bc: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """外层 KV 列块（本地攒 dK/dV），内层 Q 行块（dQ 用 atomic_add）。"""
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    offs_d = tl.arange(0, BLOCK_D)

    Q_bh = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    K_bh = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    V_bh = V_ptr + pid_b * stride_vb + pid_h * stride_vh
    O_bh = O_ptr + pid_b * stride_ob + pid_h * stride_oh
    LSE_bh = LSE_ptr + pid_b * stride_lb + pid_h * stride_lh
    dO_bh = dO_ptr + pid_b * stride_dob + pid_h * stride_doh
    dQ_bh = dQ_ptr + pid_b * stride_dqb + pid_h * stride_dqh
    dK_bh = dK_ptr + pid_b * stride_dkb + pid_h * stride_dkh
    dV_bh = dV_ptr + pid_b * stride_dvb + pid_h * stride_dvh

    # 沿序列 S 切 KV，不是沿 D
    for start_c in range(0, S, BLOCK_Bc):
        offs_c = start_c + tl.arange(0, BLOCK_Bc)
        kv_mask = (offs_c[:, None] < S) & (offs_d[None, :] < D)

        k = tl.load(
            K_bh + offs_c[:, None] * stride_km + offs_d[None, :] * stride_kd,
            mask=kv_mask,
            other=0.0,
        )
        v = tl.load(
            V_bh + offs_c[:, None] * stride_vm + offs_d[None, :] * stride_vd,
            mask=kv_mask,
            other=0.0,
        )
        dk = tl.zeros((BLOCK_Bc, BLOCK_D), dtype=tl.float32)
        dv = tl.zeros((BLOCK_Bc, BLOCK_D), dtype=tl.float32)

        for start_r in range(0, S, BLOCK_Br):
            offs_r = start_r + tl.arange(0, BLOCK_Br)
            q_mask = (offs_r[:, None] < S) & (offs_d[None, :] < D)
            row_mask = offs_r < S

            q = tl.load(
                Q_bh + offs_r[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=q_mask,
                other=0.0,
            )
            o = tl.load(
                O_bh + offs_r[:, None] * stride_om + offs_d[None, :] * stride_od,
                mask=q_mask,
                other=0.0,
            )
            dO = tl.load(
                dO_bh + offs_r[:, None] * stride_dom + offs_d[None, :] * stride_dod,
                mask=q_mask,
                other=0.0,
            )
            lse = tl.load(LSE_bh + offs_r * stride_ls, mask=row_mask, other=0.0)

            # 和正向同一套 causal：key 位置 offs_c <= query 位置 offs_r
            keep = (
                (offs_r[:, None] < S)
                & (offs_c[None, :] < S)
                & (offs_c[None, :] <= offs_r[:, None])
            )
            S_ij = tl.dot(q, tl.trans(k)) * scale
            S_ij = tl.where(keep, S_ij, -float("inf"))
            p = tl.exp(S_ij - lse[:, None])
            p = tl.where(keep, p, 0.0)

            # Δ_i = rowsum(dO * O) = rowsum(dP * P)，避免先物化完整 dP 再求和
            delta = tl.sum(dO * o, axis=1)
            dv += tl.dot(tl.trans(p.to(dO.dtype)), dO)
            dP = tl.dot(dO, tl.trans(v))
            dS = p * (dP - delta[:, None])
            dq = tl.dot(dS.to(k.dtype), k) * scale
            dk += tl.dot(tl.trans(dS.to(q.dtype)), q) * scale

            tl.atomic_add(
                dQ_bh + offs_r[:, None] * stride_dqm + offs_d[None, :] * stride_dqd,
                dq,
                mask=q_mask,
            )

        tl.store(
            dK_bh + offs_c[:, None] * stride_dkm + offs_d[None, :] * stride_dkd,
            dk,
            mask=kv_mask,
        )
        tl.store(
            dV_bh + offs_c[:, None] * stride_dvm + offs_d[None, :] * stride_dvd,
            dv,
            mask=kv_mask,
        )


def flash_attention_bwd_causal_mha(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert q.is_cuda and k.is_cuda and v.is_cuda and o.is_cuda and lse.is_cuda and do.is_cuda
    assert q.shape == k.shape == v.shape == o.shape == do.shape and q.ndim == 4
    B, H, S, D = q.shape
    assert lse.shape == (B, H, S)
    # dQ 多 KV 块用 atomic_add 累加，必须从 0 开始，不能 empty
    dq = torch.zeros_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    scale = D**-0.5
    BLOCK_D = triton.next_power_of_2(D)
    # 64²×64 在本机曾超 smem（上限约 101376）；教学默认 32×32
    BLOCK_Br, BLOCK_Bc = 32, 32
    grid = (B, H)
    flash_attention_bwd_causal_mha_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        do,
        dq,
        dk,
        dv,
        B,
        H,
        S,
        D,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        do.stride(3),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        scale,
        BLOCK_Br=BLOCK_Br,
        BLOCK_Bc=BLOCK_Bc,
        BLOCK_D=BLOCK_D,
    )
    return dq, dk, dv


class TritonFlashAttentionCausalMHA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        o, lse = flash_attention_fwd_causal_mha(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = flash_attention_bwd_causal_mha(q, k, v, o, lse, do)
        return dq, dk, dv


def _check_grad(q, k, v, do):
    """Triton FA bwd 对照 SDPA。o 必须来自可微的 torch 图，不能是裸 Triton 输出。"""
    q_ref = q.detach().requires_grad_(True)
    k_ref = k.detach().requires_grad_(True)
    v_ref = v.detach().requires_grad_(True)
    o_ref = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(o_ref, (q_ref, k_ref, v_ref), do)

    q_t = q.detach().requires_grad_(True)
    k_t = k.detach().requires_grad_(True)
    v_t = v.detach().requires_grad_(True)
    o_t = TritonFlashAttentionCausalMHA.apply(q_t, k_t, v_t)
    dq_t, dk_t, dv_t = torch.autograd.grad(o_t, (q_t, k_t, v_t), do)

    rtol, atol = 1e-2, 1e-2
    print(f"dq allclose: {torch.allclose(dq_t, dq_ref, rtol=rtol, atol=atol)}  "
          f"max abs err: {(dq_t - dq_ref).abs().max().item():.3e}")
    print(f"dk allclose: {torch.allclose(dk_t, dk_ref, rtol=rtol, atol=atol)}  "
          f"max abs err: {(dk_t - dk_ref).abs().max().item():.3e}")
    print(f"dv allclose: {torch.allclose(dv_t, dv_ref, rtol=rtol, atol=atol)}  "
          f"max abs err: {(dv_t - dv_ref).abs().max().item():.3e}")
    return dq_t, dk_t, dv_t


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    B, H, S, D = 2, 8, 256, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)

    o_triton, _lse = flash_attention_fwd_causal_mha(q, k, v)
    o_torch = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    print(f"allclose: {torch.allclose(o_torch, o_triton, rtol=1e-2, atol=1e-2)}")
    print(f"max abs err: {(o_torch - o_triton).abs().max().item():.3e}")
    print(f"shape: QKV{(B, H, S, D)}")
    do = torch.randn_like(o_triton)
    print(
        f"torch sdpa forward : {triton.testing.do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)):.3f} ms"
    )
    print(
        f"triton forward     : {triton.testing.do_bench(lambda: flash_attention_fwd_causal_mha(q, k, v)):.3f} ms"
    )
    q_ref = q.detach().requires_grad_(True)
    k_ref = k.detach().requires_grad_(True)
    v_ref = v.detach().requires_grad_(True)
    o_ref = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    print(
        f"torch sdpa backward : {triton.testing.do_bench(lambda: torch.autograd.grad(o_ref, (q_ref, k_ref, v_ref), do, retain_graph=True)):.3f} ms"
    )
    print(
        f"triton backward     : {triton.testing.do_bench(lambda: flash_attention_bwd_causal_mha(q, k, v, o_triton, _lse, do)):.3f} ms"
    )
    print("--- _check_grad ---")
    _check_grad(q, k, v, do)


if __name__ == "__main__":
    main()
