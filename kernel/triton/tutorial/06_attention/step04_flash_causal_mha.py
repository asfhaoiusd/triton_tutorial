"""FlashAttention-2：causal + 多头（训练 / 加速版）

相对教学版的改动：Q 块并行、GQA `GROUP`、causal 循环上界、dK/dV fp32 atomic。
论文双循环、`grid=(B,H)` 的教学代码在 `step04_flash_causal_mha_teach.py`。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

_TUT = Path(__file__).resolve().parent.parent
if str(_TUT) not in sys.path:
    sys.path.insert(0, str(_TUT))
from kernel_tiles import launch, tiles as tile_override


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
    GROUP: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_hkv = pid_h // GROUP
    start_m = pid_m * BLOCK_M
    offs_d = tl.arange(0, BLOCK_D)
    offs_m = start_m + tl.arange(0, BLOCK_M)

    q_bh = q_ptr + pid_b * stride_qb + pid_h * stride_qh
    k_bh = k_ptr + pid_b * stride_kb + pid_hkv * stride_kh
    v_bh = v_ptr + pid_b * stride_vb + pid_hkv * stride_vh
    o_bh = o_ptr + pid_b * stride_ob + pid_h * stride_oh
    lse_bh = lse_ptr + pid_b * stride_lb + pid_h * stride_lh

    q_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    q = tl.load(
        q_bh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=q_mask,
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # causal：只需扫到当前 Q 块最后一行；Triton 不能 break，用动态上界收循环
    end_n = tl.minimum(S, start_m + BLOCK_M)
    for start_n in range(0, end_n, BLOCK_N):
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
    tl.store(lse_bh + offs_m * stride_ls, m_i + tl.log(l_i), mask=offs_m < S)


def flash_attention_fwd_causal_mha(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """q,k,v: (B, H, S, D) → (B, H, S, D)，causal。"""
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.ndim == 4 and k.shape == v.shape
    assert q.shape[0] == k.shape[0] and q.shape[2:] == k.shape[2:]
    hq, hkv = q.shape[1], k.shape[1]
    assert hq % hkv == 0, f"Hq={hq} 必须能被 Hkv={hkv} 整除"
    group = hq // hkv

    B, H, S, D = q.shape
    o = torch.empty_like(q)
    lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(D)
    scale = D**-0.5
    ov = tile_override()
    args = (
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
    )
    if ov:
        bm = ov["fa_block_m"]
        launch(
            flash_attention_fwd_causal_and_multihead_kernel,
            (triton.cdiv(S, bm), B, H),
            *args,
            block_kwargs={
                "BLOCK_M": bm,
                "BLOCK_N": ov["fa_block_n"],
                "BLOCK_D": BLOCK_D,
                "GROUP": group,
                "num_warps": ov["fa_num_warps"],
                "num_stages": ov["fa_num_stages"],
            },
        )
        return o, lse

    grid = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B, H)
    flash_attention_fwd_causal_and_multihead_kernel[grid](
        *args, BLOCK_D=BLOCK_D, GROUP=group
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
    GROUP: tl.constexpr,
):
    """一个 program 一块 Q（本地写 dQ）；沿 KV 循环，dK/dV 用 atomic_add。GQA：pid_hkv = pid_h // GROUP。"""
    pid_r = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_hkv = pid_h // GROUP
    start_r = pid_r * BLOCK_Br
    offs_d = tl.arange(0, BLOCK_D)
    offs_r = start_r + tl.arange(0, BLOCK_Br)

    Q_bh = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    K_bh = K_ptr + pid_b * stride_kb + pid_hkv * stride_kh
    V_bh = V_ptr + pid_b * stride_vb + pid_hkv * stride_vh
    O_bh = O_ptr + pid_b * stride_ob + pid_h * stride_oh
    LSE_bh = LSE_ptr + pid_b * stride_lb + pid_h * stride_lh
    dO_bh = dO_ptr + pid_b * stride_dob + pid_h * stride_doh
    dQ_bh = dQ_ptr + pid_b * stride_dqb + pid_h * stride_dqh
    dK_bh = dK_ptr + pid_b * stride_dkb + pid_hkv * stride_dkh
    dV_bh = dV_ptr + pid_b * stride_dvb + pid_hkv * stride_dvh

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
    delta = tl.sum(dO * o, axis=1)
    dq = tl.zeros((BLOCK_Br, BLOCK_D), dtype=tl.float32)

    end_c = tl.minimum(S, start_r + BLOCK_Br)
    for start_c in range(0, end_c, BLOCK_Bc):
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
        keep = (
            (offs_r[:, None] < S)
            & (offs_c[None, :] < S)
            & (offs_c[None, :] <= offs_r[:, None])
        )
        S_ij = tl.dot(q, tl.trans(k)) * scale
        S_ij = tl.where(keep, S_ij, -float("inf"))
        p = tl.exp(S_ij - lse[:, None])
        p = tl.where(keep, p, 0.0)
        dv = tl.dot(tl.trans(p.to(v.dtype)), dO.to(v.dtype))
        dP = tl.dot(dO.to(v.dtype), tl.trans(v))
        dS = p * (dP - delta[:, None])
        dq += tl.dot(dS.to(k.dtype), k) * scale
        dk = tl.dot(tl.trans(dS.to(q.dtype)), q) * scale
        # GQA 多 Q 头写同一 KV：fp32 atomic 比 bf16 atomic 快、也更稳
        tl.atomic_add(
            dK_bh + offs_c[:, None] * stride_dkm + offs_d[None, :] * stride_dkd,
            dk,
            mask=kv_mask,
        )
        tl.atomic_add(
            dV_bh + offs_c[:, None] * stride_dvm + offs_d[None, :] * stride_dvd,
            dv.to(tl.float32),
            mask=kv_mask,
        )

    tl.store(
        dQ_bh + offs_r[:, None] * stride_dqm + offs_d[None, :] * stride_dqd,
        dq,
        mask=q_mask,
    )

def flash_attention_bwd_causal_mha(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert q.is_cuda and k.is_cuda and v.is_cuda and o.is_cuda and lse.is_cuda and do.is_cuda
    # autocast 下 QKV/O 常是 bf16，CE 回传的 dO 常是 fp32；tl.dot 要求两侧同 dtype
    if do.dtype != q.dtype:
        do = do.to(dtype=q.dtype)
    assert q.ndim == 4 and k.shape == v.shape and o.shape == do.shape == q.shape
    assert q.shape[0] == k.shape[0] and q.shape[2:] == k.shape[2:]
    hq, hkv = q.shape[1], k.shape[1]
    assert hq % hkv == 0
    group = hq // hkv
    B, H, S, D = q.shape
    assert lse.shape == (B, H, S)
    # dK/dV 多 program atomic_add（GQA 同组 Q 头会撞同一 KV）；用 fp32 累加再 cast
    dq = torch.empty_like(q)
    dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
    dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
    scale = D**-0.5
    BLOCK_D = triton.next_power_of_2(D)
    # bwd 同时驻留 Q/K/V/O/dO/P/dS，64²×64 会超 smem（本机上限 101376）
    ov = tile_override()
    if ov:
        BLOCK_Br, BLOCK_Bc = ov["fa_block_br"], ov["fa_block_bc"]
        bwd_warps = ov["fa_bwd_num_warps"]
    else:
        BLOCK_Br, BLOCK_Bc = 32, 32
        bwd_warps = None
    grid = (triton.cdiv(S, BLOCK_Br), B, H)
    launch(
        flash_attention_bwd_causal_mha_kernel,
        grid,
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
        block_kwargs={
            "BLOCK_Br": BLOCK_Br,
            "BLOCK_Bc": BLOCK_Bc,
            "BLOCK_D": BLOCK_D,
            "GROUP": group,
            **({"num_warps": bwd_warps} if bwd_warps is not None else {}),
            **({"num_stages": ov["fa_bwd_num_stages"]} if ov else {"num_stages": 1}),
        },
    )
    return dq, dk.to(dtype=k.dtype), dv.to(dtype=v.dtype)

class TritonFlashAttentionCausalMHA(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, q, k, v):
        o, lse = flash_attention_fwd_causal_mha(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
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
    # 数值对照走可微图；计时不要复用已经用过的 o_torch。
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
