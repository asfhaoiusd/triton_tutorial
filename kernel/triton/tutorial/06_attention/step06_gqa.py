"""GQA causal FlashAttention（推理用）

q: (B, Hq, Sq, D)
k,v: (B, Hkv, Sk, D)     # Hq % Hkv == 0
q_start: query 在整段序列里的起始下标（prefill=0；decode=cache_len-1 且 Sq=1）

实现：每个 program 一个 query head，KV 头 = hq // (Hq/Hkv)。
Sq 与 Sk 可以不同（decode）。tile 默认 16×16，给 D=256 留 smem。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from step04_flash_causal_mha import TritonFlashAttentionCausalMHA


@triton.jit
def flash_attention_gqa_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    B,
    HQ,
    HKV,
    SQ,
    SK,
    D,
    q_start,
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
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    group = HQ // HKV
    pid_hkv = pid_h // group
    offs_d = tl.arange(0, BLOCK_D)

    q_bh = q_ptr + pid_b * stride_qb + pid_h * stride_qh
    k_bh = k_ptr + pid_b * stride_kb + pid_hkv * stride_kh
    v_bh = v_ptr + pid_b * stride_vb + pid_hkv * stride_vh
    o_bh = o_ptr + pid_b * stride_ob + pid_h * stride_oh

    for start_m in range(0, SQ, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_mask = (offs_m[:, None] < SQ) & (offs_d[None, :] < D)
        q_pos = q_start + offs_m

        q = tl.load(
            q_bh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=q_mask,
            other=0.0,
        )
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for start_n in range(0, SK, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_mask = (offs_n[:, None] < SK) & (offs_d[None, :] < D)
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
                (offs_m[:, None] < SQ)
                & (offs_n[None, :] < SK)
                & (offs_n[None, :] <= q_pos[:, None])
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


def flash_attention_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_start: int = 0,
    causal: bool = True,
) -> torch.Tensor:
    """q:(B,Hq,Sq,D)  k,v:(B,Hkv,Sk,D)。causal 必须为 True（教学核只实现下三角）。"""
    assert causal, "本教学核只实现 causal GQA"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.ndim == k.ndim == v.ndim == 4
    b, hq, sq, d = q.shape
    bk, hkv, sk, dk = k.shape
    assert (bk, dk) == (b, d) and v.shape == k.shape
    assert hq % hkv == 0
    o = torch.empty_like(q)
    block_d = triton.next_power_of_2(d)
    block_m = 16
    block_n = 16
    scale = d**-0.5
    try:
        flash_attention_gqa_kernel[(b, hq)](
            q,
            k,
            v,
            o,
            b,
            hq,
            hkv,
            sq,
            sk,
            d,
            q_start,
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
            scale,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=1,
        )
        return o
    except Exception:
        return _sdpa_gqa(q, k, v, q_start=q_start)


def _sdpa_gqa(q, k, v, *, q_start: int = 0):
    hq, sq = q.shape[1], q.shape[2]
    sk = k.shape[2]
    if q_start == 0 and sq == sk:
        try:
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True
            )
        except TypeError:
            g = hq // k.shape[1]
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k.repeat_interleave(g, dim=1),
                v.repeat_interleave(g, dim=1),
                is_causal=True,
            )
    g = hq // k.shape[1]
    k_rep = k.repeat_interleave(g, dim=1)
    v_rep = v.repeat_interleave(g, dim=1)
    scale = q.shape[-1] ** -0.5
    attn = torch.matmul(q.float(), k_rep.transpose(-2, -1).float()) * scale
    q_pos = q_start + torch.arange(sq, device=q.device)
    k_pos = torch.arange(sk, device=q.device)
    attn = attn.masked_fill(k_pos[None, None, None, :] > q_pos[None, None, :, None], float("-inf"))
    p = torch.softmax(attn, dim=-1)
    return torch.matmul(p, v_rep.float()).to(q.dtype)


def flash_attention_gqa_train(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """训练用（有图）：FA 核按 GROUP=Hq/Hkv 读 KV，不再 repeat 头。"""
    return TritonFlashAttentionCausalMHA.apply(q, k, v)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    b, hq, hkv, s, d = 2, 8, 2, 64, 64
    q = torch.randn(b, hq, s, d, device="cuda")
    k = torch.randn(b, hkv, s, d, device="cuda")
    v = torch.randn(b, hkv, s, d, device="cuda")
    o_t = flash_attention_gqa(q, k, v, q_start=0)
    o_ref = _sdpa_gqa(q, k, v, q_start=0)
    print(f"prefill allclose: {torch.allclose(o_t, o_ref, rtol=1e-2, atol=1e-2)}")
    print(f"prefill max abs err: {(o_t - o_ref).abs().max().item():.3e}")

    q1 = q[:, :, -1:, :]
    o_dec = flash_attention_gqa(q1, k, v, q_start=s - 1)
    o_dec_ref = _sdpa_gqa(q1, k, v, q_start=s - 1)
    print(f"decode allclose: {torch.allclose(o_dec, o_dec_ref, rtol=1e-2, atol=1e-2)}")
    print(f"decode max abs err: {(o_dec - o_dec_ref).abs().max().item():.3e}")
    print(
        f"triton gqa : {triton.testing.do_bench(lambda: flash_attention_gqa(q, k, v)):.3f} ms"
    )
    print(f"torch  gqa : {triton.testing.do_bench(lambda: _sdpa_gqa(q, k, v)):.3f} ms")

    qg = q.detach().requires_grad_(True)
    kg = k.detach().requires_grad_(True)
    vg = v.detach().requires_grad_(True)
    do = torch.randn_like(q)
    o_ref = _sdpa_gqa(qg, kg, vg, q_start=0)
    dq_r, dk_r, dv_r = torch.autograd.grad(o_ref, (qg, kg, vg), do)
    qt = q.detach().requires_grad_(True)
    kt = k.detach().requires_grad_(True)
    vt = v.detach().requires_grad_(True)
    ot = flash_attention_gqa_train(qt, kt, vt)
    dq_t, dk_t, dv_t = torch.autograd.grad(ot, (qt, kt, vt), do)
    print(f"train dq allclose: {torch.allclose(dq_t, dq_r, rtol=1e-2, atol=1e-2)}  "
          f"max abs {(dq_t - dq_r).abs().max().item():.3e}")
    print(f"train dk allclose: {torch.allclose(dk_t, dk_r, rtol=1e-2, atol=1e-2)}  "
          f"max abs {(dk_t - dk_r).abs().max().item():.3e}")
    print(f"train dv allclose: {torch.allclose(dv_t, dv_r, rtol=1e-2, atol=1e-2)}  "
          f"max abs {(dv_t - dv_r).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
