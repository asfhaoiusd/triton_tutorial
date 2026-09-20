"""Multi-head Latent Attention（MLA，教学版 / Host 组合）

DeepSeek-V2 的核心思想：
  KV cache 不存完整多头 K/V，只存
    c_kv : (B, S, d_c)     # 低秩 KV latent
    k_pe : (B, S, d_r)     # 解耦 RoPE key（各头共享）
  Query 侧拆成
    q_nope : (B, S, H, d_nope)   # 内容
    q_pe   : (B, S, H, d_r)      # RoPE（可已旋转）

两条等价路径:
  1) expand:  c_kv → K_nope / V，再拼 RoPE 做标准 MHA（好懂，显存更大）
  2) absorb:  把 W_uk 吸进 Q，直接对 c_kv 算分（推理友好，少物化 K）

本文件在 Python Host 上组合 matmul + softmax；不在 @triton.jit 里互相调用。
RoPE 假定已在 q_pe / k_pe 上完成（或 demo 里用简单 rotate）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import triton

tutorial = Path(__file__).resolve().parents[1]
sys.path.append(str(tutorial / "04_softmax"))
sys.path.append(str(tutorial / "03_matmul"))

from step04_matmul_3d import (  # noqa: E402
    matrix_multiplication_three_dimension as triton_matmul,
)
from step02_online import online_softmax  # noqa: E402
from step05_rope import apply_rope, build_rope_cache  # noqa: E402

# ---------------------------------------------------------------------------
# 工具：多头 batched matmul / softmax，落到 (B*H, M, K) 再调已有 3D kernel
# ---------------------------------------------------------------------------

def _bh_matmul(a: torch.Tensor, b: torch.Tensor, *, use_triton: bool) -> torch.Tensor:
    """a:(B,H,M,K) @ b:(B,H,K,N) → (B,H,M,N)"""
    B, H, M, K = a.shape
    N = b.shape[-1]
    a3 = a.reshape(B * H, M, K).contiguous()
    b3 = b.reshape(B * H, K, N).contiguous()
    if use_triton:
        out = triton_matmul(a3, b3)
    else:
        out = torch.matmul(a3, b3)
    return out.reshape(B, H, M, N)


def _bh_softmax(x: torch.Tensor, *, use_triton: bool) -> torch.Tensor:
    """x:(B,H,Sq,Sk) 对最后一维 softmax。"""
    if use_triton:
        # online_softmax 入门版只接 2D/3D，先展成 (B*H, Sq, Sk)
        B, H, Sq, Sk = x.shape
        y = online_softmax(x.reshape(B * H, Sq, Sk).contiguous())
        return y.reshape(B, H, Sq, Sk)
    return torch.nn.functional.softmax(x, dim=-1)


def mla_expand(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    c_kv: torch.Tensor,
    k_pe: torch.Tensor,
    w_uk: torch.Tensor,
    w_uv: torch.Tensor,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    q_nope: (B, Sq, H, d_nope)
    q_pe:   (B, Sq, H, d_r)      # 已 RoPE
    c_kv:   (B, Sk, d_c)
    k_pe:   (B, Sk, d_r)         # 已 RoPE，各头共享
    w_uk:   (H, d_c, d_nope)
    w_uv:   (H, d_c, d_v)

    返回: (B, Sq, H, d_v)
    """
    B, Sq, H, d_nope = q_nope.shape
    _, Sk, d_c = c_kv.shape
    d_r = q_pe.shape[-1]
    d_v = w_uv.shape[-1]
    assert q_pe.shape == (B, Sq, H, d_r)
    assert k_pe.shape == (B, Sk, d_r)
    assert w_uk.shape == (H, d_c, d_nope)
    assert w_uv.shape == (H, d_c, d_v)

    # K_nope / V: (B, Sk, H, *)
    k_nope = torch.einsum("btc,hcd->bthd", c_kv, w_uk)
    v = torch.einsum("btc,hcd->bthd", c_kv, w_uv)

    # 拼 RoPE：k_pe 扩到各头
    k_pe_h = k_pe[:, :, None, :].expand(B, Sk, H, d_r)
    q = torch.cat([q_nope, q_pe], dim=-1)          # (B, Sq, H, d_nope+d_r)
    k = torch.cat([k_nope, k_pe_h], dim=-1)        # (B, Sk, H, d_nope+d_r)

    d = d_nope + d_r
    scale = d**-0.5

    # → (B, H, S, D) 做 batched matmul
    q_t = q.permute(0, 2, 1, 3).contiguous()
    k_t = k.permute(0, 2, 1, 3).contiguous()
    v_t = v.permute(0, 2, 1, 3).contiguous()

    scores = _bh_matmul(q_t, k_t.transpose(-1, -2), use_triton=use_triton) * scale
    attn = _bh_softmax(scores, use_triton=use_triton)
    out = _bh_matmul(attn, v_t, use_triton=use_triton)  # (B, H, Sq, d_v)
    return out.permute(0, 2, 1, 3).contiguous()


# ---------------------------------------------------------------------------
# 路径 2: absorb — W_uk 吸进 Q，直接对 c_kv 打分（MLA 推理常用）
# ---------------------------------------------------------------------------

def mla_absorb(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    c_kv: torch.Tensor,
    k_pe: torch.Tensor,
    w_uk: torch.Tensor,
    w_uv: torch.Tensor,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    与 mla_expand 数学等价（在无数值误差时）:
      score = q_nope @ W_uk^T @ c_kv^T + q_pe @ k_pe^T
            = q_c @ c_kv^T + q_pe @ k_pe^T
      其中 q_c = q_nope @ W_uk^T，形状 (B, Sq, H, d_c)

    不物化完整 K_nope；V 仍由 c_kv @ W_uv 得到。
    返回: (B, Sq, H, d_v)
    """
    B, Sq, H, d_nope = q_nope.shape
    _, Sk, d_c = c_kv.shape
    d_r = q_pe.shape[-1]
    d_v = w_uv.shape[-1]
    assert w_uk.shape == (H, d_c, d_nope)
    assert w_uv.shape == (H, d_c, d_v)

    # q_c = q_nope @ W_uk^T  → (B, Sq, H, d_c)
    # w_uk: (H, d_c, d_nope) ⇒ W_uk^T per head: (H, d_nope, d_c)
    q_c = torch.einsum("bshd,hcd->bshc", q_nope, w_uk)

    d = d_nope + d_r
    scale = d**-0.5

    # score_c: (B, H, Sq, Sk) = q_c @ c_kv^T
    q_c_t = q_c.permute(0, 2, 1, 3).contiguous()              # (B,H,Sq,d_c)
    c_kv_h = c_kv[:, None, :, :].expand(B, H, Sk, d_c).contiguous()
    score_c = _bh_matmul(q_c_t, c_kv_h.transpose(-1, -2), use_triton=use_triton)

    # score_r: (B, H, Sq, Sk) = q_pe @ k_pe^T（k_pe 各头共享）
    q_pe_t = q_pe.permute(0, 2, 1, 3).contiguous()
    k_pe_h = k_pe[:, None, :, :].expand(B, H, Sk, d_r).contiguous()
    score_r = _bh_matmul(q_pe_t, k_pe_h.transpose(-1, -2), use_triton=use_triton)

    scores = (score_c + score_r) * scale
    attn = _bh_softmax(scores, use_triton=use_triton)

    # V = c_kv @ W_uv → (B, Sk, H, d_v)
    v = torch.einsum("btc,hcd->bthd", c_kv, w_uv)
    v_t = v.permute(0, 2, 1, 3).contiguous()
    out = _bh_matmul(attn, v_t, use_triton=use_triton)
    return out.permute(0, 2, 1, 3).contiguous()


def mla_torch_ref(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    c_kv: torch.Tensor,
    k_pe: torch.Tensor,
    w_uk: torch.Tensor,
    w_uv: torch.Tensor,
) -> torch.Tensor:
    """纯 PyTorch expand 参考实现。"""
    return mla_expand(
        q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, use_triton=False
    )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False

    # 教学用小 shape（真实 DeepSeek: d_c=512, d_r=64, H=128 量级）
    B, Sq, Sk = 2, 64, 64
    H = 8
    d_c, d_nope, d_r, d_v = 128, 64, 32, 64

    q_nope = torch.randn(B, Sq, H, d_nope, device="cuda", dtype=torch.float32)
    q_pe0 = torch.randn(B, Sq, H, d_r, device="cuda", dtype=torch.float32)
    c_kv = torch.randn(B, Sk, d_c, device="cuda", dtype=torch.float32)
    k_pe0 = torch.randn(B, Sk, d_r, device="cuda", dtype=torch.float32)
    # 权重用较小初始化，避免 expand/absorb 结合律差异被放大
    w_uk = torch.randn(H, d_c, d_nope, device="cuda", dtype=torch.float32) * 0.02
    w_uv = torch.randn(H, d_c, d_v, device="cuda", dtype=torch.float32) * 0.02

    cos_q, sin_q = build_rope_cache(Sq, d_r, q_pe0.device, q_pe0.dtype)
    cos_k, sin_k = build_rope_cache(Sk, d_r, k_pe0.device, k_pe0.dtype)
    # cos/sin: (S, d_r/2)
    # q_pe:(B,S,H,D) → 插 head 维；k_pe:(B,S,D) 无 head
    q_pe = apply_rope(q_pe0, cos_q[None, :, None, :], sin_q[None, :, None, :])
    k_pe = apply_rope(k_pe0, cos_k[None, :, :], sin_k[None, :, :])

    o_ref = mla_torch_ref(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv)
    o_absorb_torch = mla_absorb(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, use_triton=False)
    o_expand = mla_expand(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, use_triton=True)
    o_absorb = mla_absorb(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, use_triton=True)

    err_eq = (o_ref - o_absorb_torch).abs().max().item()
    err_e = (o_ref - o_expand).abs().max().item()
    err_a = (o_ref - o_absorb).abs().max().item()
    print(
        f"torch expand vs absorb: {torch.allclose(o_ref, o_absorb_torch, rtol=1e-4, atol=1e-4)}"
        f"  max_err={err_eq:.3e}"
    )
    print(
        f"triton expand vs ref  : {torch.allclose(o_ref, o_expand, rtol=1e-3, atol=1e-3)}"
        f"  max_err={err_e:.3e}"
    )
    print(
        f"triton absorb vs ref  : {torch.allclose(o_ref, o_absorb, rtol=1e-3, atol=1e-3)}"
        f"  max_err={err_a:.3e}"
    )
    print(
        f"shape: q_nope{(B, Sq, H, d_nope)} c_kv{(B, Sk, d_c)} "
        f"→ O{(B, Sq, H, d_v)}; d_attn={d_nope + d_r}"
    )

    print(
        f"torch expand : {triton.testing.do_bench(lambda: mla_expand(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, use_triton=False)):.3f} ms"
    )
    print(
        f"triton expand: {triton.testing.do_bench(lambda: mla_expand(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, use_triton=True)):.3f} ms"
    )
    print(
        f"triton absorb: {triton.testing.do_bench(lambda: mla_absorb(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, use_triton=True)):.3f} ms"
    )


if __name__ == "__main__":
    main()
