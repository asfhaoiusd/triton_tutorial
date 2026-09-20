"""Naive 多头 Attention（Host 组合已有 3D matmul + softmax）

数据流:
    Q,K,V: (B, S, D) → (B, H, S, d) → 展成 (B*H, S, d)
    S = Q @ K^T / sqrt(d)
    P = softmax(S)
    O = P @ V → (B, S, D)

说明:
    - 你的 triton_matmul 只接 3D，所以用 B*H 当 batch 维
    - 这里不做 QKV 线性投影，只做「分头 + attention」（便于对照）
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


def naive_multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    q,k,v: (B, S, D)，D 能被 num_heads 整除
    返回: (B, S, D)
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape and q.ndim == 3
    assert num_heads > 0

    B, S, D = q.shape
    assert D % num_heads == 0
    d = D // num_heads

    # (B, S, H, d) → (B, H, S, d) → (B*H, S, d)
    def to_bh(x: torch.Tensor) -> torch.Tensor:
        return (
            x.view(B, S, num_heads, d)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B * num_heads, S, d)
        )

    q_h = to_bh(q)
    k_h = to_bh(k)
    v_h = to_bh(v)

    scale = d**-0.5
    k_t = k_h.transpose(-1, -2).contiguous()  # (B*H, d, S)

    if use_triton:
        scores = triton_matmul(q_h, k_t) * scale
        attn = online_softmax(scores)
        out_h = triton_matmul(attn, v_h)
    else:
        scores = torch.matmul(q_h, k_t) * scale
        attn = torch.nn.functional.softmax(scores, dim=-1)
        out_h = torch.matmul(attn, v_h)

    # (B*H, S, d) → (B, S, D)
    out = (
        out_h.view(B, num_heads, S, d)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(B, S, D)
    )
    return out


def multihead_attention_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """官方 SDPA 对照：同样不做 in_proj / out_proj，只分头做 attention。"""
    B, S, D = q.shape
    d = D // num_heads

    def to_bhsd(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, S, num_heads, d).permute(0, 2, 1, 3).contiguous()

    out = torch.nn.functional.scaled_dot_product_attention(
        to_bhsd(q), to_bhsd(k), to_bhsd(v)
    )
    return out.permute(0, 2, 1, 3).contiguous().view(B, S, D)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    B, S, D, num_heads = 8, 512, 1024, 8
    q = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, S, D, device="cuda", dtype=torch.float32)

    o_triton = naive_multihead_attention(q, k, v, num_heads, use_triton=True)
    o_torch = multihead_attention_sdpa(q, k, v, num_heads)

    ok = torch.allclose(o_torch, o_triton, rtol=5e-2, atol=5e-2)
    print(f"allclose: {ok}")
    print(f"max abs err: {(o_torch - o_triton).abs().max().item():.3e}")
    print(f"shape: QKV{(B, S, D)}, heads={num_heads}, head_dim={D // num_heads}")

    print(
        f"torch sdpa : {triton.testing.do_bench(lambda: multihead_attention_sdpa(q, k, v, num_heads)):.3f} ms"
    )
    print(
        f"triton     : {triton.testing.do_bench(lambda: naive_multihead_attention(q, k, v, num_heads)):.3f} ms"
    )


if __name__ == "__main__":
    main()
