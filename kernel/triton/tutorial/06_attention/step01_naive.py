"""Naive 单头 Attention（Host 组合已有算子）

公式:
    S = Q @ K^T / sqrt(d)
    P = softmax(S)          # 对最后一维
    O = P @ V

说明:
    - 在 Python 里依次调用 matmul / softmax，不要在 @triton.jit 里调 naive_softmax
    - 会物化完整 S、P，显存 O(S^2)，这是 naive 相对 Flash 的代价
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
from step01_naive import naive_softmax  # noqa: E402
from step02_online import online_softmax


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    use_triton_matmul: bool = True,
    use_triton_online_softmax: bool = True
) -> torch.Tensor:
    """
    单头 Attention。
    q, k, v: (B, S, D)
    返回: (B, S, D)
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape
    assert q.ndim == 3

    d = q.shape[-1]
    scale = d**-0.5

    # S = Q K^T / sqrt(d)  → (B, S, S)
    k_t = k.transpose(-1, -2).contiguous()
    if use_triton_matmul:
        scores = triton_matmul(q, k_t) * scale
    else:
        scores = torch.matmul(q, k_t) * scale

    # P = softmax(S)
    if use_triton_online_softmax:
        attn = online_softmax(scores)
    else:
        attn = naive_softmax(scores)
    

    # O = P V  → (B, S, D)
    if use_triton_matmul:
        out = triton_matmul(attn, v)
    else:
        out = torch.matmul(attn, v)
    return out


def attention_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) * (d**-0.5)
    attn = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")

    torch.manual_seed(0)
    # S 不要太大：naive 要物化 (B,S,S)；softmax 入门版也要求整行能装进 BLOCK
    B, S, D = 16, 128, 2048
    q = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, S, D, device="cuda", dtype=torch.float32)

    o_triton = naive_attention(q, k, v, use_triton_matmul=True)
    o_ref = attention_torch(q, k, v)

    # matmul 可能走 TF32，放宽一点
    ok = torch.allclose(o_ref, o_triton, rtol=5e-2, atol=5e-2)
    print(f"allclose: {ok}")
    print(f"max abs err: {(o_ref - o_triton).abs().max().item():.3e}")
    print(f"shape: QKV{(B, S, D)} → O{(B, S, D)}, scores{(B, S, S)}")

    print(
        f"torch  attn : {triton.testing.do_bench(lambda: attention_torch(q, k, v)):.3f} ms"
    )
    print(
        f"triton attn : {triton.testing.do_bench(lambda: naive_attention(q, k, v)):.3f} ms"
    )


if __name__ == "__main__":
    main()
