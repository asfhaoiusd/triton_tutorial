"""RoPE（Rotary Position Embedding），推理可 import。

对最后一维按偶数下标两两旋转：
    (x0, x1) → (x0 cos - x1 sin, x0 sin + x1 cos)
cos/sin 由位置 pos 与频率 10000^{-i/(D/2)} 外积得到。
Qwen 系常见只旋转前 `rope_dim` 维（其余原样拼回）。
"""

from __future__ import annotations

import torch
import triton


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x:(..., D) 偶数维；cos/sin:(..., D/2) 可广播。"""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)


def build_rope_cache(
    seq_len: int,
    dim: int,
    device,
    dtype,
    *,
    base: float = 10000.0,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin 形状 (seq_len, dim/2)。offset 用于 decode 从 cache 长度接着转。"""
    assert dim % 2 == 0
    half = dim // 2
    pos = torch.arange(offset, offset + seq_len, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope_offset(
    x: torch.Tensor,
    *,
    rope_dim: int | None = None,
    offset: int = 0,
    base: float = 10000.0,
) -> torch.Tensor:
    """x:(..., S, D)。只旋转前 rope_dim 维（默认整段 D）。"""
    d = x.shape[-1]
    rd = d if rope_dim is None else rope_dim
    assert rd % 2 == 0 and rd <= d
    s = x.shape[-2]
    cos, sin = build_rope_cache(s, rd, x.device, x.dtype, base=base, offset=offset)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    rot, pass_through = x[..., :rd], x[..., rd:]
    y = apply_rope(rot, cos, sin)
    if pass_through.numel() == 0:
        return y
    return torch.cat((y, pass_through), dim=-1)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 才能运行 Triton 示例")
    torch.manual_seed(0)
    b, h, s, d = 2, 4, 16, 32
    x = torch.randn(b, h, s, d, device="cuda")
    y = apply_rope_offset(x, rope_dim=16, offset=0)
    # 旋转是正交变换：每对 (x0,x1) 模长不变
    x1, x2 = x[..., :16:2], x[..., 1:16:2]
    y1, y2 = y[..., :16:2], y[..., 1:16:2]
    n0 = (x1.square() + x2.square()).sqrt()
    n1 = (y1.square() + y2.square()).sqrt()
    print(f"norm preserved: {torch.allclose(n0, n1, rtol=1e-4, atol=1e-4)}")
    print(f"tail unchanged: {torch.equal(x[..., 16:], y[..., 16:])}")
    print(f"decode offset: {apply_rope_offset(x[:, :, :1], offset=15).shape}")
    print(f"torch rope     : {triton.testing.do_bench(lambda: apply_rope_offset(x)):.3f} ms")


if __name__ == "__main__":
    main()
