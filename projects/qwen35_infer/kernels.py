"""实战用的 Triton Host API 薄封装。"""

from __future__ import annotations

import torch

from paths import TUTORIAL  # noqa: F401  # side effect: sys.path
from step03_rmsnorm import rmsnorm
from step03_swiglu_fused import swiglu_pro
from step06_gqa import flash_attention_gqa
from step05_rope import apply_rope_offset


def swiglu_mlp(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """out = (silu(x@W_gate) * (x@W_up)) @ W_down。x:(B,S,D)。"""
    orig = x.shape
    x3 = x.reshape(orig[0], -1, orig[-1]).contiguous() if x.ndim == 3 else x.unsqueeze(0)
    fused = swiglu_pro(x3, w_gate, w_up)
    y = fused @ w_down
    return y.reshape(orig) if x.ndim == 3 else y.squeeze(0)


__all__ = [
    "rmsnorm",
    "swiglu_pro",
    "swiglu_mlp",
    "flash_attention_gqa",
    "apply_rope_offset",
]
