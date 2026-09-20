"""训练用 Triton 包装：必须走 autograd.Function，不能调裸 fwd kernel。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import paths  # noqa: F401  # tutorial 课目录进 sys.path

from step01_layernorm import TritonLayerNorm
from step03_swiglu_fused import TritonSwiGLU
from step06_gqa import flash_attention_gqa_train

__all__ = [
    "TritonLayerNorm",
    "TritonSwiGLU",
    "flash_attention_gqa_train",
    "LayerNorm",
    "SwiGLUMLP",
    "set_backend",
    "set_triton_ops",
]


_TRITON_EXTRA: set[str] = set()


def set_triton_ops(spec: str) -> None:
    """triton 后端默认只换 FA。大 GEMM 的教学 SwiGLU/LN 会负优化，需要时再开。"""
    global _TRITON_EXTRA
    parts = {p.strip() for p in spec.split(",") if p.strip()}
    unknown = parts - {"fa", "ln", "swiglu"}
    if unknown:
        raise ValueError(f"未知 triton op {unknown}，只能是 fa,ln,swiglu")
    _TRITON_EXTRA = parts - {"fa"}


class LayerNorm(nn.Module):
    """backend=triton 且 --triton-ops 含 ln 时走 TritonLayerNorm；否则 F.layer_norm。"""

    def __init__(self, dim: int, eps: float = 1e-5, backend: str = "eager"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "triton" and "ln" in _TRITON_EXTRA:
            return TritonLayerNorm.apply(x, self.weight, self.bias, self.eps)
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class SwiGLUMLP(nn.Module):
    """gate/up 权重布局 (K, N)，与 swiglu_pro 一致；down 仍是 nn.Linear。"""

    def __init__(self, dim: int, hidden: int, backend: str = "eager"):
        super().__init__()
        self.backend = backend
        self.w_gate = nn.Parameter(torch.empty(dim, hidden))
        self.w_up = nn.Parameter(torch.empty(dim, hidden))
        self.down = nn.Linear(hidden, dim, bias=False)
        nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_up, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "triton" and "swiglu" in _TRITON_EXTRA:
            h = TritonSwiGLU.apply(x, self.w_gate, self.w_up)
        else:
            h = F.silu(x @ self.w_gate) * (x @ self.w_up)
        return self.down(h)


def set_backend(module: nn.Module, backend: str) -> None:
    """把子模块上的 backend 标成 eager 或 triton（compile 仍用 eager 模块再 torch.compile）。"""
    if backend not in ("eager", "triton"):
        raise ValueError(f"backend 只能是 eager|triton，收到 {backend!r}")
    for m in module.modules():
        if hasattr(m, "backend") and isinstance(getattr(m, "backend", None), str):
            m.backend = backend
