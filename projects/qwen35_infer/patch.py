"""把 HF 模块换成教学 Triton 核。DeltaNet / 未知 Attention 原样留下。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import flash_attention_gqa, rmsnorm, swiglu_pro

_RMS_NAMES = {
    "RMSNorm",
    "Qwen2RMSNorm",
    "Qwen3RMSNorm",
    "Qwen3_5RMSNorm",
    "Qwen3NextRMSNorm",
    "LlamaRMSNorm",
}
# Qwen3.5 的 RMSNorm 是 Gemma 式：y = (1+w) * x / rms，初始化 w=0。
_ONE_CENTERED_RMS = {"Qwen3_5RMSNorm", "Qwen3NextRMSNorm"}


class TritonRMSNorm(nn.Module):
    def __init__(self, orig: nn.Module):
        super().__init__()
        self.weight = orig.weight
        self.eps = getattr(orig, "variance_epsilon", getattr(orig, "eps", 1e-6))
        self.one_centered = type(orig).__name__ in _ONE_CENTERED_RMS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.one_centered:
            w = (1.0 + w).to(dtype=x.dtype)
        return rmsnorm(x, w, float(self.eps))


class TritonSwiGLUMLP(nn.Module):
    def __init__(self, orig: nn.Module):
        super().__init__()
        self.gate_proj = orig.gate_proj
        self.up_proj = orig.up_proj
        self.down_proj = orig.down_proj
        self.act_fn = getattr(orig, "act_fn", F.silu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HF Linear 是 y = x @ W.T；教学 swiglu_pro 要 (K,N) = W.T
        w_gate = self.gate_proj.weight.t().contiguous()
        w_up = self.up_proj.weight.t().contiguous()
        fused = swiglu_pro(x if x.ndim == 3 else x.unsqueeze(0), w_gate, w_up)
        if x.ndim == 2:
            fused = fused.squeeze(0)
        return self.down_proj(fused)


def _triton_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    if (
        dropout_p == 0.0
        and attn_mask is None
        and q.is_cuda
        and q.ndim == 4
    ):
        q_start = 0
        if (not is_causal) and q.shape[2] != k.shape[2]:
            q_start = k.shape[2] - q.shape[2]
        try:
            return flash_attention_gqa(q, k, v, q_start=q_start, causal=True)
        except Exception:
            pass
    return _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)


_ORIG_SDPA = F.scaled_dot_product_attention
_SDPA_HOOKED = False
_ATTN_HOOKED = False
_ORIG_ATTN: dict = {}


def _triton_attn_interface(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs):
    """transformers ALL_ATTENTION_FUNCTIONS：输入 (B,H,S,D)，返回必须是 (B,S,H,D)。"""
    drop = float(dropout or 0.0)
    if drop == 0.0 and query.is_cuda and query.ndim == 4 and key.ndim == 4:
        q_start = 0
        if query.shape[2] != key.shape[2]:
            q_start = int(key.shape[2] - query.shape[2])
        try:
            o = flash_attention_gqa(query, key, value, q_start=q_start, causal=True)
            return o.transpose(1, 2).contiguous(), None
        except Exception:
            pass
    impl = _ORIG_ATTN.get("sdpa") or _ORIG_ATTN.get("eager")
    if impl is not None:
        return impl(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    o = _ORIG_SDPA(query, key, value, attn_mask=attention_mask, dropout_p=drop, is_causal=True)
    return o.transpose(1, 2).contiguous(), None


def _hook_attention_functions() -> int:
    """transformers 5 的 Gated Attention 走 ALL_ATTENTION_FUNCTIONS.get_interface，不一定调用 F.sdpa。"""
    global _ATTN_HOOKED
    if _ATTN_HOOKED:
        return 0
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return 0
    for key in ("sdpa", "eager"):
        try:
            _ORIG_ATTN[key] = ALL_ATTENTION_FUNCTIONS.get(key)
        except Exception:
            _ORIG_ATTN[key] = None
        ALL_ATTENTION_FUNCTIONS[key] = _triton_attn_interface
    try:
        ALL_ATTENTION_FUNCTIONS.register("triton_gqa", _triton_attn_interface)
    except Exception:
        ALL_ATTENTION_FUNCTIONS["triton_gqa"] = _triton_attn_interface
    _ATTN_HOOKED = True
    return 1


def patch_model(model: nn.Module, *, rmsnorm_on: bool = True, mlp_on: bool = True, attn_on: bool = True) -> dict[str, int]:
    """就地替换。返回替换计数。Gated DeltaNet 不会走 SDPA / 注意力接口，钩子碰不到它们。"""
    counts = {"rmsnorm": 0, "mlp": 0, "sdpa_hook": 0, "attn_fn_hook": 0}
    if rmsnorm_on:
        for name, child in list(model.named_modules()):
            if type(child).__name__ in _RMS_NAMES and name:
                parent_name, _, attr = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, attr, TritonRMSNorm(child))
                counts["rmsnorm"] += 1
    if mlp_on:
        for name, child in list(model.named_modules()):
            if hasattr(child, "gate_proj") and hasattr(child, "up_proj") and hasattr(child, "down_proj") and name:
                parent_name, _, attr = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, attr, TritonSwiGLUMLP(child))
                counts["mlp"] += 1
    global _SDPA_HOOKED
    if attn_on and not _SDPA_HOOKED:
        F.scaled_dot_product_attention = _triton_sdpa  # type: ignore[assignment]
        _SDPA_HOOKED = True
        counts["sdpa_hook"] = 1
    if attn_on:
        counts["attn_fn_hook"] = _hook_attention_functions()
        cfg = getattr(model, "config", None)
        text_cfg = getattr(cfg, "text_config", cfg) if cfg is not None else None
        if text_cfg is not None and hasattr(text_cfg, "_attn_implementation"):
            text_cfg._attn_implementation = "sdpa"
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "sdpa"
    return counts


def unhook_sdpa() -> None:
    global _SDPA_HOOKED, _ATTN_HOOKED
    F.scaled_dot_product_attention = _ORIG_SDPA  # type: ignore[assignment]
    _SDPA_HOOKED = False
    if _ATTN_HOOKED:
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            for key, fn in _ORIG_ATTN.items():
                ALL_ATTENTION_FUNCTIONS[key] = fn
        except Exception:
            pass
        _ATTN_HOOKED = False
