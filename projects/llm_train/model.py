"""Decoder：LayerNorm + RoPE + GQA + SwiGLU。

triton 后端默认只换 FA Function（GQA 核内 GROUP，不 repeat KV）。
LN / SwiGLU 仍走 PyTorch；要用教学核时设 --triton-ops fa,ln,swiglu。
down 投影始终是 nn.Linear。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from kernels import LayerNorm, SwiGLUMLP, flash_attention_gqa_train, set_backend
from step05_rope import apply_rope_offset


@dataclass
class TrainConfig:
    name: str
    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    n_kv: int
    head_dim: int
    ffn: int
    seq: int
    eps: float = 1e-5

    def n_params(self) -> int:
        d, h, kv, hd, ff, v, L = (
            self.dim,
            self.n_heads,
            self.n_kv,
            self.head_dim,
            self.ffn,
            self.vocab_size,
            self.n_layers,
        )
        attn = d * (h * hd) + 2 * d * (kv * hd) + (h * hd) * d
        mlp = 3 * d * ff
        # 每层 2 个 LN（γ+β）+ 末尾 LN
        return v * d + L * (attn + mlp + 4 * d) + 2 * d + v * d


CONFIGS = {
    "small": TrainConfig(
        name="small",
        vocab_size=8192,
        dim=2048,
        n_layers=6,
        n_heads=16,
        n_kv=4,
        head_dim=128,
        ffn=4096,
        seq=256,
    ),
    "1b": TrainConfig(
        name="1b",
        vocab_size=32000,
        dim=2048,
        n_layers=22,
        n_heads=32,
        n_kv=8,
        head_dim=64,
        ffn=5632,
        seq=256,
    ),
}


class GQAAttention(nn.Module):
    def __init__(self, cfg: TrainConfig, backend: str):
        super().__init__()
        self.cfg = cfg
        self.backend = backend
        d, hq, hkv, hd = cfg.dim, cfg.n_heads, cfg.n_kv, cfg.head_dim
        if d != hq * hd:
            raise ValueError(f"dim={d} 必须等于 n_heads*head_dim={hq*hd}")
        if hq % hkv != 0:
            raise ValueError("n_heads 必须能被 n_kv 整除")
        self.wq = nn.Linear(d, hq * hd, bias=False)
        self.wk = nn.Linear(d, hkv * hd, bias=False)
        self.wv = nn.Linear(d, hkv * hd, bias=False)
        self.wo = nn.Linear(hq * hd, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        cfg = self.cfg
        q = self.wq(x).view(b, s, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, cfg.n_kv, cfg.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, cfg.n_kv, cfg.head_dim).transpose(1, 2)
        q = apply_rope_offset(q).contiguous()
        k = apply_rope_offset(k).contiguous()
        v = v.contiguous()
        if self.backend == "triton":
            o = flash_attention_gqa_train(q, k, v)
        else:
            try:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            except TypeError:
                g = cfg.n_heads // cfg.n_kv
                o = F.scaled_dot_product_attention(
                    q,
                    k.repeat_interleave(g, dim=1),
                    v.repeat_interleave(g, dim=1),
                    is_causal=True,
                )
        o = o.transpose(1, 2).contiguous().view(b, s, cfg.dim)
        return self.wo(o)


class Block(nn.Module):
    def __init__(self, cfg: TrainConfig, backend: str):
        super().__init__()
        self.n1 = LayerNorm(cfg.dim, cfg.eps, backend)
        self.attn = GQAAttention(cfg, backend)
        self.n2 = LayerNorm(cfg.dim, cfg.eps, backend)
        self.mlp = SwiGLUMLP(cfg.dim, cfg.ffn, backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class LlamaLike(nn.Module):
    def __init__(self, cfg: TrainConfig, backend: str = "eager", grad_ckpt: bool = True):
        super().__init__()
        self.cfg = cfg
        self.backend = backend
        self.grad_ckpt = grad_ckpt
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg, backend) for _ in range(cfg.n_layers))
        self.norm = LayerNorm(cfg.dim, cfg.eps, backend)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.lm_head(self.norm(x))


__all__ = ["TrainConfig", "CONFIGS", "LlamaLike", "set_backend"]
