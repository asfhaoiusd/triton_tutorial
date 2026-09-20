"""玩具混合模型：DeltaNet 占位 + GQA Attention，用来练 generate / 换核。

布局模仿 Qwen3.5 的「若干 DeltaNet 块 + 一块 Gated Attention」，尺寸缩小以便无权重可跑。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import apply_rope_offset, flash_attention_gqa, rmsnorm, swiglu_pro


@dataclass
class ToyConfig:
    vocab_size: int = 128
    dim: int = 64
    n_groups: int = 1          # 每组：n_delta 个 DeltaNet + 1 个 Attn
    n_delta_per_group: int = 1
    n_heads_q: int = 4
    n_heads_kv: int = 2
    head_dim: int = 16
    mlp_hidden: int = 128
    eps: float = 1e-6
    max_seq: int = 256


class ToyRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.use_triton = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton:
            return rmsnorm(x, self.weight, self.eps)
        return F.rms_norm(x, (x.shape[-1],), weight=self.weight, eps=self.eps)


class ToySwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Parameter(torch.empty(dim, hidden))
        self.w_up = nn.Parameter(torch.empty(dim, hidden))
        self.w_down = nn.Parameter(torch.empty(hidden, dim))
        nn.init.normal_(self.w_gate, std=0.02)
        nn.init.normal_(self.w_up, std=0.02)
        nn.init.normal_(self.w_down, std=0.02)
        self.use_triton = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton:
            fused = swiglu_pro(x, self.w_gate, self.w_up)
            return fused @ self.w_down
        return (F.silu(x @ self.w_gate) * (x @ self.w_up)) @ self.w_down


class ToyDeltaNet(nn.Module):
    """Gated DeltaNet 的占位：一层 Linear，始终走 PyTorch。"""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def reset_cache(self) -> None:
        return None


class ToyGatedAttention(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.cfg = cfg
        d, hq, hkv, hd = cfg.dim, cfg.n_heads_q, cfg.n_heads_kv, cfg.head_dim
        assert hq * hd == d and hkv * hd < 10**9
        self.wq = nn.Linear(d, hq * hd, bias=False)
        self.wk = nn.Linear(d, hkv * hd, bias=False)
        self.wv = nn.Linear(d, hkv * hd, bias=False)
        self.wo = nn.Linear(hq * hd, d, bias=False)
        self.use_triton = False
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def reset_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None

    def forward(self, x: torch.Tensor, *, use_cache: bool = False) -> torch.Tensor:
        b, s, _ = x.shape
        cfg = self.cfg
        q = self.wq(x).view(b, s, cfg.n_heads_q, cfg.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, cfg.n_heads_kv, cfg.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, cfg.n_heads_kv, cfg.head_dim).transpose(1, 2)
        q_start = 0
        if use_cache and self.k_cache is not None:
            q_start = self.k_cache.shape[2]
        q = apply_rope_offset(q, offset=q_start)
        k = apply_rope_offset(k, offset=q_start)
        if use_cache:
            if self.k_cache is None:
                self.k_cache, self.v_cache = k, v
            else:
                self.k_cache = torch.cat((self.k_cache, k), dim=2)
                self.v_cache = torch.cat((self.v_cache, v), dim=2)
            k, v = self.k_cache, self.v_cache
        if self.use_triton:
            o = flash_attention_gqa(q, k, v, q_start=q_start)
        elif q_start == 0 and q.shape[2] == k.shape[2]:
            try:
                o = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True, enable_gqa=True
                )
            except TypeError:
                g = cfg.n_heads_q // cfg.n_heads_kv
                o = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k.repeat_interleave(g, dim=1),
                    v.repeat_interleave(g, dim=1),
                    is_causal=True,
                )
        else:
            g = cfg.n_heads_q // cfg.n_heads_kv
            k_rep = k.repeat_interleave(g, dim=1)
            v_rep = v.repeat_interleave(g, dim=1)
            scale = cfg.head_dim**-0.5
            attn = torch.matmul(q.float(), k_rep.transpose(-2, -1).float()) * scale
            q_pos = q_start + torch.arange(s, device=x.device)
            k_pos = torch.arange(k.shape[2], device=x.device)
            attn = attn.masked_fill(
                k_pos[None, None, None, :] > q_pos[None, None, :, None], float("-inf")
            )
            o = torch.matmul(torch.softmax(attn, dim=-1), v_rep.float()).to(q.dtype)
        o = o.transpose(1, 2).contiguous().view(b, s, cfg.dim)
        return self.wo(o)


class ToyBlock(nn.Module):
    def __init__(self, cfg: ToyConfig, kind: str):
        super().__init__()
        self.kind = kind
        self.n1 = ToyRMSNorm(cfg.dim, cfg.eps)
        self.n2 = ToyRMSNorm(cfg.dim, cfg.eps)
        self.mix = ToyDeltaNet(cfg.dim) if kind == "delta" else ToyGatedAttention(cfg)
        self.mlp = ToySwiGLU(cfg.dim, cfg.mlp_hidden)

    def forward(self, x: torch.Tensor, *, use_cache: bool = False) -> torch.Tensor:
        h = self.n1(x)
        if self.kind == "delta":
            h = self.mix(h)
        else:
            h = self.mix(h, use_cache=use_cache)
        x = x + h
        return x + self.mlp(self.n2(x))

    def reset_cache(self) -> None:
        self.mix.reset_cache()


class ToyLM(nn.Module):
    def __init__(self, cfg: ToyConfig | None = None):
        super().__init__()
        self.cfg = cfg or ToyConfig()
        c = self.cfg
        self.embed = nn.Embedding(c.vocab_size, c.dim)
        blocks: list[ToyBlock] = []
        for _ in range(c.n_groups):
            for _ in range(c.n_delta_per_group):
                blocks.append(ToyBlock(c, "delta"))
            blocks.append(ToyBlock(c, "attn"))
        self.blocks = nn.ModuleList(blocks)
        self.norm = ToyRMSNorm(c.dim, c.eps)
        self.lm_head = nn.Linear(c.dim, c.vocab_size, bias=False)

    def set_kernels(self, use_triton: bool) -> None:
        for m in self.modules():
            if hasattr(m, "use_triton"):
                m.use_triton = use_triton

    def reset_cache(self) -> None:
        for b in self.blocks:
            b.reset_cache()

    def describe(self) -> str:
        kinds = [b.kind for b in self.blocks]
        c = self.cfg
        return (
            f"{len(kinds)} blocks {kinds}  "
            f"(real 0.8B is 6×(3×delta + 1×attn); toy is shrunk)  "
            f"GQA {c.n_heads_q}q/{c.n_heads_kv}kv D={c.head_dim}"
        )

    def forward(self, input_ids: torch.Tensor, *, use_cache: bool = False) -> torch.Tensor:
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x, use_cache=use_cache)
        return self.lm_head(self.norm(x))
