"""Triton 算子 tile。改本文件的 DEFAULT，或用命令行覆盖，不必改课里的 .py。

BLOCK 必须是 2 的幂。FA bwd 默认 32×32：64²×head_dim 在本机曾超 smem（约 101376）。
显式 tile 必须带 num_stages（FA 前向默认 1；SwiGLU 默认 2，FA bwd / LN 默认 1）。JIT 默认 3 级流水会把 smem 乘到超限。
eager / compile 忽略这些值。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace

import paths  # noqa: F401

from kernel_tiles import set_tile_override


@dataclass
class KernelTiles:
    # FlashAttention 前向：沿序列的 Q 块 / KV 块
    fa_block_m: int = 64
    fa_block_n: int = 64
    fa_num_warps: int = 4
    fa_num_stages: int = 1
    # FlashAttention 反向（外 KV / 内 Q）
    fa_block_br: int = 32
    fa_block_bc: int = 32
    fa_bwd_num_warps: int = 4
    fa_bwd_num_stages: int = 1
    # 融合 SwiGLU 前向
    swiglu_block_m: int = 64
    swiglu_block_n: int = 64
    swiglu_block_k: int = 64
    swiglu_num_warps: int = 4
    swiglu_num_stages: int = 2
    # SwiGLU 激活反向
    swiglu_bwd_block_m: int = 64
    swiglu_bwd_block_n: int = 64
    swiglu_bwd_num_warps: int = 4
    swiglu_bwd_num_stages: int = 2
    # LayerNorm：一行 N 仍整行装；只改沿 token 的 BLOCK_M
    ln_block_m: int = 16
    ln_num_warps: int = 8
    ln_num_stages: int = 1


# 想换默认 tile 就改这里
DEFAULT = KernelTiles()

_POW2_FIELDS = (
    "fa_block_m",
    "fa_block_n",
    "fa_block_br",
    "fa_block_bc",
    "swiglu_block_m",
    "swiglu_block_n",
    "swiglu_block_k",
    "swiglu_bwd_block_m",
    "swiglu_bwd_block_n",
    "ln_block_m",
)


def _pow2(name: str, n: int) -> None:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"{name}={n} 必须是 2 的幂")


def validate(t: KernelTiles) -> None:
    for name in _POW2_FIELDS:
        _pow2(name, getattr(t, name))
    for name in (
        "fa_num_warps",
        "fa_bwd_num_warps",
        "swiglu_num_warps",
        "swiglu_bwd_num_warps",
        "ln_num_warps",
    ):
        v = getattr(t, name)
        if v not in (1, 2, 4, 8, 16):
            raise ValueError(f"{name}={v} 只能是 1/2/4/8/16")
    for name in (
        "fa_num_stages",
        "fa_bwd_num_stages",
        "swiglu_num_stages",
        "swiglu_bwd_num_stages",
        "ln_num_stages",
    ):
        v = getattr(t, name)
        if v not in (1, 2, 3, 4, 5):
            raise ValueError(f"{name}={v} 只能是 1–5；本机 smem 101376，FA/SwiGLU 建议 1 或 2")
    if t.ln_block_m > 32:
        raise ValueError(
            f"ln_block_m={t.ln_block_m} 太大：LayerNorm 核一次装 BLOCK_M×hidden，"
            "建议 8/16/32，256 会让编译极慢或炸寄存器"
        )


def apply_tiles(t: KernelTiles) -> None:
    validate(t)
    set_tile_override(asdict(t))


def add_tile_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Triton tile（只影响 --backend triton）")
    for f in fields(KernelTiles):
        g.add_argument(f"--{f.name.replace('_', '-')}", type=int, default=None)


def tiles_from_args(args: argparse.Namespace) -> KernelTiles:
    t = DEFAULT
    kw = {}
    for f in fields(KernelTiles):
        v = getattr(args, f.name, None)
        if v is not None:
            kw[f.name] = v
    if kw:
        t = replace(t, **kw)
    return t


def apply_from_args(args: argparse.Namespace) -> KernelTiles:
    t = tiles_from_args(args)
    apply_tiles(t)
    return t
