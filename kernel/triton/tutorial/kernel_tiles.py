"""Triton BLOCK 覆盖：llm_train 实验用。None 表示走文件里的 autotune / 默认 tile。"""

from __future__ import annotations

_OVERRIDE: dict[str, int] | None = None
_LAUNCH_KEYS = ("num_warps", "num_stages")


def set_tile_override(d: dict[str, int] | None) -> None:
    global _OVERRIDE
    _OVERRIDE = None if not d else dict(d)


def tiles() -> dict[str, int] | None:
    return _OVERRIDE


def launch(kernel, grid, *args, block_kwargs: dict | None = None, **kwargs):
    """block_kwargs 为 None 时走 autotune；否则固定 BLOCK_* / num_warps / num_stages。

    显式 launch 必须带 num_stages：JIT 默认 3 级流水会把 smem 乘上去，
    FA 64×64 在上限 101376 的卡上会 OutOfResources。
    """
    if not block_kwargs:
        kernel[grid](*args, **kwargs)
        return
    runner = kernel.fn if hasattr(kernel, "configs") and hasattr(kernel, "fn") else kernel
    hints = {k: block_kwargs[k] for k in _LAUNCH_KEYS if k in block_kwargs and block_kwargs[k] is not None}
    cs = {k: v for k, v in block_kwargs.items() if k not in _LAUNCH_KEYS}
    runner[grid](*args, **kwargs, **cs, **hints)
