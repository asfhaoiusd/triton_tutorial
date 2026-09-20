"""torch.profiler 对照（本机 CUPTI 常失败，只剩 CPU 调用链）。

5070 Laptop + WSL 上优先用 nsys：
  bash kernel/triton/tutorial/07_profiling/run_nsys.sh

本脚本保留：看 aten 拆成几步（matmul / softmax / empty）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

tutorial = Path(__file__).resolve().parents[1]
sys.path.append(str(tutorial / "06_attention"))

from step03_flash_single import flash_atten_fwd  # noqa: E402
from step01_naive import attention_torch, naive_attention  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def _warmup(fn, *args, n: int = 8):
    for _ in range(n):
        fn(*args)
    torch.cuda.synchronize()


def _run_profiler(name: str, fn, *args, loops: int = 20):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with record_function(name):
            for _ in range(loops):
                fn(*args)
            torch.cuda.synchronize()

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    trace_path = OUT_DIR / f"{name}.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"chrome trace → {trace_path}")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU")

    torch.manual_seed(0)
    # S 不要太大：naive 会物化 (B,S,S)
    B, S, D = 4, 256, 64
    q = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, S, D, device="cuda", dtype=torch.float32)

    print(f"shape QKV{(B, S, D)}  naive 会物化 scores{(B, S, S)}")

    _warmup(attention_torch, q, k, v)
    _warmup(naive_attention, q, k, v)
    _warmup(flash_atten_fwd, q, k, v)

    _run_profiler("torch_attn", attention_torch, q, k, v)
    _run_profiler("naive_attn", naive_attention, q, k, v)
    _run_profiler("flash_attn", flash_atten_fwd, q, k, v)

    print("\n对照时问自己:")
    print("  - naive 是否出现多次独立 kernel（matmul / softmax / matmul）?")
    print("  - flash 是否基本是一次（或很少）自定义 kernel?")
    print("  - 把观察记进 NOTES.md")


if __name__ == "__main__":
    main()
