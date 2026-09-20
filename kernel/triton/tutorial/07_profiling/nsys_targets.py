"""给 nsys 包一层：一次进程只跑一种实现。

先 warmup（把 Triton autotune 跑完），再用 cudaProfiler 圈出正式循环。
run_nsys.sh 必须带 --capture-range=cudaProfilerApi，报告里才不含 autotune。

  source .venv/bin/activate
  python nsys_targets.py --impl flash
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.cuda import profiler as cuda_profiler

tutorial = Path(__file__).resolve().parents[1]
sys.path.append(str(tutorial / "06_attention"))

from step03_flash_single import flash_atten_fwd  # noqa: E402
from step01_naive import attention_torch, naive_attention  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impl",
        choices=("torch", "naive", "flash"),
        required=True,
    )
    parser.add_argument("--loops", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--b", type=int, default=4)
    parser.add_argument("--s", type=int, default=256)
    parser.add_argument(
        "--d",
        type=int,
        default=256,
        help="head dim。教学 flash 整段 D 在 SRAM，本机请用 64（512 会 OOM）",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU")

    torch.manual_seed(0)
    B, S, D = args.b, args.s, args.d
    q = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, S, D, device="cuda", dtype=torch.float32)

    fn = {
        "torch": attention_torch,
        "naive": naive_attention,
        "flash": flash_atten_fwd,
    }[args.impl]

    # autotune / 编译只发生在这里，nsys 窗口还没开
    for _ in range(args.warmup):
        fn(q, k, v)
    torch.cuda.synchronize()

    cuda_profiler.start()
    for _ in range(args.loops):
        fn(q, k, v)
    torch.cuda.synchronize()
    cuda_profiler.stop()

    meta = Path(__file__).resolve().parent / "nsys_out" / "meta.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(
        f'{{"B": {B}, "S": {S}, "D": {D}, "loops": {args.loops}, "warmup": {args.warmup}}}\n',
        encoding="utf-8",
    )
    print(f"done impl={args.impl} loops={args.loops} shape={(B, S, D)} (capture excludes warmup/autotune)")


if __name__ == "__main__":
    main()
