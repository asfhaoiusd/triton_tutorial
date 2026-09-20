"""eager / torch.compile / Triton(Function) 训练对照。"""

from __future__ import annotations

import argparse
import gc
import time

import torch

from kernels import set_triton_ops
from model import CONFIGS
from tiles import add_tile_args, apply_from_args
from train import make_model, memory_bill, train_step

WHY = """
对照说明
- eager：F.layer_norm + SDPA + silu(x@Wg)*(x@Wu) + Linear down
- compile：同一套 eager + torch.compile
- triton 默认（--triton-ops fa）：只换 FA Function；LN / SwiGLU 仍走 cuBLAS
- --triton-ops fa,ln,swiglu：再换教学 LN / 融合 SwiGLU（大 GEMM 常负优化）
compile 常赢端到端。混合精度不管 Adam 状态：1B 全参在 8GB 上仍容易 OOM。
"""


def _sync():
    torch.cuda.synchronize()


def bench_backend(name: str, cfg_name: str, steps: int, warmup: int) -> dict:
    cfg = CONFIGS[cfg_name]
    torch.manual_seed(0)
    compile_mode = "reduce-overhead"
    grad_ckpt = True
    model = make_model(cfg_name, name, "cuda", grad_ckpt=grad_ckpt, compile_mode=compile_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batches = [
        torch.randint(0, cfg.vocab_size, (1, cfg.seq), device="cuda") for _ in range(warmup + steps)
    ]
    losses = []
    try:
        for i in range(warmup):
            losses.append(train_step(model, batches[i], opt))
    except Exception as e:
        if name != "compile" or compile_mode == "default":
            raise
        print(f"compile {compile_mode} runtime failed ({type(e).__name__}: {e}); retry mode=default")
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(0)
        model = make_model(cfg_name, name, "cuda", grad_ckpt=False, compile_mode="default")
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        losses = [train_step(model, batches[i], opt) for i in range(warmup)]
    torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.perf_counter()
    for i in range(steps):
        losses.append(train_step(model, batches[warmup + i], opt))
    _sync()
    elapsed = time.perf_counter() - t0
    tokens = steps * 1 * (cfg.seq - 1)
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "backend": name,
        "loss_last": losses[-1],
        "tok_s": tokens / elapsed if elapsed > 0 else 0.0,
        "ms_step": elapsed / steps * 1e3,
        "peak_mb": peak,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=tuple(CONFIGS), default="small")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--backends", default="eager,compile,triton")
    p.add_argument("--triton-ops", default="fa", help="fa 或 fa,ln,swiglu")
    add_tile_args(p)
    args = p.parse_args()
    print(WHY)
    set_triton_ops(args.triton_ops)
    tiles = apply_from_args(args)
    cfg = CONFIGS[args.config]
    print(f"config={cfg.name} params≈{cfg.n_params()/1e6:.1f}M seq={cfg.seq} bs=1 bf16 autocast")
    print(f"tiles={tiles}")
    print(memory_bill(cfg))
    rows = []
    for name in args.backends.split(","):
        name = name.strip()
        try:
            row = bench_backend(name, args.config, args.steps, args.warmup)
            rows.append(row)
            print(row)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{name}: OOM。混合精度不管 Adam 状态。")
            print(memory_bill(cfg))
        except Exception as e:
            print(f"{name}: failed: {type(e).__name__}: {e}")
    if len(rows) >= 2:
        print("loss 同量级即可，不要求 bit 级一致（compile / Triton 数值路径不同）。")


if __name__ == "__main__":
    main()
