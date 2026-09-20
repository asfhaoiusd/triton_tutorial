"""对比 eager / torch.compile / Triton 的训练速度。

默认三个后端各跑一遍：每 step 打 ms、tok/s；warmup（compile / autotune）不计入平均。
最后一张表相对 eager 的加速比。
"""

from __future__ import annotations

import argparse
import gc
import time

import torch

from kernels import set_triton_ops
from model import CONFIGS
from tiles import add_tile_args, apply_from_args
from train import make_model, memory_bill, train_step

BACKENDS = ("eager", "compile", "triton")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_backend(name: str, cfg_name: str, steps: int, warmup: int, lr: float) -> dict:
    cfg = CONFIGS[cfg_name]
    tok_per_step = cfg.seq - 1
    torch.manual_seed(0)
    print(f"\n=== {name} ===")
    print(f"{'step':>6}  {'loss':>8}  {'ms':>8}  {'tok/s':>8}  {'peak_MB':>8}  note")
    model = make_model(cfg_name, name, "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    torch.cuda.reset_peak_memory_stats()
    timed_ms: list[float] = []
    last_loss = None
    try:
        for i in range(warmup + steps):
            batch = torch.randint(0, cfg.vocab_size, (1, cfg.seq), device="cuda")
            _sync()
            t0 = time.perf_counter()
            last_loss = train_step(model, batch, opt)
            _sync()
            ms = (time.perf_counter() - t0) * 1e3
            warm = i < warmup
            if not warm:
                timed_ms.append(ms)
            tok_s = tok_per_step / (ms / 1e3) if ms > 0 else 0.0
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            print(
                f"{i:6d}  {last_loss:8.4f}  {ms:8.1f}  {tok_s:8.0f}  {peak:8.1f}  "
                f"{'warmup' if warm else ''}"
            )
    except Exception as e:
        if name != "compile":
            raise
        print(f"compile runtime failed ({type(e).__name__}: {e}); retry mode=default")
        del model, opt
        _free()
        torch.manual_seed(0)
        model = make_model(cfg_name, name, "cuda", grad_ckpt=False, compile_mode="default")
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        return run_backend_after_model(name, cfg, model, opt, steps, warmup, tok_per_step)

    peak = torch.cuda.max_memory_allocated() / (1024**2)
    del model, opt
    _free()
    mean_ms = sum(timed_ms) / len(timed_ms) if timed_ms else float("nan")
    tok_s = tok_per_step / (mean_ms / 1e3) if timed_ms and mean_ms > 0 else 0.0
    return {
        "backend": name,
        "loss": last_loss,
        "ms_step": mean_ms,
        "tok_s": tok_s,
        "peak_mb": peak,
    }


def run_backend_after_model(name, cfg, model, opt, steps, warmup, tok_per_step) -> dict:
    torch.cuda.reset_peak_memory_stats()
    timed_ms: list[float] = []
    last_loss = None
    for i in range(warmup + steps):
        batch = torch.randint(0, cfg.vocab_size, (1, cfg.seq), device="cuda")
        _sync()
        t0 = time.perf_counter()
        last_loss = train_step(model, batch, opt)
        _sync()
        ms = (time.perf_counter() - t0) * 1e3
        if i >= warmup:
            timed_ms.append(ms)
        tok_s = tok_per_step / (ms / 1e3) if ms > 0 else 0.0
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        print(
            f"{i:6d}  {last_loss:8.4f}  {ms:8.1f}  {tok_s:8.0f}  {peak:8.1f}  "
            f"{'warmup' if i < warmup else ''}"
        )
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    del model, opt
    _free()
    mean_ms = sum(timed_ms) / len(timed_ms) if timed_ms else float("nan")
    return {
        "backend": name,
        "loss": last_loss,
        "ms_step": mean_ms,
        "tok_s": tok_per_step / (mean_ms / 1e3) if timed_ms and mean_ms > 0 else 0.0,
        "peak_mb": peak,
    }


def print_table(rows: list[dict]) -> None:
    if not rows:
        return
    eager = next((r for r in rows if r["backend"] == "eager"), rows[0])
    base = eager["ms_step"] if eager["ms_step"] and eager["ms_step"] == eager["ms_step"] else None
    print("\n=== 对照（warmup 之后平均）===")
    print(f"{'backend':<10}  {'ms/step':>10}  {'tok/s':>8}  {'vs eager':>10}  {'peak_MB':>8}  {'loss':>8}")
    for r in rows:
        if base and r["ms_step"] == r["ms_step"] and r["ms_step"] > 0:
            rel = base / r["ms_step"]
            vs = f"{rel:.2f}x"
        else:
            vs = "-"
        print(
            f"{r['backend']:<10}  {r['ms_step']:10.1f}  {r['tok_s']:8.0f}  {vs:>10}  "
            f"{r['peak_mb']:8.1f}  {r['loss']:8.4f}"
        )
    print("vs eager >1 表示更快。compile 常赢端到端。")
    print("默认 --triton-ops fa：只换 FlashAttention，LN/MLP 仍走 cuBLAS。")
    print("教学 LN/SwiGLU 大矩阵会负优化，要对齐课用 --triton-ops fa,ln,swiglu。")
    print("loss 同量级即可，不要求 bit 级一致。")


def main():
    p = argparse.ArgumentParser(description="对比 eager / compile / triton 训练速度")
    p.add_argument("--config", choices=tuple(CONFIGS), default="small")
    p.add_argument(
        "--backends",
        default=",".join(BACKENDS),
        help="逗号分隔，默认 eager,compile,triton",
    )
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--triton-ops", default="fa", help="fa 或 fa,ln,swiglu")
    add_tile_args(p)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")

    names = [s.strip() for s in args.backends.split(",") if s.strip()]
    bad = [n for n in names if n not in BACKENDS]
    if bad:
        raise SystemExit(f"未知 backend {bad}，只能是 {BACKENDS}")

    set_triton_ops(args.triton_ops)
    tiles = apply_from_args(args)
    cfg = CONFIGS[args.config]
    print(
        f"config={cfg.name}  params≈{cfg.n_params()/1e6:.1f}M  seq={cfg.seq}  "
        f"bs=1  tokens/step={cfg.seq - 1}  backends={','.join(names)}  "
        f"triton-ops={args.triton_ops}"
    )
    print(f"tiles={tiles}")
    print(memory_bill(cfg))

    rows = []
    for name in names:
        try:
            rows.append(run_backend(name, args.config, args.steps, args.warmup, args.lr))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{name}: OOM。混合精度减的是激活和 matmul，不是 Adam 状态。")
            print(memory_bill(cfg))
        except Exception as e:
            msg = str(e)
            extra = ""
            if "shared memory" in msg or type(e).__name__ == "OutOfResources":
                extra = (
                    "  这是 kernel tile 太大：降 BLOCK，或把 num_stages 设成 1"
                    "（--fa-num-stages 1 --swiglu-num-stages 1 --ln-block-m 16）。"
                    " 本机 smem 上限 101376。"
                )
            print(f"{name}: failed: {type(e).__name__}: {e}{extra}")
        _free()
    print_table(rows)


if __name__ == "__main__":
    main()
