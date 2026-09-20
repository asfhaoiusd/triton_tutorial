"""混合精度训练一步：随机 token → CE。"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from kernels import set_backend, set_triton_ops
from model import CONFIGS, LlamaLike, TrainConfig
from tiles import add_tile_args, apply_from_args


def memory_bill(cfg: TrainConfig) -> str:
    n = cfg.n_params()
    return (
        f"全参 AdamW 粗账（{n/1e9:.2f}B params）:\n"
        f"  权重 fp32 {n*4/1e9:.2f} GB；若改 bf16 则 {n*2/1e9:.2f} GB\n"
        f"  Adam m/v 常仍是 fp32：{n*8/1e9:.2f} GB（autocast 几乎不减这项）\n"
        f"  梯度约 {n*2/1e9:.2f}–{n*4/1e9:.2f} GB，再加激活（随 seq/layer）\n"
        "  因此 8GB 默认跑 small；1b 失败时用更短 seq / 8-bit Adam，不算课程失败。"
    )


def make_model(
    cfg_name: str,
    backend: str,
    device: str,
    *,
    grad_ckpt: bool = True,
    compile_mode: str = "reduce-overhead",
) -> LlamaLike:
    cfg = CONFIGS[cfg_name]
    model = LlamaLike(cfg, backend="eager", grad_ckpt=grad_ckpt).to(device)
    if backend == "triton":
        set_backend(model, "triton")
    if backend == "compile":
        try:
            model = torch.compile(model, mode=compile_mode)
        except Exception as e:
            if compile_mode != "default":
                print(f"compile {compile_mode} failed ({e}); fallback mode=default")
                model = torch.compile(model, mode="default")
            else:
                raise
    return model


def train_step(model: LlamaLike, batch: torch.Tensor, opt: torch.optim.Optimizer) -> float:
    model.train()
    opt.zero_grad(set_to_none=True)
    x, y = batch[:, :-1], batch[:, 1:]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()
    opt.step()
    return float(loss.detach())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=tuple(CONFIGS), default="small")
    p.add_argument("--backend", choices=("eager", "compile", "triton"), default="eager")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--triton-ops",
        default="fa",
        help="triton 换哪些核：fa（默认）或 fa,ln,swiglu。MLP 教学核打不过 cuBLAS。",
    )
    add_tile_args(p)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")
    set_triton_ops(args.triton_ops)
    tiles = apply_from_args(args)
    cfg = CONFIGS[args.config]
    print(f"config={cfg.name} params≈{cfg.n_params()/1e6:.1f}M seq={cfg.seq}")
    print(f"tiles={tiles}")
    print(memory_bill(cfg))
    try:
        torch.manual_seed(0)
        model = make_model(args.config, args.backend, "cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for i in range(args.steps):
            batch = torch.randint(0, cfg.vocab_size, (1, cfg.seq), device="cuda")
            loss = train_step(model, batch, opt)
            print(f"step {i}  loss={loss:.4f}  backend={args.backend}")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("OOM。混合精度减的是激活和 matmul，不是 Adam 状态。")
        print(memory_bill(cfg))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
