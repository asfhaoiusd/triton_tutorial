"""简单 greedy generate：prefill 一次，再逐步 decode（KV cache）。"""

from __future__ import annotations

import argparse
import time

import torch

from paths import ROOT  # noqa: F401
from toy_model import ToyLM


@torch.no_grad()
def generate_toy(
    model: ToyLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    model.reset_cache()
    logits = model(input_ids, use_cache=True)
    out = input_ids
    for _ in range(max_new_tokens):
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat((out, nxt), dim=1)
        logits = model(nxt, use_cache=True)
    return out


def _load_hf(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return tok, model


@torch.no_grad()
def generate_hf(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("toy", "hf"), default="toy")
    p.add_argument("--kernels", choices=("eager", "triton"), default="eager")
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max-new", type=int, default=16)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")

    if args.backend == "toy":
        torch.manual_seed(0)
        model = ToyLM().to(args.device)
        model.set_kernels(args.kernels == "triton")
        kinds = [b.kind for b in model.blocks]
        print(
            f"toy layout: {kinds}  "
            f"GQA {model.cfg.n_heads_q}q/{model.cfg.n_heads_kv}kv  D={model.cfg.head_dim}  "
            f"kernels={args.kernels}"
        )
        ids = torch.randint(0, model.cfg.vocab_size, (1, 8), device=args.device)
        t0 = time.perf_counter()
        out = generate_toy(model, ids, args.max_new)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        print(f"backend=toy tokens={out[0].tolist()}  {ms:.1f} ms")
        return

    try:
        tok, model = _load_hf(args.model, args.device)
    except Exception as e:
        raise SystemExit(
            f"加载 {args.model} 失败：{type(e).__name__}: {e}\n"
            "无网或未缓存权重时用：python projects/qwen35_infer/generate.py --backend toy --kernels triton"
        ) from e
    text_cfg = getattr(model.config, "text_config", model.config)
    print(
        f"hf {args.model}  hidden={getattr(text_cfg, 'hidden_size', '?')}  "
        f"layers={getattr(text_cfg, 'num_hidden_layers', '?')}  "
        f"layer_types={getattr(text_cfg, 'layer_types', None)}"
    )
    if args.kernels == "triton":
        from patch import patch_model

        counts = patch_model(model)
        print(f"patched: {counts}  (DeltaNet 仍走 PyTorch)")
    text = generate_hf(model, tok, args.prompt, args.max_new, args.device)
    print(text)


if __name__ == "__main__":
    main()
