"""对照：玩具 eager / 玩具 Triton /（可选）HF / vLLM。

vLLM 通常端到端更快，因为 PagedAttention、CUDA Graph、调度，不是某一个教学 Softmax 核。
"""

from __future__ import annotations

import argparse
import time

import torch

from generate import generate_toy
from toy_model import ToyLM

WHY_VLLM = """
为什么 vLLM 通常更快（不要只比 kernel ms）
- PagedAttention：KV 按块分页，显存碎片少，长上下文更稳
- CUDA Graph / 融合：decode 少 launch
- continuous batching：多请求时 GPU 更满（单请求短上下文优势更小）
本课教学 generate 没有以上三项；DeltaNet 仍是 eager PyTorch。
对照价值：greedy 是否同 token；换核后 TTFT / tok/s 相对 naive eager 如何。
"""


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def bench_toy(kernels: str, prompt_len: int, max_new: int, warmup: int = 2) -> dict:
    torch.manual_seed(0)
    model = ToyLM().cuda()
    model.set_kernels(kernels == "triton")
    ids = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device="cuda")
    for _ in range(warmup):
        model.reset_cache()
        generate_toy(model, ids, max_new)
    _sync()
    torch.cuda.reset_peak_memory_stats()
    model.reset_cache()
    t0 = time.perf_counter()
    logits = model(ids, use_cache=True)
    _sync()
    ttft_ms = (time.perf_counter() - t0) * 1e3
    t1 = time.perf_counter()
    out = ids
    for _ in range(max_new):
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat((out, nxt), dim=1)
        logits = model(nxt, use_cache=True)
    _sync()
    decode_s = time.perf_counter() - t1
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    return {
        "backend": f"toy_{kernels}",
        "ttft_ms": ttft_ms,
        "tok_s": max_new / decode_s if decode_s > 0 else float("inf"),
        "peak_mb": peak_mb,
        "tokens": out[0].tolist(),
    }


def bench_hf(model_id: str, prompt: str, max_new: int, kernels: str, warmup: int = 1) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    if kernels == "triton":
        from patch import patch_model

        print("patched", patch_model(model))
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    gen_kw = dict(max_new_tokens=max_new, do_sample=False, use_cache=True)
    for _ in range(warmup):
        model.generate(ids, **gen_kw)
        _sync()
    torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.perf_counter()
    out = model.generate(ids, **gen_kw)
    _sync()
    elapsed = time.perf_counter() - t0
    n = out.shape[1] - ids.shape[1]
    text = tok.decode(out[0], skip_special_tokens=True)
    del model
    torch.cuda.empty_cache()
    return {
        "backend": f"hf_{kernels}",
        "prompt_tok": int(ids.shape[1]),
        "ttft_ms": None,
        "tok_s": n / elapsed if elapsed > 0 else float("inf"),
        "wall_ms": elapsed * 1e3,
        "peak_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "text": text[:80],
    }


def bench_vllm(model_id: str, prompt: str, max_new: int) -> dict:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_id, max_model_len=1024, trust_remote_code=True, dtype="float16")
    params = SamplingParams(temperature=0.0, max_tokens=max_new)
    t0 = time.perf_counter()
    outs = llm.generate([prompt], params)
    elapsed = time.perf_counter() - t0
    text = outs[0].outputs[0].text
    n = len(outs[0].outputs[0].token_ids)
    return {
        "backend": "vllm",
        "ttft_ms": None,
        "tok_s": n / elapsed if elapsed > 0 else float("inf"),
        "wall_ms": elapsed * 1e3,
        "peak_mb": None,
        "text": text,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend-list", default="toy_eager,toy_triton")
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--prompt-len", type=int, default=32)
    p.add_argument("--max-new", type=int, default=16)
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()
    print(WHY_VLLM)
    rows = []
    for name in args.backend_list.split(","):
        name = name.strip()
        try:
            if name == "toy_eager":
                rows.append(bench_toy("eager", args.prompt_len, args.max_new, warmup=max(args.warmup, 2)))
            elif name == "toy_triton":
                rows.append(bench_toy("triton", args.prompt_len, args.max_new, warmup=max(args.warmup, 2)))
            elif name == "hf_eager":
                rows.append(bench_hf(args.model, args.prompt, args.max_new, "eager", warmup=args.warmup))
            elif name == "hf_triton":
                rows.append(bench_hf(args.model, args.prompt, args.max_new, "triton", warmup=args.warmup))
            elif name == "vllm":
                rows.append(bench_vllm(args.model, args.prompt, args.max_new))
            else:
                print(f"skip unknown backend {name}")
        except ImportError as e:
            print(f"{name}: missing dependency ({e}). 见 projects/qwen35_infer/README.md")
        except Exception as e:
            print(f"{name}: failed: {type(e).__name__}: {e}")
    for r in rows:
        print(r)
    if len(rows) >= 2 and "tokens" in rows[0] and "tokens" in rows[1]:
        same = rows[0]["tokens"] == rows[1]["tokens"]
        print(f"toy greedy tokens match: {same}")


if __name__ == "__main__":
    main()
