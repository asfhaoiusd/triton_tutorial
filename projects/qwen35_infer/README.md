# Part 3：用自己的 Triton 核跑 Qwen3.5，对照 vLLM

不要写迷你 vLLM。这里是一条 **generate**：prefill 整段 prompt，decode 时维护 KV。只换 **RMSNorm / SwiGLU / Gated Attention（GQA Flash）**。

## Qwen3.5 架构（必须记住）

`Qwen/Qwen3.5-0.8B` 不是纯 Transformer：

- 24 层：`6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`
- 大约 18 层线性注意力 + 6 层 softmax 注意力（GQA 8 个 Q 头 / 2 个 KV 头，`head_dim=256`）
- 第一版 **DeltaNet 仍用 PyTorch**（玩具模型里用一层 Linear 占位）

8GB 不要一上来 bf16 4B/9B。0.8B 短上下文够用。

## 和 vLLM 比什么

同一权重、同一组短 prompt、greedy、相同 `max_new_tokens`。指标：文本/token 是否一致、TTFT、decode tok/s、峰值显存。

vLLM 通常更快，因为 **PagedAttention、continuous batching、CUDA Graph、调度**，不是某一个 Softmax 核。课的结论应是：换核后这一层相对 eager 快了多少，端到端慢在哪。

## 环境

基础环境见仓库根 README。本目录额外：

```bash
source .venv/bin/activate
# 真模型需要 transformers 5（Qwen3.5 的 RMSNorm 是 1+w）
uv pip install -r projects/qwen35_infer/requirements-capstone.txt
# 对照 vLLM（可选，体积大，WSL + Blackwell 可能难装）
uv pip install vllm
```

## 怎么跑

```bash
# 不下载权重：核对齐 + 玩具 generate
python projects/qwen35_infer/smoke_test.py

# 玩具模型 greedy（eager vs 换核）
python projects/qwen35_infer/generate.py --backend toy --kernels eager
python projects/qwen35_infer/generate.py --backend toy --kernels triton

# 计时：玩具 eager / 玩具 triton；（有 vLLM 且有权重时再加 hf/vllm）
python projects/qwen35_infer/bench.py --backend-list toy_eager,toy_triton

# HuggingFace Qwen3.5-0.8B（需自己下载）
python projects/qwen35_infer/generate.py --backend hf --model Qwen/Qwen3.5-0.8B --kernels triton
python projects/qwen35_infer/bench.py --backend-list hf_eager,hf_triton,vllm --model Qwen/Qwen3.5-0.8B
```

短上下文（256–1024），不要官方 256K。无 Hugging Face 网络或未缓存权重时，`--backend hf` 会失败；玩具路径不需要权重。

greedy 时 eager 与 Triton 应对齐。端到端略快不等于教学核赢过 FlashAttention：约 18/24 层仍是 DeltaNet 的 PyTorch 参考实现（未装 `causal_conv1d` / `flash-linear-attention` 时两边都慢）。

Qwen3.5 的 RMSNorm 是 `y=(1+w)·x/rms`（`w` 初始化为 0）。`patch.py` 换核时会加上 `1+w`，不要直接把 `w` 丢进课上的 Llama 式 `rmsnorm`。
