# Part 4：用自己的核加速训练，对照 torch.compile

架构：**LayerNorm + RoPE + GQA + SwiGLU**（训练骨架用 LN，不是 RMSNorm）。不是 Qwen3.5 的 DeltaNet。
对照实验只跑若干 step：看 **step 时间、tok/s、峰值显存、loss 是否同量级**，不训到可用 chatbot。

推理课 [`qwen35_infer`](../qwen35_infer/) 仍可调裸 Triton **前向**。训练必须 `autograd.Function`。

## 显存（必须先看）

混合精度（`torch.amp.autocast('cuda', dtype=bf16)`）**减的是激活和 matmul，不是 Adam 状态**。

约 1B 参数全参 AdamW：

| 项 | 粗算 |
|----|------|
| 权重 bf16 | ~2 GB |
| Adam `m/v` 常仍是 fp32 | ~8 GB |
| 梯度 + 激活 | 再加几 GB |

**8GB 笔记本全参 1B+Adam 会 OOM。** 同一套代码两档配置：

| 配置 | 规模（约） | 8GB + bf16 autocast + checkpoint |
|------|------------|----------------------------------|
| `small`（默认，验收用） | `dim=2048`、6 层 GQA，约 0.25B，`seq=256`，`bs=1` | 应能跑完若干 step |
| `1b` | hidden=2048、22 层 GQA、vocab=32k，约 1.1B | 可能 OOM；失败时脚本会打印显存账 |

`1b` OOM **不算课程失败**。要硬塞 1B，需要 8-bit Adam / 更短 seq / 更少层，本课不实现。

本目录默认参数仍是 **fp32 权重 + autocast bf16**（Adam 状态跟参数走 fp32，和课上的 8 byte/param 一致）。

## 换哪些核

默认 `--triton-ops fa`（端到端要对齐速度用这个）：

```text
hidden → F.layer_norm → FA Function（GQA 核内 GROUP）→ residual
      → F.layer_norm → silu(x@Wg)*(x@Wu) + Linear down → residual
```

教学对照再开 `--triton-ops fa,ln,swiglu`：

```text
hidden → LayerNorm_triton → FA Function → residual
      → LayerNorm_triton → TritonSwiGLU(gate+up) → nn.Linear down → residual
```

- **Attention**：[`TritonFlashAttentionCausalMHA`](../../kernel/triton/tutorial/06_attention/step04_flash_causal_mha.py)，`GROUP = Hq/Hkv`，不再 `repeat_interleave` KV。`small` 的 `head_dim=128`。
- **LayerNorm / SwiGLU**：默认不换。教学核大矩阵打不过 cuBLAS，开了会负优化。
- MLP **down** 始终 `nn.Linear`。

`--backend compile` 用 eager 模块再 `torch.compile`。

## 三后端预期

| 后端 | 实际在跑什么 |
|------|----------------|
| `eager` | `F.layer_norm` + SDPA(`enable_gqa`) + `silu(x@Wg)*(x@Wu)` + Linear down |
| `compile` | 同一套 eager + `torch.compile(mode="reduce-overhead")`（失败降 `default`） |
| `triton`（默认 `fa`） | 只换 FA；LN / MLP 仍 cuBLAS |
| `triton` + `fa,ln,swiglu` | 再换教学 LN / 融合 SwiGLU |

**compile 常赢端到端**。Triton 默认只比 FA vs SDPA（本机 `small` 大约能略快于 eager）。教学 GEMM 不要拿来和 cuBLAS 比端到端。

## 改 tile（实验不同 kernel 尺寸）

模型宽深在 [`model.py`](model.py) 的 `CONFIGS`。Triton **BLOCK** 在 [`tiles.py`](tiles.py) 的 `DEFAULT`，或命令行覆盖（只影响 `triton` 后端）：

```bash
# 改 tiles.py 里 DEFAULT 后直接跑
python projects/llm_train/speed.py --config small --backends triton --steps 10

# 不改文件：FA 前向 128×32
python projects/llm_train/speed.py --backends triton --fa-block-m 128 --fa-block-n 32 --fa-num-warps 4 --fa-num-stages 1
python projects/llm_train/train.py --backend triton --triton-ops fa,ln,swiglu --swiglu-block-m 32 --swiglu-block-n 32 --swiglu-block-k 32
```

BLOCK 必须是 2 的幂。显式 launch 必须带 `num_stages`（FA 前向默认 1；SwiGLU 默认 2；FA bwd / LN 为 1），否则 JIT 默认 3 级流水会把 smem 乘过本机 101376。FA 反向不要上 64×64。eager / compile 忽略这些参数。

## 怎么跑

```bash
source .venv/bin/activate

python kernel/triton/tutorial/05_norm/step01_layernorm.py
python kernel/triton/tutorial/02_activations/step03_swiglu_fused.py   # 含 dx/dW 对齐
python kernel/triton/tutorial/06_attention/step04_flash_causal_mha.py

python projects/llm_train/train.py --config small --backend eager --steps 5
python projects/llm_train/speed.py --config small --steps 20
python projects/llm_train/bench.py --config small --steps 10
python projects/llm_train/bench.py --config 1b --steps 2 --backends eager
```

`small` 三个后端都应跑完；有限步内 loss 同量级即可。
