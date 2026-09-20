# ai_infra

Triton course: syntax → operators → plug your kernels into Qwen3.5-0.8B generate → small-model training vs `torch.compile`.

这是**教学仓库**，不是迷你 vLLM，也不追求端到端打赢生产推理引擎。

面向会一点 Python / PyTorch、想自己写 GPU kernel 的人。课程分四阶段：

1. **语法**：`pid`、分块、`mask`、stride，写出 `load → 算 → store`
2. **算子**：按课号 `01`→`07` 写常见算子，对齐 PyTorch，并分析为什么快/慢
3. **推理实战**：用自己的核接进 **Qwen3.5-0.8B** 的 generate，和 eager /（可选）vLLM 比正确性与吞吐
4. **训练对照**：Llama 式小模型上，用带反向的 Triton 核对照 eager / `torch.compile`

**从 [`kernel/triton/tutorial/README.md`](kernel/triton/tutorial/README.md) 按编号走。** 中文讲义 PDF：[`docs/textbook/main.pdf`](docs/textbook/main.pdf)（[GitHub 打开](https://github.com/asfhaoiusd/triton_tutorial/blob/main/docs/textbook/main.pdf)）。源码在 [`docs/textbook/`](docs/textbook/)，本地重编：`cd docs/textbook && make`。作者学习日记：[`学习计划书.md`](kernel/triton/tutorial/学习计划书.md)。

```text
LICENSE / NOTICE                 本仓库 MIT；第三方见 NOTICE
scripts/setup_env.sh             建 venv，装 cu128 轮子
kernel/triton/tutorial/          主课 00–07
kernel/triton/Triton-Puzzles/    旁路练习（vendored，Apache-2.0）
projects/qwen35_infer/           Part 3：generate，只换 RMSNorm / SwiGLU / GQA
projects/llm_train/              Part 4：小模型训练对照
docs/textbook/                   讲义源码；编译好的 PDF 见 docs/textbook/main.pdf
```

---

## 环境

在 **RTX 5070 Laptop + WSL2 + CUDA 12.8**（Blackwell / `sm_120`）上验过。其它 NVIDIA GPU 一般也能跑第 1–2 阶段；请装和本机 CUDA 匹配的 PyTorch 轮子。锁文件见 `requirements.lock.txt`（测过 `torch==2.11.0+cu128`、`triton==3.6.0`）。

```bash
git clone https://github.com/asfhaoiusd/triton_tutorial.git
cd ai_infra
bash scripts/setup_env.sh
source .venv/bin/activate
python kernel/triton/tutorial/01_vector_add/step01_add_1d.py
```

不要 bash 直接执行没有 shebang 的 `.py`。Part 3 另装 HuggingFace / 可选 vLLM，见 [`projects/qwen35_infer/README.md`](projects/qwen35_infer/README.md)。

国内拉 Hugging Face 权重可选用镜像（不要写进代码当默认）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

---

## Part 1 — 语法

读 [`00_syntax.md`](kernel/triton/tutorial/00_syntax.md)，边读边跑第 1 课：

| 课 | 文件 | 概念 |
|----|------|------|
| 1.1 | [`01_vector_add/step01_add_1d.py`](kernel/triton/tutorial/01_vector_add/step01_add_1d.py) | `program_id`、`BLOCK`、`mask` |
| 1.2 | [`step02_add_2d.py`](kernel/triton/tutorial/01_vector_add/step02_add_2d.py) | stride |
| 1.3 | [`step03_add_3d.py`](kernel/triton/tutorial/01_vector_add/step03_add_3d.py) | 三维 grid |

旁路练习：[`kernel/triton/Triton-Puzzles/`](kernel/triton/Triton-Puzzles/)（[gpu-mode/Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles)，Apache-2.0）。

---

## Part 2 — 算子（01→07）

课表见 tutorial README。依赖：`01 → 03/04 → 06`。`02` 的 ReLU 可紧跟语法；SwiGLU 深融合先扫一眼 `03_matmul`。

推理会用到的 Host API：

- RMSNorm：`05_norm/step03_rmsnorm.py` 里的 `rmsnorm`
- RoPE：`06_attention/step05_rope.py`
- GQA Flash：`06_attention/step06_gqa.py`

---

## Part 3 — 推理实战

[`projects/qwen35_infer/`](projects/qwen35_infer/)：只换 RMSNorm / SwiGLU / Gated Attention。DeltaNet 第一版仍走 PyTorch。裸 Triton 前向 **没有 autograd 图**。

```bash
python projects/qwen35_infer/smoke_test.py
python projects/qwen35_infer/bench.py --backend-list toy_eager,toy_triton
```

真模型 greedy 时，eager 与 Triton 应对齐。端到端略快不等于教学核赢过 FlashAttention：约 18/24 层仍是 DeltaNet 的参考实现。

---

## Part 4 — 训练：自己的核 vs torch.compile

[`projects/llm_train/`](projects/llm_train/)：LayerNorm + RoPE + GQA + SwiGLU，混合精度若干 step。Triton **默认只换 Flash Attention**（`--triton-ops fa`）；教学 LN / SwiGLU 用 `--triton-ops fa,ln,swiglu`（大 GEMM 常负优化）。MLP down 始终 `nn.Linear`。

8GB 上全参 1B+Adam **会 OOM**（bf16 减激活，不减 Adam 状态）。默认 `--config small`；`--config 1b` 失败时看显存账即可。

```bash
python kernel/triton/tutorial/05_norm/step01_layernorm.py
python kernel/triton/tutorial/02_activations/step03_swiglu_fused.py
python kernel/triton/tutorial/06_attention/step04_flash_causal_mha.py
python projects/llm_train/bench.py --config small --steps 10
```

---

## 许可证

课代码与讲义源码为 [MIT](LICENSE)。第三方见 [NOTICE](NOTICE)。第 8 章若引用 FlashAttention-2 插图，出处是 Dao, ICLR 2024，[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)，不要把论文 PDF 放进本仓库。
