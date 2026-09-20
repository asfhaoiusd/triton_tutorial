# Triton 课：从这里开始

打开本目录，按编号往下走。`ls` 的顺序就是推荐顺序。

```text
00_syntax.md     语法笔记（先读，边读边跑 01）
01_vector_add    pid / BLOCK / mask / stride
02_activations   ReLU → SwiGLU（深融合建议先扫一眼 03）
03_matmul        dot → 2D/3D GEMM；step90 是选修反向
04_softmax       naive → online；step90 选修反向
05_norm          LayerNorm / RMSNorm
06_attention     naive → MHA → Flash → RoPE / GQA → MLA
07_profiling     用 nsys 看 launch（先做完 06）
```

每课里的 `.py` 也带 `stepXX_` 前缀。`step90_` 是选修（autograd 反向），初学者可以跳过。

```bash
source .venv/bin/activate
python kernel/triton/tutorial/01_vector_add/step01_add_1d.py
```

仓库总入口还是根目录 [`README.md`](../../../README.md)（含环境、Part 3 推理实战、Part 4 训练对照）。中文讲义：[`docs/textbook/`](../../../docs/textbook/)。作者进度板：[`学习计划书.md`](学习计划书.md)，可后看。

训练对照（有反向的 RMSNorm / FA，不是 generate 那条 fwd-only API）：[`projects/llm_train/`](../../../projects/llm_train/)。

分析怎么写：[`ANALYSIS_TEMPLATE.md`](ANALYSIS_TEMPLATE.md)。每课有一份 `NOTES.md`。
