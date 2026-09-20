# 第 6 课：attention

- **先跑：** `step01_naive.py` → `step02_mha.py` → `step03_flash_single.py` → `step04_flash_causal_mha_teach.py` → `step05_rope.py` → `step06_gqa.py`
- **选修：** `step07_mla.py`；加速版 `step04_flash_causal_mha.py`（Q 并行 / GQA，`llm_train` 用这个）
- **依赖：** 第 3 课 3D matmul + 第 4 课 softmax（naive 会 import）

## 1. 公式与数据流

Naive：`S=QKᵀ/√d → softmax → PV`，物化 `(S,S)`。Flash：外层 Q、内层 KV，片上 `(m,l,acc)`。多头 `grid=(B,H)`；causal 打在 `qk` 上。GQA：`Hq > Hkv`。Qwen3.5 的 DeltaNet 本课不实现。

## 2. 怎么跑

```bash
python kernel/triton/tutorial/06_attention/step01_naive.py
python kernel/triton/tutorial/06_attention/step02_mha.py
python kernel/triton/tutorial/06_attention/step03_flash_single.py
python kernel/triton/tutorial/06_attention/step04_flash_causal_mha_teach.py
python kernel/triton/tutorial/06_attention/step04_flash_causal_mha.py   # 加速版
python kernel/triton/tutorial/06_attention/step05_rope.py
python kernel/triton/tutorial/06_attention/step06_gqa.py
python kernel/triton/tutorial/06_attention/step07_mla.py
```

对照：`F.scaled_dot_product_attention`（GQA 用 `enable_gqa=True`）。

## 3. 正确性

fwd fp32、`rtol=1e-2`。裸 Triton `o` 没有 `grad_fn`。`lse` 是 `(B,H,S)`。`dQ` atomic 必须 `zeros`。

## 4. 效果

Flash 省 HBM 上的 `S/P`。`D` 太大炸 smem（本机上限约 101376）。看 launch 去做第 7 课。
