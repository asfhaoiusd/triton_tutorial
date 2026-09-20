# 第 5 课：norm

- **先跑：** `step01_layernorm.py` → `step02_online_layernorm.py` → `step03_rmsnorm.py`
- **选修：** LN 反向写在 `step01_layernorm.py` 同文件；RMS bwd 可跳
- **依赖：** 第 4 课的「沿最后一维归约」；推理 API 是 `from step03_rmsnorm import rmsnorm`

## 1. 公式与数据流

LayerNorm：`(x-μ)/√(σ²+ε) * γ + β`。RMSNorm：`weight * x / √(mean(x²)+ε)`。online 沿 N 分块累加。LN 反向：`dγ/dβ` 用 `atomic_add`（autotune 要 `reset_to_zero`）。

## 2. 怎么跑

```bash
python kernel/triton/tutorial/05_norm/step01_layernorm.py
python kernel/triton/tutorial/05_norm/step02_online_layernorm.py
python kernel/triton/tutorial/05_norm/step03_rmsnorm.py
```

## 3. 正确性

LN 对 `F.layer_norm`。RMS 对 `F.rms_norm`。fp32，`rtol=1e-4` 量级。

## 4. 效果

大 N 时分块版接近 torch，是因为 naive 整行会炸寄存器。
