# 第 4 课：softmax

- **先跑：** `step01_naive.py` → `step02_online.py`
- **选修：** `step90_backward.py`（`autograd.Function`）
- **依赖：** 第 1 课；归约维不能用 `program_id` 切开

## 1. 公式与数据流

最后一维：`m=max(x); y=exp(x-m)/sum(exp(x-m))`。naive 整行一次装下；online 沿列分块维护 `(m,d)`。反向：`dx = y * (dy - sum(y*dy))`，存 `y` 不存 `x`。

## 2. 怎么跑

```bash
python kernel/triton/tutorial/04_softmax/step01_naive.py
python kernel/triton/tutorial/04_softmax/step02_online.py
python kernel/triton/tutorial/04_softmax/step90_backward.py
```

## 3. 正确性

对 `F.softmax(..., dim=-1)`；反向 `autograd.grad`。`TritonSoftmax.apply` 才建图。

## 4. 效果

完整输出 `y` 时 online **不一定更快**（多读一遍）。意义是 N 很大装不下，以及给 Flash 准备统计量。`N` 先 ≤1024。
