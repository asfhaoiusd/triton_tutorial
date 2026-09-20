# 第 3 课：matmul

- **先跑：** `step01_dot_1d.py` → `step02_dot_3d.py` → `step03_matmul_2d.py` → `step04_matmul_3d.py`
- **选修：** `step90_linear_backward.py`（未收完，初学者跳过）
- **依赖：** 第 1 课的 stride / mask；后面第 6 课 naive attn 会 import `step04_matmul_3d`

## 1. 公式与数据流

`C = A @ B`。输出块 `(BLOCK_M, BLOCK_N)` 沿 K 循环 `tl.dot` 累加。3D 用 `pid_b` 一次 launch。dot 的 atomic + autotune 必须 `reset_to_zero`。

## 2. 怎么跑

```bash
python kernel/triton/tutorial/03_matmul/step01_dot_1d.py
python kernel/triton/tutorial/03_matmul/step03_matmul_2d.py
python kernel/triton/tutorial/03_matmul/step04_matmul_3d.py
```

## 3. 正确性

关 `allow_tf32` 再比。对照 `x @ w`（3D `w` 不是 `F.linear`）。

## 4. 效果

教学核通常慢于 cuBLAS，课看的是分块数据流。
