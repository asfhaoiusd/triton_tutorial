# 第 1 课：vector_add

- **先跑：** `step01_add_1d.py` → `step02_add_2d.py` → `step03_add_3d.py`
- **选修：** 无
- **依赖：** 先读 `../00_syntax.md`

## 1. 公式与数据流

`out = x + y`。每个 program 处理一块 `BLOCK` 元素；越界用 `mask`。2D/3D 用各自的 stride，不要假设连续。

## 2. 怎么跑

```bash
python kernel/triton/tutorial/01_vector_add/step01_add_1d.py
python kernel/triton/tutorial/01_vector_add/step02_add_2d.py
python kernel/triton/tutorial/01_vector_add/step03_add_3d.py
```

## 3. 正确性

对照 `x + y`。elementwise 应对齐到 atol 1e-5 量级。

## 4. 效果

这是带宽绑定：算的是一次加法，时间看 load/store。autotune 主要扫 `BLOCK_SIZE` / `num_warps`。用来练 launch 套路，不拿它和 cuBLAS 比。
