# 算子分析模板

每个课目录的 `NOTES.md` 按这四块写。数字以你机器上 `python ...` 打印为准，下面只规定**比什么**。

## 1. 公式与数据流

- 输入 / 输出形状
- 哪一维 `program_id` 并行，哪一维 kernel 内 `for` / 归约
- 中间结果写不写 HBM（融合 vs Host 串）

## 2. 怎么跑

```bash
source .venv/bin/activate
python kernel/triton/tutorial/<课号_课名>/stepXX_<file>.py
```

对照对象写死（`F.softmax` / `F.layer_norm` / `F.scaled_dot_product_attention` / `x @ w`，不要混）。

## 3. 正确性

- dtype：教学默认 fp32；关 TF32 再比 GEMM
- `rtol` / `atol` 与 max abs err
- 反向：`torch.autograd.grad`，不要对 Triton 裸输出求 `grad`

## 4. 效果

至少回答三句：

1. **wall time**：`do_bench` 教学核 vs torch（同一形状、warmup 默认）
2. **快在哪**：少 FLOPs、少 HBM、还是少 launch？教学 Flash 往往 FLOPs 不少、访存少
3. **nsys（可选）**：本机 `torch.profiler` 常采不到 GPU，用 [`profiling/run_nsys.sh`](profiling/run_nsys.sh)。看 kernel **名字和次数**，不要只看总 µs 是否赢 SDPA

形状建议先小后大。归约维 `N=8192` 整行装会炸寄存器；Flash `D` 太大炸 smem（本机上限约 101376 字节）。
