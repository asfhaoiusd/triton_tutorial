# 第 2 课：activations

- **先跑：** `step01_relu.py` → `step02_swiglu.py` → `step03_swiglu_fused.py`
- **选修：** ReLU 反向在 `step01_relu.py` 同文件（`TritonReLU`）；SwiGLU 反向在 `step03_swiglu_fused.py`（`TritonSwiGLU`）
- **依赖：** ReLU 只需 01；**SwiGLU 深融合建议先扫一眼第 3 课 matmul**（里面有两次 `tl.dot`）

## 1. 公式与数据流

- ReLU：`y = max(x, 0)`；反向 `dx = dy * (x>0)`
- SwiGLU（浅）：Host 上 `x@W1`、`x@W2`，kernel 只做 `silu(a)*b`，中间 `a/b` 写 HBM
- SwiGLU（深）：同一 kernel 沿 K 做两次 `tl.dot`，SiLU×mul 后写回，不物化 `a/b`

## 2. 怎么跑

```bash
python kernel/triton/tutorial/02_activations/step01_relu.py
python kernel/triton/tutorial/02_activations/step02_swiglu.py
python kernel/triton/tutorial/02_activations/step03_swiglu_fused.py
```

## 3. 正确性

ReLU / `TritonReLU` 对 `F.relu` + `autograd.grad`。SwiGLU 对 `silu(x@W1)*(x@W2)`，fp32 建议关 TF32，`rtol=1e-3`。`TritonSwiGLU` 的 `dx/dW1/dW2` 对同一公式的 `autograd.grad`。

## 4. 效果

深融合快在少两次中间 GEMM 的 HBM 往返。`do_bench` 看 step02 vs step03。
