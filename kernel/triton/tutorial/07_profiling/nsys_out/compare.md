# Attention nsys 对照

生成时间：2026-08-16 19:25

形状：`(B, S, D) = (4, 256, 128)`，warmup 8（不采），正式循环 30。 只看 `cuda_gpu_kern_sum`。`ioctl` / NVTX SKIPPED / memcpy SKIPPED 可忽略。

## 总览


| 实现    | GPU kernel 总时间 µs | kernel 种类 | launch 次数 | 最重的 kernel                                   |
| ----- | ----------------- | --------- | --------- | -------------------------------------------- |
| torch | 1253.3            | 4         | 120       | `cutlass_80_simt_sgemm_128x32_8x5_tn_align1` |
| naive | 682.7             | 4         | 150       | `matmul_kernel_three_dimension`              |
| flash | 3115.7            | 1         | 30        | `flash_atten_fwd_kernel`                     |


预期（排除 autotune 后）：

- **torch**：多种 kernel（GEMM + softmax + scale），每种约 30 次
- **naive**：matmul + softmax + matmul，launch 更多
- **flash**：主要一条 `flash_atten_fwd_kernel`，约 30 次



## torch

- GPU kernel 总时间：**1253.3 µs**（120 次 launch，4 种 kernel）


| Time % | Total µs | Inst | Avg µs | Med µs | Kernel                                       |
| ------ | -------- | ---- | ------ | ------ | -------------------------------------------- |
| 51.1   | 639.9    | 30   | 21.3   | 16.6   | `cutlass_80_simt_sgemm_128x32_8x5_tn_align1` |
| 36.9   | 462.7    | 30   | 15.4   | 11.8   | `cutlass_80_simt_sgemm_128x32_8x5_nn_align1` |
| 6.8    | 85.4     | 30   | 2.8    | 2.2    | `softmax_warp_forward`                       |
| 5.2    | 65.2     | 30   | 2.2    | 1.7    | `vectorized_elementwise_kernel`              |




## naive

- GPU kernel 总时间：**682.7 µs**（150 次 launch，4 种 kernel）


| Time % | Total µs | Inst | Avg µs | Med µs | Kernel                          |
| ------ | -------- | ---- | ------ | ------ | ------------------------------- |
| 54.3   | 370.5    | 60   | 6.2    | 6.2    | `matmul_kernel_three_dimension` |
| 26.4   | 180.5    | 30   | 6.0    | 6.0    | `elementwise_kernel`            |
| 11.0   | 75.2     | 30   | 2.5    | 2.5    | `online_softmax_kernel`         |
| 8.3    | 56.5     | 30   | 1.9    | 1.9    | `vectorized_elementwise_kernel` |




## flash

- GPU kernel 总时间：**3115.7 µs**（30 次 launch，1 种 kernel）


| Time % | Total µs | Inst | Avg µs | Med µs | Kernel                   |
| ------ | -------- | ---- | ------ | ------ | ------------------------ |
| 100.0  | 3115.7   | 30   | 103.9  | 103.2  | `flash_atten_fwd_kernel` |




## 原始报告

```
kernel/triton/tutorial/07_profiling/nsys_out/torch.nsys-rep
kernel/triton/tutorial/07_profiling/nsys_out/naive.nsys-rep
kernel/triton/tutorial/07_profiling/nsys_out/flash.nsys-rep
```

