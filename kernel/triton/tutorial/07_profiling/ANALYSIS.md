# 算子分析：profiling / nsys

## 1. 公式与数据流

不写新算子。对 **同一形状** 的 naive / flash / torch attention 采 GPU kernel 表：名字、次数、总 µs。

## 2. 怎么跑

见同目录 [NOTES.md](NOTES.md)。先做完第 6 课再采。

## 3. 正确性

profiler 不替代 `allclose`。先保证三种实现数值对，再采。

## 4. 效果（已有一组，2026-08-16）

形状 `(4,256,128)`：torch 多种 cutlass+softmax；naive 多次 3D matmul + online softmax；flash 一种 `flash_atten_fwd_kernel`。flash **总 µs 可以更大**——教学核未调到生产级；课看 launch 结构。autotune 必须排除在采集窗外，否则 FillFunctor 会污染次数。
