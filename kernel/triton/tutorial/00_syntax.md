# Triton 上手教程：基本语法与常用函数

面向：会一点 Python / PyTorch，想快速看懂并写出简单 GPU kernel。  
环境：NVIDIA GPU + CUDA；推荐 Linux 或 WSL2。Windows 原生对 Triton 支持很弱。

```bash
# 推荐：在仓库根目录用已配好的环境（RTX 50 系需 cu128）
cd /path/to/ai_infra && source .venv/bin/activate
# 或重新安装： bash scripts/setup_env.sh
```

---

## 1. 先建立心智模型

Triton 用 **类 Python 语法** 写 GPU kernel。你不写「一个线程处理一个元素」，而是写：

> **每个 program（类似 CUDA block）一次处理一整块数据（`BLOCK_SIZE` 个元素）。**

| 概念 | 含义 |
|------|------|
| `program` / `pid` | 并行执行的一份 kernel 实例，用 `tl.program_id` 区分 |
| `BLOCK_SIZE` | 每个 program 一次处理多少元素，通常是 2 的幂 |
| pointer + offset | 用指针算术定位，再 `tl.load` / `tl.store` |
| `mask` | 边界保护：长度不是 block 整数倍时，多余位置不读不写 |

数据被切成多块，每个 `pid` 负责一块：

```text
数组:  [0 ............. n-1]
       |-- block0 --|-- block1 --|-- block2 --|
pid:        0            1            2
```

---

## 2. 最小完整程序

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


x = torch.randn(10000, device="cuda")
y = torch.randn(10000, device="cuda")
assert torch.allclose(add(x, y), x + y)
```

记住固定套路：

1. `pid` + `BLOCK_SIZE` → `offsets`
2. `mask = offsets < n`
3. `load` → 计算 → `store`
4. Host 侧算 `grid` 并启动 kernel

可运行版本：`01_vector_add/step01_add_1d.py`。

---

## 3. Kernel 结构与启动

### 3.1 装饰器

```python
@triton.jit
def my_kernel(...):
    ...
```

- `@triton.jit`：把函数编译成 GPU kernel。
- Kernel 内只能用 Triton 语言（`tl.*`）和受限 Python 子集，**不能**随意调用普通 Python / NumPy / PyTorch API。

### 3.2 启动语法

```python
# grid: 启动多少个 program
grid = (num_programs,)                 # 一维
# 或
grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

my_kernel[grid](arg0, arg1, ..., BLOCK_SIZE=1024)
```

- `kernel[grid](...)`：按 grid 启动。
- `triton.cdiv(a, b)`：向上取整除法，等价 `(a + b - 1) // b`。

### 3.3 参数类型

| 类型 | 例子 | 说明 |
|------|------|------|
| 指针 | `x_ptr` | 由 CUDA tensor 自动传入，kernel 里当指针用 |
| 运行时标量 | `n_elements` | 每次调用可变 |
| 编译期常量 | `BLOCK_SIZE: tl.constexpr` | 影响代码生成，常用于块大小、展开 |

```python
@triton.jit
def kernel(x_ptr, n, BLOCK_SIZE: tl.constexpr):
    ...
```

`tl.constexpr` 适合：`BLOCK_SIZE`、`num_stages` 相关的形状特化；不要把每次都变的长度标成 constexpr（除非你确实要为每个长度特化一份代码）。

---

## 4. 基本语法

### 4.1 program id 与网格

```python
pid = tl.program_id(axis=0)   # 一维网格
pid_x = tl.program_id(0)
pid_y = tl.program_id(1)      # 二维网格时
```

二维示例（处理矩阵按 tile）：

```python
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
# kernel 内:
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
```

### 4.2 构造索引向量

```python
# [0, 1, 2, ..., BLOCK_SIZE-1]
offs = tl.arange(0, BLOCK_SIZE)

# 当前 program 负责的全局下标
offsets = pid * BLOCK_SIZE + offs
```

二维索引常用写法：

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
# 广播成二维：shape [BLOCK_M, BLOCK_N]
offs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
```

`x[:, None]` / `x[None, :]` 是 Triton 里做广播的常用技巧。

### 4.3 mask（几乎每个 kernel 都要）

```python
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
tl.store(out_ptr + offsets, y, mask=mask)
```

- `mask=False` 的位置：**不读 / 不写** 真实内存。
- `other=`：被 mask 掉的 load 用什么填充（默认常当 0，建议显式写出）。

多维边界：

```python
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
```

### 4.4 指针算术

```python
# 一维连续
tl.load(x_ptr + offsets, mask=mask)

# 带 stride（非连续 / 矩阵行主序）
tl.load(x_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, mask=mask)
```

PyTorch tensor 传入后，Triton 把它当 **首元素指针**；你要自己用 stride 表达布局。

### 4.5 控制流（受限）

```python
# 编译期可知的循环：可展开
for k in tl.static_range(0, K, BLOCK_K):
    ...

# 运行时循环（不要太大、不要太发散）
for _ in range(num_iters):   # 有时可用，但要谨慎
    ...

# 数据相关分支：优先用 tl.where，而不是 Python if 逐元素
y = tl.where(x > 0, x, 0.0)
```

原则：

- **向量级选择** → `tl.where`
- **编译期常量循环** → `tl.static_range`
- 避免在 kernel 里写复杂、不规则的 Python 控制流

---

## 5. 常用函数速查

### 5.1 内存：load / store

```python
x = tl.load(ptr + offs, mask=mask, other=0.0)
tl.store(ptr + offs, value, mask=mask)
```

| 参数 | 含义 |
|------|------|
| `ptr + offs` | 要访问的地址（可向量化） |
| `mask` | 哪些 lane 有效 |
| `other` | load 时无效 lane 的填充值 |

### 5.2 索引与形状

| 函数 | 作用 |
|------|------|
| `tl.program_id(axis)` | 当前 program 编号 |
| `tl.num_programs(axis)` | 该轴 program 总数 |
| `tl.arange(start, end)` | 构造索引向量，`end-start` 通常为 constexpr 幂次 |
| `tl.zeros(shape, dtype=...)` | 全 0 |
| `tl.full(shape, value, dtype=...)` | 填充常量 |

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```

### 5.3 类型与转换

常见 dtype：`tl.float16` / `tl.bfloat16` / `tl.float32` / `tl.int32` / `tl.int64` 等。

```python
x_f32 = x.to(tl.float32)          # 或 tl.cast(x, tl.float32)
x_i32 = x.to(tl.int32)
```

累加器常用 **FP32**，即使输入是 FP16/BF16（数值更稳）。

### 5.4 逐元素算术

```python
c = a + b
c = a - b
c = a * b
c = a / b
c = -a
c = a * scale + bias
```

### 5.5 比较与选择

```python
m = a > b
m = a >= b
m = a == b
y = tl.where(cond, x, 0.0)
y = tl.maximum(a, b)
y = tl.minimum(a, b)
```

### 5.6 数学函数

```python
tl.exp(x)
tl.log(x)
tl.sqrt(x)
tl.rsqrt(x)          # 1/sqrt(x)，softmax / layernorm 常用
tl.abs(x)
tl.sin(x) / tl.cos(x)
tl.sigmoid(x)        # 若版本支持；否则可用 1/(1+exp(-x))
```

数值稳定 softmax 片段：

```python
x = x - tl.max(x, axis=0)
num = tl.exp(x)
den = tl.sum(num, axis=0)
y = num / den
```

### 5.7 归约（reduction）

```python
s = tl.sum(x, axis=0)
m = tl.max(x, axis=0)
m = tl.min(x, axis=0)
```

- `axis`：沿哪个维度归约。
- Softmax、LayerNorm、Attention 里几乎都会用到。

### 5.8 广播与形状操作

```python
# 扩维广播
a2d = a[:, None]          # [N] -> [N, 1]
b2d = b[None, :]          # [M] -> [1, M]
c = a2d * b2d             # [N, M]

# 常见：行向量减各行最大值
x = x - m[:, None]
```

### 5.9 矩阵乘核心：`tl.dot`

```python
# a: [BLOCK_M, BLOCK_K], b: [BLOCK_K, BLOCK_N]
c = tl.dot(a, b)                    # -> [BLOCK_M, BLOCK_N]
c = tl.dot(a, b, acc)               # 累加到 acc（部分版本写法）
acc += tl.dot(a, b)                 # 更常见的累加写法
```

这是写 matmul / attention 的核心原语。输入类型、layout（是否转置）会影响性能与合法性。

### 5.10 原子操作（少用但有用）

```python
tl.atomic_add(ptr + offs, val, mask=mask)
tl.atomic_max(ptr + offs, val, mask=mask)
```

用于直方图、稀疏累积等；吞吐通常不如普通 store，能避免则避免。

---

## 6. Host 侧常用工具

```python
import triton

triton.cdiv(n, BLOCK)          # ceil division
triton.next_power_of_2(n)      # 下一个 2 的幂（定 BLOCK 时有用）

# 简单计时
ms = triton.testing.do_bench(lambda: add(x, y))
```

Autotune（进阶，知道即可）：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def kernel(...):
    ...
```

Triton 会按 `key` 缓存最优配置。上手阶段可先手写固定 `BLOCK_SIZE`。

---

## 7. 再看两个巩固例子

### 7.1 ReLU

```python
@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.where(x > 0, x, 0.0), mask=mask)
```

### 7.2 原地乘标量

```python
@triton.jit
def mul_scalar_kernel(x_ptr, out_ptr, scalar, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * scalar, mask=mask)
```

完整可运行代码：`examples/02_relu_and_scale.py`。

---

## 8. 常见坑

1. **忘了 mask** → 越界读写，结果错或直接崩溃。  
2. **CPU tensor** → 必须 `device="cuda"`。  
3. **`BLOCK_SIZE` 不是 2 的幂** → 很多情况能跑，但性能/限制更差，习惯用 128/256/1024。  
4. **在 kernel 里用 PyTorch** → 不行；只在 Host 准备数据、启动、校验。  
5. **FP16 累加不稳** → 归约/累加用 FP32。  
6. **把 Python `len` / 动态 list 直接搬进 kernel** → 通常不行；长度作标量参数传入。

---

## 9. 建议练习顺序

| 顺序 | 练习 | 练什么 |
|------|------|--------|
| 1 | 向量加减、ReLU、scale | `pid` / `arange` / `load` / `store` / `mask` |
| 2 | 行 softmax | `max` / `sum` / `exp` / 广播 |
| 3 | 小 matmul | `tl.dot` + 二维 `program_id` |
| 4 | fused bias+activation | 少读少写、融合 |

官方教程（更深入）：  
https://triton-lang.org/main/getting-started/tutorials/index.html

---

## 10. 一张纸速记

```text
@triton.jit
def kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = ...                      # + - * / where exp sum max dot
    tl.store(out_ptr + offs, y, mask=mask)

grid = (triton.cdiv(n, BLOCK),)
kernel[grid](x, out, n, BLOCK=BLOCK)
```

把上面十行背熟，再去读 Softmax / Matmul，会轻松很多。

下一份可接着写：`02_softmax_and_matmul.md`（归约 + `tl.dot` 实战）。
