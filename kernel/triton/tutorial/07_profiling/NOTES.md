# 第 7 课：profiling / nsys

- **先跑：** 做完第 6 课 Flash 再采；`bash run_nsys.sh`
- **选修：** `ncu` 不做
- **依赖：** `06_attention/step01_naive.py` 与 `step03_flash_single.py`

日期：2026-08-16  
本机：RTX 5070 Laptop + WSL2（`sm_120`）。`torch.profiler` 的 CUPTI 采不到 GPU。  
**当前主线：`nsys`。** `ncu` 先不做。

---

## 1. 怎么跑

```bash
source .venv/bin/activate
bash kernel/triton/tutorial/07_profiling/run_nsys.sh
# 换形状：D=128 bash kernel/triton/tutorial/07_profiling/run_nsys.sh
```

教学 flash 把整段 `D` 放进 SRAM。本机上限约 99KB；`tl.dot` 又要求 tile ≥16，所以 **`D=512` 教学核跑不了**（不是 nsys 坏了）。对照三种实现用默认 `D=64`。

或手动（一次一种实现）：

```bash
cd kernel/triton/tutorial/07_profiling
nsys profile --stats=true --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  -o nsys_out/flash \
  python nsys_targets.py --impl flash
```

`--impl`：`torch` | `naive` | `flash`

看结果：

```bash
# 三种实现对照表（run_nsys.sh 结束会自动写）
less kernel/triton/tutorial/07_profiling/nsys_out/compare.md

# 或只从已有 .nsys-rep 重写 markdown，不用再采
python kernel/triton/tutorial/07_profiling/nsys_to_md.py

nsys stats nsys_out/flash.nsys-rep
# 时间线：Windows 上用 Nsight Systems 打开 .nsys-rep
```

未装 `nsys` 时先装 CUDA toolkit / Nsight Systems CLI，再 `nsys --version`。

若 nsys 也报 CUPTI / invalid device：和 `torch.profiler` 同一类环境限制，继续用 `do_bench`。

---

## 2. `nsys` 基本语法

```bash
# 包住整个 Python 进程
nsys profile -o out --stats=true python script.py

# 常用开关
nsys profile \
  --stats=true \              # 结束后打 CUDA kernel 汇总表
  --force-overwrite=true \    # 覆盖已有 out.nsys-rep
  --trace=cuda,nvtx,osrt \    # GPU kernel + NVTX + 部分 OS
  -o out \
  python script.py
```

看什么（和当初看 profiler 一样）：

1. CUDA kernel **条数 / 名字**（naive 是否 matmul+softmax+matmul 或多次 Triton）  
2. flash 是否接近 **一条** 长 kernel  
3. 中间有没有为 `(B,S,S)` 服务的大拷贝 / 大 GEMM  

`do_bench` 问快慢；`nsys` 问时间花在哪、launch 几次。

分层：

| 层级 | 工具 | 状态 |
|------|------|------|
| 0 | `do_bench` | ✅ |
| 1 | `torch.profiler` | 本机 CUDA 列不可用，仅作 CPU 对照 |
| 2 | `nsys` | **现在学** |
| 3 | `ncu` | 以后 |

---

## 3. `torch.profiler`（附录，本机残缺）

CUPTI 失败时只有 CPU：`aten::matmul` / `softmax` 看得到，Triton kernel 名看不到。

```python
from torch.profiler import profile, ProfilerActivity, record_function

for _ in range(8):
    my_fn(x)
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with record_function("my_fn"):
        for _ in range(20):
            my_fn(x)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("out.json")
```

已跑过 `profile_attention.py` 的结论：

| 实现 | 本机能看到的 | 含义 |
|------|----------------|------|
| torch | matmul/bmm/softmax 多次 | 路径拆开 |
| naive | contig/copy/empty，看不见 Triton | Host 碎 + 物化 S |
| flash | 几乎只有 empty_like | Host 干净，kernel 被 CUPTI 漏掉 |

---

## 4. nsys 实验记录（2026.4.1，已出 GPU 表）

看 **`cuda_gpu_kern_sum`**，不要看 `osrt_sum` 里的 `poll`/`ioctl`（那是进程在等，不是算子）。

warmup 8 + loops 30 = **38** 次正式调用。torch 正好每种 kernel 38 次。

| 实现 | 真正干活的 GPU kernel | Instances | 含义 |
|------|----------------------|-----------|------|
| torch | cutlass/magma **sgemm** | 38+38 | QKᵀ、PV 两次 GEMM |
| torch | **softmax_warp_forward** | 38 | 单独 softmax |
| torch | elementwise（×scale） | 38 | 单独 mul |
| naive | `matmul_kernel_three_dimension` | 1657 | 2 次 matmul + **autotune 乱打** |
| naive | `online_softmax_kernel` | 999 | 1 次 softmax + autotune |
| flash | `flash_atten_fwd_kernel` | 697 | 融合一次 + autotune |

观察：

- torch 路径干净：`GEMM → softmax → GEMM`，和公式一一对应。  
- flash 业务 kernel 只有一种，符合「不物化 S、一次融合」。  
- naive/flash 表里 **FillFunctor 占了 90%+ 时间、上千次**，是 Triton **autotune / 编译** 被一起采进去了，不是 attention 本身那么慢。  
- 已改：`nsys_targets.py` 先 warmup，再用 `cudaProfiler.start/stop`；`run_nsys.sh` 加 `--capture-range=cudaProfilerApi`，报告不含 autotune。再跑一次 `bash run_nsys.sh`。
