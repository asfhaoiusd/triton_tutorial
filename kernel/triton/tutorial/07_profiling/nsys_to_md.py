"""把 nsys 的 cuda_gpu_kern_sum 写成对照 markdown。"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "nsys_out"
IMPLS = ("torch", "naive", "flash")


def short_name(name: str, limit: int = 80) -> str:
    raw = name.strip().strip('"')
    if "cutlass_" in raw:
        token = raw.split("cutlass_", 1)[1]
        token = token.split(">", 1)[0].split("(", 1)[0]
        return "cutlass_" + token
    name = raw
    if name.startswith("void "):
        name = name[5:]
    if name.startswith("<unnamed>::"):
        name = name[len("<unnamed>::") :]
    name = name.replace("at::native::", "")
    if "<" in name:
        name = name.split("<", 1)[0]
    if "(" in name:
        name = name.split("(", 1)[0]
    if len(name) > limit:
        return name[: limit - 1] + "…"
    return name


def ns_to_us(ns: float) -> str:
    return f"{ns / 1000:.1f}"


def load_kern_sum(rep: Path) -> list[dict]:
    cmd = [
        "nsys",
        "stats",
        "--force-export=true",
        "--report",
        "cuda_gpu_kern_sum",
        "--format",
        "csv",
        "--output",
        "-",
        str(rep),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    text = proc.stdout
    start = text.find("Time (%),")
    if start < 0:
        raise SystemExit(f"读不到 cuda_gpu_kern_sum: {rep}\n{proc.stderr}")
    reader = csv.DictReader(io.StringIO(text[start:]))
    rows = []
    for raw in reader:
        if not raw.get("Name"):
            continue
        rows.append(
            {
                "pct": float(raw["Time (%)"]),
                "total_ns": float(raw["Total Time (ns)"]),
                "n": int(float(raw["Instances"])),
                "avg_ns": float(raw["Avg (ns)"]),
                "med_ns": float(raw["Med (ns)"]),
                "name": raw["Name"],
            }
        )
    return rows


def impl_section(impl: str, rows: list[dict]) -> str:
    total = sum(r["total_ns"] for r in rows)
    n_kind = len(rows)
    n_launch = sum(r["n"] for r in rows)
    lines = [
        f"## {impl}",
        "",
        f"- GPU kernel 总时间：**{ns_to_us(total)} µs**（{n_launch} 次 launch，{n_kind} 种 kernel）",
        "",
        "| Time % | Total µs | Inst | Avg µs | Med µs | Kernel |",
        "|-------:|---------:|-----:|-------:|-------:|--------|",
    ]
    for r in rows:
        lines.append(
            "| {pct:.1f} | {tot} | {n} | {avg} | {med} | `{name}` |".format(
                pct=r["pct"],
                tot=ns_to_us(r["total_ns"]),
                n=r["n"],
                avg=ns_to_us(r["avg_ns"]),
                med=ns_to_us(r["med_ns"]),
                name=short_name(r["name"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    out_md = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "compare.md"
    sections = []
    summary_rows = []
    missing = []

    for impl in IMPLS:
        rep = OUT / f"{impl}.nsys-rep"
        if not rep.exists():
            missing.append(impl)
            continue
        rows = load_kern_sum(rep)
        sections.append(impl_section(impl, rows))
        total = sum(r["total_ns"] for r in rows)
        summary_rows.append(
            {
                "impl": impl,
                "total_us": ns_to_us(total),
                "kinds": str(len(rows)),
                "launches": str(sum(r["n"] for r in rows)),
                "top": short_name(rows[0]["name"]) if rows else "-",
            }
        )

    if not sections:
        raise SystemExit(f"{OUT} 里没有 .nsys-rep，先跑 run_nsys.sh")

    meta_path = OUT / "meta.json"
    if meta_path.exists():
        shape = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        shape = {"B": 4, "S": 256, "D": 128, "loops": 30, "warmup": 8}

    lines = [
        "# Attention nsys 对照",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"形状：`(B, S, D) = ({shape['B']}, {shape['S']}, {shape['D']})`，"
        f"warmup {shape['warmup']}（不采），正式循环 {shape['loops']}。",
        "只看 **`cuda_gpu_kern_sum`**。`ioctl` / NVTX SKIPPED / memcpy SKIPPED 可忽略。",
        "",
        "## 总览",
        "",
        "| 实现 | GPU kernel 总时间 µs | kernel 种类 | launch 次数 | 最重的 kernel |",
        "|------|---------------------:|------------:|------------:|---------------|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['impl']} | {r['total_us']} | {r['kinds']} | {r['launches']} | `{r['top']}` |"
        )
    lines.extend(
        [
            "",
            "预期（排除 autotune 后）：",
            "",
            "- **torch**：多种 kernel（GEMM + softmax + scale），每种约 30 次",
            "- **naive**：matmul + softmax + matmul，launch 更多",
            "- **flash**：主要一条 `flash_atten_fwd_kernel`，约 30 次",
            "",
        ]
    )
    lines.extend(sections)
    if missing:
        lines.extend(
            [
                "## 缺失",
                "",
                f"没有报告：{', '.join(missing)}。再跑 `bash run_nsys.sh`。",
                "",
            ]
        )
    lines.extend(
        [
            "## 原始报告",
            "",
            "```",
            f"{OUT}/torch.nsys-rep",
            f"{OUT}/naive.nsys-rep",
            f"{OUT}/flash.nsys-rep",
            "```",
            "",
        ]
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
