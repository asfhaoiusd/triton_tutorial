#!/usr/bin/env bash
# 对 torch / naive / flash 各采一份 nsys。
# capture-range=cudaProfilerApi：只采 warmup 之后的正式循环，不含 autotune。
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/nsys_out"
mkdir -p "$OUT"

if ! command -v nsys >/dev/null 2>&1; then
  echo "未找到 nsys。"
  echo "安装: apt install nsight-systems-cli  (NVIDIA devtools 源)"
  echo "确认: nsys --version"
  exit 1
fi

REPO_ROOT="$(cd "$ROOT/../../../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

PY="$ROOT/nsys_targets.py"
# 默认 D=64。D=512 时教学 flash 很容易 SRAM OOM：D=512 bash run_nsys.sh
B="${B:-4}"
S="${S:-256}"
D="${D:-128}"

for impl in torch naive flash; do
  echo "======== nsys $impl  shape=($B,$S,$D) ========"
  rm -f "$OUT/$impl.nsys-rep" "$OUT/$impl.sqlite"
  set +e
  nsys profile \
    --stats=true \
    --force-overwrite=true \
    --trace=cuda,osrt \
    --capture-range=cudaProfilerApi \
    -o "$OUT/$impl" \
    python "$PY" --impl "$impl" --b "$B" --s "$S" --d "$D"
  rc=$?
  set +e
  if [ ! -f "$OUT/$impl.nsys-rep" ]; then
    echo "错误: $impl 没有新报告（python/nsys 失败，退出码=$rc）。不沿用旧 .nsys-rep。"
    exit 1
  fi
  if [ "$rc" -ne 0 ]; then
    echo "提示: nsys $impl 退出码=$rc（143=SIGTERM，SKIPPED 统计常见；报告已生成则无妨）"
  fi
  echo "report -> $OUT/$impl.nsys-rep"
done

echo
echo "写成 markdown 对照..."
python "$ROOT/nsys_to_md.py" "$OUT/compare.md"
echo "对照: $OUT/compare.md"
