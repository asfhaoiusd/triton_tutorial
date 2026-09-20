#!/usr/bin/env bash
# Create ai_infra venv with PyTorch cu128 + Triton (RTX 50-series / Blackwell).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  uv venv .venv --python "$PYTHON_VERSION"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install -r requirements.txt --index-url "$TORCH_INDEX"

python - <<'PY'
import torch
import triton

assert torch.cuda.is_available(), "CUDA not available"
print("torch ", torch.__version__)
print("cuda  ", torch.version.cuda)
print("triton", triton.__version__)
print("device", torch.cuda.get_device_name(0))
print("cap   ", torch.cuda.get_device_capability(0))
x = torch.randn(1024, device="cuda")
assert torch.allclose(x + x, x * 2)
print("OK: CUDA + Triton environment ready")
PY
