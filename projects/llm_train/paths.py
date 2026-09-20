"""tutorial 课目录进 sys.path。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = ROOT / "kernel" / "triton" / "tutorial"

for sub in ("", "02_activations", "05_norm", "06_attention"):
    p = str(TUTORIAL / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
