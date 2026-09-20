"""把 tutorial 各课目录加进 sys.path，供实战 import。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = ROOT / "kernel" / "triton" / "tutorial"

for sub in ("05_norm", "02_activations", "06_attention"):
    p = str(TUTORIAL / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
