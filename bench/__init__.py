"""Model x operator x execution-mode (eager / CUDA graph) benchmark harness.

Entry point: ``python -m bench <run|profile|e2e|report|list> ...`` from the
``kernels`` repo root. See ``bench/README.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
