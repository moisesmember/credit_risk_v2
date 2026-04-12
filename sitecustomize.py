"""Project-local Python path bootstrap.

This keeps the src layout importable in scripts, notebooks and ad-hoc shells
started from the repository root without requiring editable installs.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    src_as_text = str(SRC_PATH)
    if src_as_text not in sys.path:
        sys.path.insert(0, src_as_text)
