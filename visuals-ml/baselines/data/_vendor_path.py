"""Put the vendored MonoDETR tree on sys.path so its `lib.*` / `utils.*` packages
import as top level (the upstream code uses those absolute import roots)."""

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "monodetr"


def ensure_vendor_on_path():
    p = str(_VENDOR)
    if p not in sys.path:
        sys.path.insert(0, p)
