"""Path normalisation for index records.

Index files (`*.jsonl`) are sometimes built on Windows, where `str(Path(...))`
emits backslash separators (`..\\visuals_dataset\\output\\...`). Those strings are
not openable on Linux, so training on HiPerGator fails with FileNotFoundError.

Builders now emit POSIX separators via `Path.as_posix()`, but indexes generated
before that fix still exist, so every read path also normalises defensively.
Backslashes are legal in POSIX filenames in principle; no path in this dataset
contains one, so the substitution is safe here.
"""

from typing import Iterable, List


def to_posix(path) -> str:
    """Normalise a single index path to forward slashes."""
    return str(path).replace("\\", "/")


def to_posix_all(paths: Iterable) -> List[str]:
    """Normalise a list of index paths (e.g. a flow-frame neighbour list)."""
    return [to_posix(p) for p in paths]
