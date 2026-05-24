"""task016 — fixed involution color swap.

Rule:
- Each cell's color is mapped by the involution:
    1 <-> 5
    2 <-> 6
    3 <-> 4
    8 <-> 9
    0 and 7 map to themselves
- Shape is preserved.
"""

from __future__ import annotations
import numpy as np

_MAP = np.array([0, 5, 6, 4, 3, 1, 2, 7, 9, 8], dtype=np.int64)


def solve(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64)
    return _MAP[grid]


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from handbuild.validate_rule import report
    report(16, solve)
