"""task276 — constant color swap 6 -> 2.

Rule:
- Replace every cell with color 6 by color 2.
- All other colors (including 7 and 0) are unchanged.
"""

from __future__ import annotations
import numpy as np


def solve(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64)
    out = grid.copy()
    out[grid == 6] = 2
    return out


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from handbuild.validate_rule import report
    report(276, solve)
