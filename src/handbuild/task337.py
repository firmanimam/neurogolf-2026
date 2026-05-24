"""task337 — swap colors 5 and 8.

Rule:
- Identity transform on all cells, except colors 5 and 8 are swapped.
- 5 -> 8, 8 -> 5, all other colors unchanged.
"""

from __future__ import annotations
import numpy as np


def solve(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64)
    out = grid.copy()
    out[grid == 5] = 8
    out[grid == 8] = 5
    return out


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from handbuild.validate_rule import report
    report(337, solve)
