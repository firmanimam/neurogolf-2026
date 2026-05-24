"""task309 — constant recolor 7->5.

Rule:
- Every cell with color 7 becomes color 5.
- All other cells are preserved as-is.
"""

from __future__ import annotations
import numpy as np


def solve(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64)
    out = grid.copy()
    out[grid == 7] = 5
    return out


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from handbuild.validate_rule import report
    report(309, solve)
