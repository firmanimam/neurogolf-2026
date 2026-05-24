"""task317 — 3x3 super-cell fill with 1 where any 5 marker is present.

Rule:
- The 9x9 grid is conceptually a 3x3 grid of 3x3 super-cells.
- For each super-cell, if it contains any cell with value 5, fill the entire
  super-cell with 1s in the output. Otherwise leave it 0.
"""

from __future__ import annotations
import numpy as np


def solve(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64)
    h, w = grid.shape
    out = np.zeros_like(grid)
    for br in range(0, h, 3):
        for bc in range(0, w, 3):
            block = grid[br:br+3, bc:bc+3]
            if np.any(block == 5):
                out[br:br+3, bc:bc+3] = 1
    return out


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from handbuild.validate_rule import report
    report(317, solve)
