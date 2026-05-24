"""task261 — shift down + recolor 8→2.

Rule (derived from train + visual inspection):
- Input contains only color 8 (and color 0 as background).
- Output: every 8 in input becomes a 2 shifted down by exactly 1 row.
- Original 8 positions become color 0.
- Cells outside the active grid remain "no color".

This is a fully STATIC rule — no dynamic colors to detect.
"""

from __future__ import annotations
import numpy as np


def solve(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(grid)
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 8:
                if r + 1 < h:
                    out[r + 1][c] = 2
    return out


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from handbuild.validate_rule import report
    report(261, solve)
