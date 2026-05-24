"""task095 — 3x3 dilation around 5s, fill with 1, keep 5 at center.

Rule:
- Input contains 5s scattered on a 9x9 (or similar) grid.
- For each 5 in input:
    * Center cell remains color 5 in output.
    * The 8 surrounding cells (3x3 neighborhood minus center) become color 1.
- Cells far from any 5 stay color 0 (within the active grid).
- Cells outside the active grid remain "no color".
"""

from __future__ import annotations
import numpy as np


def solve(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64)
    h, w = grid.shape
    out = np.zeros_like(grid)
    # Find positions of 5s
    fives = np.argwhere(grid == 5)
    # Paint 3x3 of 1s first
    for r, c in fives:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr][nc] = 1
    # Overwrite 5 centers
    for r, c in fives:
        out[r][c] = 5
    return out


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from handbuild.validate_rule import report
    report(95, solve)
