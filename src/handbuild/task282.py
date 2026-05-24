"""task282 — 3x3 ring around each color-5 marker.

Rule:
- For each cell with value 5 in the input, paint a 3x3 region centered at it:
    corners → 5
    edges (4-neighbors) → 1
    center (the marker itself) → 0
- Clipped at grid borders.
- Regions of different markers do not overlap on conflicting cells in the
  observed examples.
"""

from __future__ import annotations
import numpy as np


def solve(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64)
    H, W = grid.shape
    out = np.zeros_like(grid)
    # 3x3 stamp: corners=5, edges+center=1
    stamp = np.array([
        [5, 1, 5],
        [1, 0, 1],
        [5, 1, 5],
    ], dtype=np.int64)
    for r, c in zip(*np.where(grid == 5)):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = stamp[dr + 1, dc + 1]
    return out


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from handbuild.validate_rule import report
    report(282, solve)
