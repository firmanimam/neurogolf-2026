"""task267 — shape recolored by single-cell marker.

Rule:
- Input has two non-zero colors: a "shape" color X (many cells)
  and a "marker" color Y (exactly one cell, typically at bottom-left).
- Output:
    * positions where input == X  -> color Y
    * position where input == Y   -> 0
    * all other cells             -> 0
- X is the more frequent non-zero color; Y is the less frequent one.
"""

from __future__ import annotations
import numpy as np


def solve(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(grid)
    counts = {}
    for c in range(1, 10):
        n = int((grid == c).sum())
        if n > 0:
            counts[c] = n
    if len(counts) < 2:
        return out
    # X = most frequent non-zero, Y = least frequent non-zero
    X = max(counts, key=counts.get)
    Y = min(counts, key=counts.get)
    out[grid == X] = Y
    return out


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from handbuild.validate_rule import report
    report(267, solve)
