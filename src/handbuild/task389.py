"""task389 — color-5 marker swap.

Rule (derived from train examples):
- Input always contains exactly two non-zero colors: 5 and one other (X).
- Output:
    * positions where input == 5 → output color X
    * positions where input == X → output color 0
    * other positions (if any) → 0

I.e. color 5 marks "where to put the X", and X's original positions get cleared.
"""

from __future__ import annotations
import numpy as np


def solve(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64)
    # Find X: the non-zero color that isn't 5.
    unique = set(grid.flatten().tolist())
    others = [c for c in unique if c != 0 and c != 5]
    if not others:
        # No "other" color — return zeros (shouldn't happen on valid examples).
        return np.zeros_like(grid)
    X = others[0]
    out = np.zeros_like(grid)
    out[grid == 5] = X
    return out


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from handbuild.validate_rule import report
    report(389, solve)
