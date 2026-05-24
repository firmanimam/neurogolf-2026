"""task129 — fill output with the most-frequent color of the input.

Rule:
- Output grid has same shape as input.
- Every cell of the output is set to the color that appears most often in the input.
- Expressible as channel-level statistic: ArgMax over per-channel ReduceSum.
"""

from __future__ import annotations
import numpy as np


def solve(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64)
    counts = np.bincount(grid.flatten(), minlength=10)
    color = int(np.argmax(counts))
    return np.full_like(grid, color)


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from handbuild.validate_rule import report
    report(129, solve)
