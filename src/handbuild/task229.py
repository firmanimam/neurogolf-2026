"""task229 — keep most-frequent non-zero color, recolor all others to 5.

Rule (derived from all 267 examples):
- Determine "anchor" color = the non-zero color with the highest count in the input
  (ties broken by lowest color index).
- Output:
    * positions where input == 0 → 0
    * positions where input == anchor → anchor (preserved)
    * positions where input is any other non-zero color → 5

Channel-level expressible: per-channel spatial sum gives counts, argmax (with
lowest-index tiebreak) picks the anchor channel.
"""

from __future__ import annotations
import numpy as np
import collections


def solve(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64)
    flat = grid[grid != 0].flatten().tolist()
    if not flat:
        return np.zeros_like(grid)
    cnt = collections.Counter(flat)
    maxc = max(cnt.values())
    anchor = min(c for c, v in cnt.items() if v == maxc)
    out = np.where(grid == 0, 0, np.where(grid == anchor, anchor, 5))
    return out.astype(np.int64)


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from handbuild.validate_rule import report
    report(229, solve)
