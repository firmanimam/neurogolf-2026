"""task004 — shift parallelogram outline toward bottom by 1 column.

Rule (derived from train + arc-gen examples):
- Each non-zero color forms a parallelogram outline (top edge, bottom edge,
  and two diagonal sides) slanting down-right. The bottom edge is the
  rightmost (highest column-index) edge.
- For each colored shape:
    * Identify the bottom row of the shape (the highest row index with that
      color present).
    * For every cell of the shape NOT in that bottom row, shift its column
      right by 1.
    * If the shifted column would exceed the maximum column of the bottom
      row, clamp it to that maximum column instead.
    * Bottom-row cells stay in place.
- Multiple disjoint shapes (different colors) are processed independently.
"""

from __future__ import annotations
import numpy as np


def solve(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(grid)
    H, W = grid.shape

    colors = [c for c in np.unique(grid).tolist() if c != 0]
    for color in colors:
        mask = grid == color
        rows_with = np.where(mask.any(axis=1))[0]
        # The shape could span a contiguous range of rows; but to be safe
        # treat just the cells with this color.
        if rows_with.size == 0:
            continue
        bottom_row = int(rows_with.max())
        bottom_cols = np.where(mask[bottom_row])[0]
        max_bot_col = int(bottom_cols.max())

        # Bottom row stays as-is.
        out[bottom_row, bottom_cols] = color

        # Other rows: shift right by 1, clamp at max_bot_col.
        for r in rows_with:
            if r == bottom_row:
                continue
            cols = np.where(mask[r])[0]
            new_cols = cols + 1
            new_cols = np.minimum(new_cols, max_bot_col)
            # Clamp to grid width just in case.
            new_cols = np.minimum(new_cols, W - 1)
            out[r, new_cols] = color

    return out


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from handbuild.validate_rule import report
    report(4, solve)
