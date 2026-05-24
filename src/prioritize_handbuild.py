"""Phase 4 — Prioritize hand-build targets.

Goal: list tasks where replacing the current best ONNX with a hand-built
static graph at cost ≤ 500 would yield the largest LB gain.

Scoring formula: points = max(1, 25 - ln(cost))
- Current winner at cost  10,000 → 15.79 pts
- Hand-build at cost 200          → 19.70 pts  → gain of  3.91 pts
- Current winner at cost 100,000 → 13.48 pts
- Hand-build at cost 200          → 19.70 pts  → gain of  6.22 pts
- Current winner at cost 500,000 → 11.87 pts
- Hand-build at cost 200          → 19.70 pts  → gain of  7.83 pts

Prioritize by (potential gain, smaller grid sizes, fewer unique colors).
Smaller grids and fewer colors usually mean simpler rules → faster derivation.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
CANDIDATES = ROOT / "output" / "candidates.csv"
DATASET = ROOT / "dataset"

HANDBUILD_TARGET_COST = 200  # what we aim for with a hand-built ONNX


def pts(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


def task_features(task_id: int) -> dict:
    """Compute simple complexity metrics from train+test examples."""
    p = DATASET / f"task{task_id:03d}.json"
    with open(p) as f:
        t = json.load(f)
    pairs = t["train"] + t["test"]
    grid_sizes = []
    colors_per_grid = []
    shape_changes = 0
    for ex in pairs:
        gi, go = ex["input"], ex["output"]
        h_i, w_i = len(gi), len(gi[0])
        h_o, w_o = len(go), len(go[0])
        grid_sizes.append(max(h_i, w_i, h_o, w_o))
        colors_per_grid.append(len({c for row in gi for c in row} | {c for row in go for c in row}))
        if (h_i, w_i) != (h_o, w_o):
            shape_changes += 1
    return {
        "n_train": len(t["train"]),
        "n_arc_gen": len(t.get("arc-gen", [])),
        "max_grid": max(grid_sizes),
        "n_colors": int(sum(colors_per_grid) / len(colors_per_grid)),
        "shape_changes": shape_changes,  # 0 = output same size as input
    }


def main():
    # Load winning cost per task
    winners = {}  # tid -> cost
    with open(CANDIDATES) as f:
        for row in csv.DictReader(f):
            if int(row["valid"]) != 1:
                continue
            tid = int(row["task_id"])
            cost = int(row["cost"])
            if tid not in winners or cost < winners[tid]:
                winners[tid] = cost

    # For each task, compute potential gain and complexity
    candidates = []
    for tid in range(1, 401):
        if tid not in winners:
            continue
        cur_cost = winners[tid]
        cur_pts = pts(cur_cost)
        hb_pts = pts(HANDBUILD_TARGET_COST)
        gain = hb_pts - cur_pts
        feats = task_features(tid)
        # Composite priority score: high gain, small grid, few colors
        priority = gain * 10 - feats["max_grid"] - feats["n_colors"]
        candidates.append({
            "task_id": tid,
            "cur_cost": cur_cost,
            "cur_pts": cur_pts,
            "potential_gain": gain,
            "priority": priority,
            **feats,
        })

    # Sort by priority descending
    candidates.sort(key=lambda x: -x["priority"])

    print(f"{'task':<6}{'cur_cost':>10}{'cur_pts':>8}{'gain':>7}"
          f"{'maxg':>5}{'nclr':>5}{'shape_ch':>10}")
    print("-" * 60)
    for c in candidates[:40]:
        print(f"task{c['task_id']:03d}{c['cur_cost']:>10}"
              f"{c['cur_pts']:>8.2f}{c['potential_gain']:>+7.2f}"
              f"{c['max_grid']:>5}{c['n_colors']:>5}{c['shape_changes']:>10}")

    print()
    print(f"Total tasks rankable: {len(candidates)}")
    print(f"Tasks where hand-build would gain >5 pts: "
          f"{sum(1 for c in candidates if c['potential_gain'] > 5)}")
    print(f"Tasks where hand-build would gain >2 pts: "
          f"{sum(1 for c in candidates if c['potential_gain'] > 2)}")


if __name__ == "__main__":
    main()
