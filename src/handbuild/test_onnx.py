"""Test a hand-built ONNX: (1) run inference against all task examples,
(2) compute (macs, mem, params) cost via onnx_tool to predict LB delta.

Usage:
    python3 src/handbuild/test_onnx.py 389
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import onnx_tool
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATASET = ROOT / "dataset"
ONNX_DIR = ROOT / "output" / "handbuild_onnx"

_CHANNELS, _HEIGHT, _WIDTH = 10, 30, 30
_EXCLUDED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION"}


def encode(grid):
    h, w = len(grid), len(grid[0])
    if max(h, w) > 30:
        return None, None
    arr = np.zeros((1, _CHANNELS, _HEIGHT, _WIDTH), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, col in enumerate(row):
            arr[0, col, r, c] = 1.0
    return arr, (h, w)


def main():
    if len(sys.argv) < 2:
        print("usage: test_onnx.py <task_id>")
        sys.exit(1)
    tid = int(sys.argv[1])
    onnx_path = ONNX_DIR / f"task{tid:03d}.onnx"
    if not onnx_path.exists():
        print(f"Missing: {onnx_path}")
        sys.exit(2)

    # ── 1) Run inference against all examples ──────────────────────────────
    with open(DATASET / f"task{tid:03d}.json") as f:
        t = json.load(f)
    examples = t["train"] + t["test"] + t.get("arc-gen", [])

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    right = wrong = 0
    failures = []
    for i, ex in enumerate(examples):
        inp, _ = encode(ex["input"])
        exp, _ = encode(ex["output"])
        if inp is None:
            continue
        pred = (sess.run(["output"], {"input": inp})[0] > 0.0).astype(float)
        if np.array_equal(pred, exp):
            right += 1
        else:
            wrong += 1
            if len(failures) < 3:
                failures.append(i)
    total = right + wrong
    print(f"Validation: {right}/{total} pass, {wrong} fail")
    if failures:
        print(f"First failed example indices: {failures}")
        return

    # ── 2) Profile cost ────────────────────────────────────────────────────
    m = onnx_tool.loadmodel(str(onnx_path), {"verbose": False})
    g = m.graph
    g.graph_reorder_nodes()
    g.shape_infer(None)
    g.profile()
    if not g.valid_profile:
        print("Profiling FAILED")
        return
    for key in g.nodemap:
        op = g.nodemap[key].op_type.upper()
        if op in _EXCLUDED:
            print(f"FORBIDDEN OP: {op}")
            return
    macs = int(sum(g.macs))
    mem = int(g.memory)
    params = int(g.params)
    cost = macs + mem + params
    pts = max(1.0, 25.0 - math.log(max(1, cost)))

    print(f"\nCost breakdown:")
    print(f"  macs    = {macs:>12,}")
    print(f"  memory  = {mem:>12,}")
    print(f"  params  = {params:>12,}")
    print(f"  TOTAL   = {cost:>12,}")
    print(f"  points  = {pts:.3f}")

    # ── 3) Compare to current best for this task ───────────────────────────
    import csv
    cur_cost = None
    with open(ROOT / "output" / "candidates.csv") as f:
        for row in csv.DictReader(f):
            if int(row["task_id"]) != tid or int(row["valid"]) != 1:
                continue
            c = int(row["cost"])
            if cur_cost is None or c < cur_cost:
                cur_cost = c
    if cur_cost is not None:
        cur_pts = max(1.0, 25.0 - math.log(cur_cost))
        delta = pts - cur_pts
        print(f"\nCurrent winner cost = {cur_cost:,}  pts = {cur_pts:.3f}")
        print(f"New hand-build cost = {cost:,}  pts = {pts:.3f}")
        print(f"Delta:                 {delta:+.3f} pts")
        if delta > 0:
            print(f"WIN — adopt this hand-build for task{tid:03d}")
        else:
            print(f"LOSS — keep current winner")


if __name__ == "__main__":
    main()
