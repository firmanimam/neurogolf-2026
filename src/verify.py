"""Locally verify ONNX models against competition scoring rules.

Mirrors the logic of neurogolf_utils.verify_network without IPython dependencies.

Usage:
    python src/verify.py --task 1          # verify task001.onnx
    python src/verify.py --all             # verify all exported ONNX files
    python src/verify.py --all --quiet     # only print failures
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import onnxruntime as ort

BASE_DIR = pathlib.Path(__file__).parent.parent
ONNX_DIR = BASE_DIR / "output" / "onnx"
DATA_DIR = BASE_DIR

CHANNELS, HEIGHT, WIDTH = 10, 30, 30
FILESIZE_LIMIT = 1.44 * 1024 * 1024
NUM_TASKS = 400


def grid_to_numpy(grid):
    t = np.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            t[0, color, r, c] = 1.0
    return t


def run_model(session, grid):
    result = session.run(["output"], {"input": grid_to_numpy(grid)})
    return (result[0] > 0.0).astype(float)


def check_examples(session, examples):
    correct, total = 0, 0
    for ex in examples:
        if max(len(ex["input"]), len(ex["input"][0])) > 30:
            continue
        pred = run_model(session, ex["input"])
        expected = grid_to_numpy(ex["output"])
        total += 1
        if np.array_equal(pred, expected):
            correct += 1
    return correct, total


def verify_task(task_num, quiet=False):
    onnx_path = ONNX_DIR / f"task{task_num:03d}.onnx"
    task_path = DATA_DIR / f"task{task_num:03d}.json"
    tag = f"[task{task_num:03d}]"

    if not onnx_path.exists():
        if not quiet:
            print(f"{tag} No ONNX file found.")
        return None

    size = onnx_path.stat().st_size
    if size > FILESIZE_LIMIT:
        print(f"{tag} FAIL — file too large: {size:,} bytes (limit {FILESIZE_LIMIT:,.0f})")
        return 0

    with open(task_path) as f:
        data = json.load(f)

    try:
        session = ort.InferenceSession(str(onnx_path))
    except Exception as e:
        print(f"{tag} FAIL — cannot load ONNX: {e}")
        return 0

    arc_ok, arc_total = check_examples(session, data["train"] + data["test"])
    gen_ok, gen_total = check_examples(session, data.get("arc-gen", []))
    all_ok = (arc_ok == arc_total) and (gen_ok == gen_total)

    if all_ok:
        # Estimate score (requires onnx_tool for exact MACs; use file size as proxy)
        try:
            import onnx_tool
            model = onnx_tool.loadmodel(str(onnx_path), {"verbose": False})
            g = model.graph
            g.graph_reorder_nodes()
            g.shape_infer(None)
            g.profile()
            macs = int(sum(g.macs))
            memory = int(g.memory)
            params = int(g.params)
            points = max(1.0, 25.0 - math.log(macs + memory + params))
            stat = f"MACs={macs:,} mem={memory:,} params={params:,} → {points:.3f} pts"
        except Exception:
            points = None
            stat = f"size={size:,} bytes (install onnx_tool for exact score)"

        if not quiet:
            print(f"{tag} PASS — arc-agi {arc_ok}/{arc_total}, arc-gen {gen_ok}/{gen_total} | {stat}")
        return points if points else 1.0
    else:
        print(f"{tag} FAIL — arc-agi {arc_ok}/{arc_total}, arc-gen {gen_ok}/{gen_total}")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, help="Single task to verify")
    parser.add_argument("--all", action="store_true", help="Verify all ONNX files")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--quiet", action="store_true", help="Only print failures")
    args = parser.parse_args()

    if args.all:
        total_points = 0.0
        passed = 0
        checked = 0
        for t in range(args.start, NUM_TASKS + 1):
            result = verify_task(t, quiet=args.quiet)
            if result is not None:
                checked += 1
                if result > 0:
                    passed += 1
                    total_points += result
        print(f"\nSummary: {passed}/{checked} tasks pass | estimated score: {total_points:.2f}")
    elif args.task:
        verify_task(args.task, quiet=args.quiet)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
