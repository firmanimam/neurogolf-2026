"""Validate a Python solve function against all examples for a task.

Usage:
    from handbuild.validate_rule import validate
    ok, fails = validate(389, solve_389)
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATASET = ROOT / "dataset"


def load_examples(task_id: int) -> list[dict]:
    with open(DATASET / f"task{task_id:03d}.json") as f:
        t = json.load(f)
    return t["train"] + t["test"] + t.get("arc-gen", [])


def validate(task_id: int, solve_fn) -> tuple[bool, list[dict]]:
    """Returns (all_passed, list_of_failures).

    solve_fn takes a 2D numpy array (grid) and returns a 2D numpy array.
    """
    examples = load_examples(task_id)
    failures = []
    for i, ex in enumerate(examples):
        try:
            inp = np.array(ex["input"], dtype=np.int64)
            exp = np.array(ex["output"], dtype=np.int64)
            out = solve_fn(inp)
            out = np.asarray(out, dtype=np.int64)
            if out.shape != exp.shape:
                failures.append({"i": i, "reason": "shape", "expected": exp.shape, "got": out.shape, "input": ex["input"]})
                continue
            if not np.array_equal(out, exp):
                failures.append({"i": i, "reason": "values", "expected": exp.tolist(), "got": out.tolist(), "input": ex["input"]})
        except Exception as e:
            failures.append({"i": i, "reason": f"exception: {e}", "input": ex["input"]})
    return len(failures) == 0, failures


def report(task_id: int, solve_fn, max_show: int = 3):
    ok, fails = validate(task_id, solve_fn)
    examples = load_examples(task_id)
    print(f"task{task_id:03d}: {len(examples) - len(fails)}/{len(examples)} pass"
          f"  ({'OK' if ok else 'FAIL'})")
    if fails:
        print(f"First {min(max_show, len(fails))} failures:")
        for f in fails[:max_show]:
            print(f"  example {f['i']}: {f['reason']}")
            if f["reason"] == "values":
                print(f"    input:    {f['input']}")
                print(f"    expected: {f['expected']}")
                print(f"    got:      {f['got']}")
    return ok
