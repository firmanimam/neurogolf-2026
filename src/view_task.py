"""View a task's grid pairs as ASCII art to derive the transformation rule.

Usage:
    python3 src/view_task.py 389
    python3 src/view_task.py 389 --n 3   # show only 3 pairs
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"

# Color names for clarity in ASCII
COLOR_CHAR = "0123456789"  # 0-9 → characters


def render(grid, indent="  "):
    return "\n".join(indent + "".join(COLOR_CHAR[c] for c in row) for row in grid)


def render_pair(ex):
    gi, go = ex["input"], ex["output"]
    in_lines = render(gi).split("\n")
    out_lines = render(go).split("\n")
    h = max(len(in_lines), len(out_lines))
    in_lines += [" " * len(in_lines[0])] * (h - len(in_lines))
    out_lines += [" " * len(out_lines[0])] * (h - len(out_lines))
    return "\n".join(f"{a}    {b}" for a, b in zip(in_lines, out_lines))


def main():
    if len(sys.argv) < 2:
        print("usage: view_task.py <task_id> [--n N]")
        sys.exit(1)
    tid = int(sys.argv[1])
    n_show = 5
    if "--n" in sys.argv:
        n_show = int(sys.argv[sys.argv.index("--n") + 1])

    with open(DATASET / f"task{tid:03d}.json") as f:
        t = json.load(f)

    print(f"=== task{tid:03d} ===")
    print(f"train: {len(t['train'])} examples")
    print(f"test:  {len(t['test'])} examples")
    print(f"arc-gen: {len(t.get('arc-gen', []))} examples")
    print()

    all_pairs = t["train"] + t["test"]
    print(f"Showing first {min(n_show, len(all_pairs))} train+test pairs:")
    print(f"{'input':<20}    output")
    for i, ex in enumerate(all_pairs[:n_show]):
        print(f"--- pair {i} ---")
        print(render_pair(ex))
        print()


if __name__ == "__main__":
    main()
