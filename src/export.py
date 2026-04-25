"""Export PyTorch checkpoints to ONNX format for competition submission.

Usage:
    python src/export.py --task 1             # export task001.pt → task001.onnx
    python src/export.py --all                # export all available checkpoints
    python src/export.py --all --start 50     # resume from task050
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from model import TinyCNN

BASE_DIR = pathlib.Path(__file__).parent.parent
CHECKPOINT_DIR = BASE_DIR / "output" / "checkpoints"
ONNX_DIR = BASE_DIR / "output" / "onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

NUM_TASKS = 400
_DUMMY_INPUT = torch.zeros(1, 10, 30, 30)  # shape matches competition I/O contract
_FILESIZE_LIMIT = 1.44 * 1024 * 1024


def export_task(task_num, hidden=16):
    ckpt_path = CHECKPOINT_DIR / f"task{task_num:03d}.pt"
    if not ckpt_path.exists():
        print(f"[task{task_num:03d}] No checkpoint found, skipping.")
        return False

    onnx_path = ONNX_DIR / f"task{task_num:03d}.onnx"

    model = TinyCNN(hidden=hidden)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    torch.onnx.export(
        model,
        _DUMMY_INPUT,
        str(onnx_path),
        dynamo=False,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
    )

    size = onnx_path.stat().st_size
    status = "OK" if size <= _FILESIZE_LIMIT else "OVER LIMIT"
    print(f"[task{task_num:03d}] Exported → {onnx_path.name} ({size:,} bytes) [{status}]")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, help="Single task number to export")
    parser.add_argument("--all", action="store_true", help="Export all checkpoints")
    parser.add_argument("--start", type=int, default=1, help="Start task (used with --all)")
    parser.add_argument("--hidden", type=int, default=16, help="Hidden channels in TinyCNN")
    args = parser.parse_args()

    if args.all:
        exported = 0
        for t in range(args.start, NUM_TASKS + 1):
            if export_task(t, hidden=args.hidden):
                exported += 1
        print(f"\nExported {exported} ONNX files to {ONNX_DIR}")
    elif args.task:
        export_task(args.task, hidden=args.hidden)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
