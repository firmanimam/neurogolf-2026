"""Train one TinyCNN per ARC task, save checkpoints to output/checkpoints/.

Usage:
    python src/train.py --task 1                    # train task001
    python src/train.py --task 1 --epochs 1000      # custom epochs
    python src/train.py --all                        # train all 400 tasks
    python src/train.py --all --start 50             # resume from task050
"""

import argparse
import pathlib
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from get_dataset import ARCDataset
from model import TinyCNN, count_params

DATA_DIR = pathlib.Path(__file__).parent.parent
CHECKPOINT_DIR = DATA_DIR / "output" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

NUM_TASKS = 400


def train_task(task_num, epochs=2000, lr=0.01, device="cuda", verbose=True):
    task_path = DATA_DIR / f"task{task_num:03d}.json"
    if not task_path.exists():
        print(f"[task{task_num:03d}] File not found, skipping.")
        return False

    train_ds = ARCDataset(task_path, splits=("train", "arc-gen"))
    val_ds = ARCDataset(task_path, splits=("test",))

    if len(train_ds) == 0:
        print(f"[task{task_num:03d}] No training examples, skipping.")
        return False

    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True)

    model = TinyCNN(hidden=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, factor=0.5, min_lr=1e-5
    )
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step(total_loss)

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"task{task_num:03d}.pt")

        if total_loss < 1e-6:
            if verbose:
                print(f"[task{task_num:03d}] Converged at epoch {epoch}, loss={total_loss:.2e}")
            break

        if verbose and epoch % 200 == 0:
            print(f"[task{task_num:03d}] epoch={epoch}, loss={total_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")

    # Quick accuracy check on val set
    if len(val_ds) > 0:
        model.load_state_dict(torch.load(CHECKPOINT_DIR / f"task{task_num:03d}.pt", map_location=device))
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in DataLoader(val_ds, batch_size=1):
                x, y = x.to(device), y.to(device)
                pred = (model(x) > 0).float()
                correct += (pred == y).all().item()
        if verbose:
            print(f"[task{task_num:03d}] val accuracy: {correct}/{len(val_ds)} | params: {count_params(model)}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, help="Single task number to train")
    parser.add_argument("--all", action="store_true", help="Train all tasks")
    parser.add_argument("--start", type=int, default=1, help="Start task (used with --all)")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--cpu", action="store_true", help="Force CPU (default: CUDA if available)")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.all:
        for t in range(args.start, NUM_TASKS + 1):
            train_task(t, epochs=args.epochs, lr=args.lr, device=device)
    elif args.task:
        train_task(args.task, epochs=args.epochs, lr=args.lr, device=device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
