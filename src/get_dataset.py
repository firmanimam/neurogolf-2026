import json
import numpy as np
import torch
from torch.utils.data import Dataset

CHANNELS = 10
HEIGHT = 30
WIDTH = 30


def grid_to_tensor(grid):
    """Convert ARC color grid (2D list of ints 0-9) to one-hot tensor (10, 30, 30)."""
    t = np.zeros((CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            t[color, r, c] = 1.0
    return t


def load_task(task_path):
    with open(task_path) as f:
        return json.load(f)


class ARCDataset(Dataset):
    """Loads ARC task examples from a JSON file as (input, output) tensor pairs.

    Args:
        task_path: Path to taskXXX.json
        splits: Which splits to include. Train uses ("train", "arc-gen"),
                validation uses ("test",).
    """

    def __init__(self, task_path, splits=("train", "arc-gen")):
        data = load_task(task_path)
        self.pairs = []
        for split in splits:
            for ex in data.get(split, []):
                if self._is_valid(ex):
                    inp = torch.from_numpy(grid_to_tensor(ex["input"]))
                    out = torch.from_numpy(grid_to_tensor(ex["output"]))
                    self.pairs.append((inp, out))

    def _is_valid(self, ex):
        """Skip grids larger than 30x30."""
        grid = ex["input"]
        return len(grid) <= HEIGHT and len(grid[0]) <= WIDTH

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
