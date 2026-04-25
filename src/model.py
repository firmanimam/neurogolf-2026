import torch.nn as nn


class TinyCNN(nn.Module):
    """Baseline CNN for NeuroGolf. ~4K params, fits well within 1.44 MB limit.

    Input/output shape: (1, 10, 30, 30) — one-hot color channels, 30x30 grid.
    Output is raw logits; a cell has color c if output[0][c][r][col] > 0.
    """

    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(10, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 10, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
