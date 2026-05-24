# NeuroGolf 2026

Work-in-progress submission for the [2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026) (IJCAI-ECAI 2026 / The Neurosynthetic Research Institute).

## What the competition is

For each of 400 ARC-AGI image-transformation tasks, submit one ONNX file. The file must produce the correct output grid for every train/test/arc-gen example. Each task is then scored:

```
points = max(1.0, 25.0 - log(macs + memory_bytes + params))
```

A task that gets any example wrong is worth **0**. A task that's solved correctly with a smaller graph scores higher. Maximum ~25 points × 400 tasks = ~10,000 points theoretical ceiling.

Forbidden ONNX ops: `LOOP`, `SCAN`, `NONZERO`, `UNIQUE`, `SCRIPT`, `FUNCTION`. Each ONNX file ≤ 1.44 MB.

## Current standing

- **Best public LB: 6,044.76** (rank 164 / 1,319 teams)
- **Bronze cutoff: 6,062.21** (top 10%)
- **Gap to bronze: ~17 LB**

## Strategy

The submission is assembled as a per-task selection of static ONNX graphs. The current pipeline:

1. **Base layer — `6042` public bundle.** The best single public bundle (`octaviograu/6042-85-per-task-hand-built-onnx-solvers`) scores 6,042.85 alone. It contributes the ONNX for all 400 tasks as a starting floor.
2. **Hand-built overlays.** For specific tasks where we can express the transformation as a small static ONNX graph, we replace the bundle's ONNX with our own. The verified gain so far: `task389` (+1.91 LB).

The fundamental insight is that trained CNNs cannot compete with hand-crafted ONNX on this scoring formula — the log-cost penalty rewards graphs of a few hundred bytes, which is below the parameter budget of even a tiny convolutional model. The leaderboard above ~6,000 LB consists entirely of hand-built per-task solvers.

### Hand-build workflow

For each candidate task:

1. **View grids:** `python3 src/view_task.py <task_id>` renders all train/test/arc-gen pairs as ASCII.
2. **Derive a Python rule:** write `src/handbuild/task<NNN>.py` with a `solve(grid)` function, then run it to verify it passes every example (typically 265+/265+).
3. **Build the static ONNX:** write `src/handbuild/build_task<NNN>.py` using only the allowed op set. Save to `output/handbuild_onnx/task<NNN>.onnx`.
4. **Validate and measure:** `python3 src/handbuild/test_onnx.py <task_id>` re-runs against all examples and reports the cost vs the public bundle's cost for the same task.
5. **Package and submit:** overlay the new ONNX onto the 6042 base bundle and submit via the Kaggle CLI.

### Findings on what scores well

- **Pure channel-permutation tasks** (e.g. `output[c] = input[perm[c]]`) survive the grader's hidden test cases because they make no assumption about grid shape. `task389` is the only confirmed-winning hand-build in this category.
- **Trivial constant swaps** (e.g. "color 5 ↔ color 8") already match the public bundle's cost at the theoretical floor of ~36 KB (single `Gather` + output tensor memory). No gain available.
- **Positional / spatial-stamp rules that hardcode the example grid size** consistently fail the grader despite passing every local example — the leading hypothesis is that the grader has hidden test cases with grid sizes that don't appear in our splits. Four such hand-builds (`task261`, `task095`, `task317`, `task282`) were each verified 265+/265+ locally but lost 4–18 LB when submitted.
- **The remaining gold seam:** tasks where the 6042 bundle's ONNX is materially above the theoretical floor *and* the rule can be expressed purely at the channel level (no positional logic). These are rare and require per-task cost inspection of the bundle to find.

For the full session-by-session log of what was tried, what worked, and what didn't, see [`strategy.md`](strategy.md).

## Repository layout

```
neurogolf-2026/
├── strategy.md                       # Running handover / progress log
├── analysis.md                       # Initial analysis of the top public kernels
├── requirements.txt                  # Python deps (onnx, onnxruntime, onnx-tool, numpy)
│
├── src/
│   ├── view_task.py                  # ASCII task viewer
│   ├── analyze_candidates.py         # Per-source per-task cost matrix builder
│   ├── prioritize_handbuild.py       # Ranks tasks by potential LB gain
│   └── handbuild/
│       ├── validate_rule.py          # Python rule → all-example validator
│       ├── test_onnx.py              # ONNX validation + cost + delta vs current winner
│       ├── task<NNN>.py              # Hand-derived Python rule per task
│       └── build_task<NNN>.py        # Static-ONNX builder per task
│
└── notebooks/
    ├── ensemble_kaggle.ipynb         # Kaggle GPU notebook that assembles & validates the submission
    └── ensemble-kernel-metadata.json # Kaggle kernel attachment metadata
```

The competition data (`task001.json … task400.json`, `neurogolf_utils/`) and all generated outputs (`output/`, `submit/`) are gitignored. Pull the data with `kaggle competitions download -c neurogolf-2026` before running anything locally.

## Setup

```bash
pip install -r requirements.txt
kaggle competitions download -c neurogolf-2026 -p .
unzip neurogolf-2026.zip
```

To validate an existing hand-build:

```bash
python3 src/handbuild/test_onnx.py 389
```

To assemble and submit a new combination on top of the 6042 base:

```bash
# Assumes the 6042 bundle's submission.zip is at the path below.
mkdir -p /tmp/vNEW
unzip -q -o /tmp/bundle-tests/best-score-the-2026-neurogolf-championship/submission.zip \
    -d /tmp/vNEW/onnx_files
cp output/handbuild_onnx/task389.onnx /tmp/vNEW/onnx_files/   # any number of overlays
cd /tmp/vNEW/onnx_files && zip -q -r /tmp/vNEW/submission.zip *.onnx && cd -
kaggle competitions submit -c neurogolf-2026 -f /tmp/vNEW/submission.zip -m "description"
```

## License

Not licensed for redistribution — this is a personal competition repository.
