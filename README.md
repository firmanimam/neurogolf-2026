# NeuroGolf 2026

Work-in-progress submission for the [2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026) (IJCAI-ECAI 2026 / The Neurosynthetic Research Institute).

## The competition

For each of 400 ARC-AGI image-transformation tasks, submit one ONNX file. The file must produce the correct output grid for every train / test / arc-gen example. Each task is then scored:

```
points = max(1.0, 25.0 - log(macs + memory_bytes + params))
```

A task that gets any example wrong is worth **0**. A task that's solved correctly with a smaller graph scores higher. Maximum ~25 points × 400 tasks ≈ 10,000 points theoretical ceiling.

Hard constraints: each ONNX file ≤ 1.44 MB; forbidden ops are `LOOP`, `SCAN`, `NONZERO`, `UNIQUE`, `SCRIPT`, `FUNCTION`.

## Strategy: per-task hand-built static ONNX

Because score = `25 − log(cost)`, the optimal graph for any task is the smallest static ONNX that still produces the correct output. We treat each of the 400 tasks as a separate small reverse-engineering problem: read the train/test/arc-gen examples, derive the transformation rule by inspection, then write the graph by hand.

The pipeline for each task is:

1. **View the grids** — `python3 src/view_task.py <task_id>` renders every example pair as ASCII so the transformation can be read off directly.
2. **Derive a Python rule** — write `src/handbuild/task<NNN>.py` with a `solve(grid)` function and validate it against all examples. The rule must pass 100% before any ONNX work begins.
3. **Build the static ONNX** — write `src/handbuild/build_task<NNN>.py` using only the allowed op set. The graph is fully constant-folded; there is no training and no learned weight.
4. **Validate and measure** — `python3 src/handbuild/test_onnx.py <task_id>` re-runs the ONNX against every example via `onnxruntime` and reports the cost via `onnx_tool`.
5. **Package and submit** — combine the per-task ONNX files into `submission.zip` and upload via the Kaggle CLI.

### What we've learned about the cost surface

A few patterns have shown up repeatedly:

- **The output tensor sets the floor.** A `(1, 10, 30, 30)` float32 output costs ~36 KB of memory regardless of how the graph computes it, so the minimum reachable cost for any task is ~36,090. Tasks where this floor is reachable are mostly **pure channel-permutation** rules (e.g. "swap color 5 with color 8 everywhere"), which collapse to a single `Gather` over a constant index vector.
- **Runtime channel inspection still permutation-friendly.** Tasks where the output is a permutation but the permutation depends on which colors are present in the input (e.g. "find the most-frequent color and recolor everything to it") can be solved with `ReduceSum` + `ArgMax` + `Gather` on the channel axis without ever touching the spatial dimensions.
- **Positional / spatial-stamp rules are dangerous.** Several rules that involve writing a fixed 3×3 pattern around marker cells, or that hardcode the example grid size into the graph, pass every local example but fail on the grader. The leading hypothesis is that the grader's hidden examples include grid sizes that don't appear in the public splits, so any solver that bakes in a shape silently produces wrong output for those cells. Until this is understood, the safest hand-builds are the ones that operate purely on the channel axis and let the spatial dimensions flow through unmodified.

The full session-by-session log of attempted builds, costs, and leaderboard deltas is tracked privately and not part of the public repo.

## Repository layout

```
neurogolf-2026/
├── requirements.txt                  # onnx, onnxruntime, onnx-tool, numpy
│
├── src/
│   ├── view_task.py                  # ASCII task viewer
│   ├── analyze_candidates.py         # Per-task cost matrix builder
│   ├── prioritize_handbuild.py       # Ranks tasks by potential LB gain
│   └── handbuild/
│       ├── validate_rule.py          # Python rule → all-example validator
│       ├── test_onnx.py              # ONNX validation + cost measurement
│       ├── task<NNN>.py              # Hand-derived Python rule per task
│       └── build_task<NNN>.py        # Static-ONNX builder per task
│
└── notebooks/
    ├── ensemble_kaggle.ipynb         # Kaggle notebook that assembles + validates the submission
    └── ensemble-kernel-metadata.json # Kaggle kernel attachment metadata
```

Competition data (`task001.json … task400.json`, `neurogolf_utils/`) and all generated outputs (`output/`, `submit/`) are gitignored. Pull the data with `kaggle competitions download -c neurogolf-2026` before running anything locally.

## Setup

```bash
pip install -r requirements.txt
kaggle competitions download -c neurogolf-2026 -p .
unzip neurogolf-2026.zip
```

To validate an existing hand-build end-to-end:

```bash
python3 src/handbuild/test_onnx.py 389
```

## License

Not licensed for redistribution — this is a personal competition repository.
