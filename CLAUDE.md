# NeuroGolf 2026 — Project Guide

## Competition Summary

**URL:** https://www.kaggle.com/competitions/neurogolf-2026
**Organizer:** The Neurosynthetic Research Institute (IJCAI-ECAI 2026)
**Prize:** $50,000 USD | **Deadline:** 2026-07-15 | **Team merger:** 2026-07-08

**Goal:** Build the smallest neural networks that correctly solve ARC-AGI image
transformation tasks. This is a "golf" competition — correctness is required,
then smaller model = higher score.

---

## Scoring Formula

```python
points = max(1.0, 25.0 - math.log(macs + memory_bytes + params))
```

- A task is worth **0 points** if ANY train/test/arc-gen example is wrong.
- A task is worth `max(1.0, 25 - log(...))` points if ALL examples are correct.
- Maximum ~25 points per task × 400 tasks = ~10,000 points theoretical max.
- **File size limit:** 1.44 MB per `.onnx` file (one floppy disk — intentional golf theme).

**Forbidden ONNX ops:** `LOOP`, `SCAN`, `NONZERO`, `UNIQUE`, `SCRIPT`, `FUNCTION`

---

## Data Format

### Task Files
- 400 tasks: `task001.json` … `task400.json`
- Each task has three splits: `"train"` (5–10 pairs), `"test"` (1 pair), `"arc-gen"` (many pairs)
- The model must pass **all three splits** with zero errors.

### Grid Encoding
- Grids are 2D arrays of integers 0–9 (colors), max 30×30.
- Encoded as one-hot tensors: shape `(1, 10, 30, 30)` float32.
  - `tensor[0][color][row][col] = 1.0` where that cell has that color.
  - Unused cells remain 0.0.

### ONNX I/O Contract
```
input  tensor: name="input",  shape=(1, 10, 30, 30), dtype=float32
output tensor: name="output", shape=(1, 10, 30, 30), dtype=float32
```
Prediction rule: a cell gets color `c` if `output[0][c][row][col] > 0.0`.

---

## Directory Structure

```
neurogolf-2026/
├── CLAUDE.md                    # This file
├── requirements.txt             # Python dependencies for local dev
├── task001.json … task400.json  # Competition data (already downloaded)
├── neurogolf_utils/
│   └── neurogolf_utils.py       # Official scoring + verification utilities
│
├── src/
│   ├── model.py                 # TinyCNN model definition (PyTorch)
│   ├── get_dataset.py           # ARCDataset loader (grid JSON → tensors)
│   ├── train.py                 # Training script (one model per task)
│   ├── export.py                # PyTorch checkpoint → ONNX export
│   └── verify.py                # Local verification against neurogolf_utils
│
├── notebooks/
│   └── train_kaggle.ipynb       # Self-contained Kaggle GPU training notebook
│
└── output/
    ├── checkpoints/             # Saved PyTorch weights: taskXXX.pt
    └── onnx/                    # Exported ONNX models: taskXXX.onnx
```

---

## Workflow

### Step 1 — Train on Kaggle GPU

All training must run on Kaggle GPU (P100 or T4 free tier).
**Do NOT train locally** — use the Kaggle notebook.

1. Go to https://www.kaggle.com/competitions/neurogolf-2026/code
2. Create a new notebook → enable GPU accelerator
3. Attach the competition data: `neurogolf-2026`
4. Upload `notebooks/train_kaggle.ipynb` or copy its contents
5. Run all cells — outputs saved to `output/onnx/`

Data path on Kaggle: `/kaggle/input/neurogolf-2026/`

### Step 2 — Download ONNX Files

After training, download from the Kaggle notebook output:
```
output/onnx/task001.onnx … task400.onnx
```
Copy them to `output/onnx/` locally.

### Step 3 — Verify Locally (Optional)

```bash
python src/verify.py --task 1
python src/verify.py --all
```

### Step 4 — Package and Submit

```bash
cd output/onnx && zip ../../submit/submission.zip *.onnx
```
Submit `submit/submission.zip` at:
https://www.kaggle.com/competitions/neurogolf-2026/submit

---

## Model Architecture (Baseline v1)

`TinyCNN` — 3-layer CNN, ~4K parameters, ~16 KB on disk.

```
Input (1, 10, 30, 30)
  → Conv2d(10, 16, 3, padding=1) → ReLU
  → Conv2d(16, 16, 3, padding=1) → ReLU
  → Conv2d(16, 10, 1)
Output (1, 10, 30, 30)
```

**Why this shape:**
- 10 input channels = one-hot color encoding
- 10 output channels = logits per color per cell
- padding=1 preserves the 30×30 spatial size
- Final 1×1 conv collapses hidden channels to colors cheaply

---

## Training Details

- **Loss:** `BCEWithLogitsLoss` — each channel independently predicts active/inactive
- **Optimizer:** Adam, lr=0.01
- **Schedule:** ReduceLROnPlateau (patience=50)
- **Epochs:** up to 2000 per task, early stop when loss < 1e-6
- **One model per task** — each task gets its own checkpoint and ONNX file

**Data augmentation for training:** Use all of `train + arc-gen` split.
The `test` split is held out for validation only.

---

## Key Technical Constraints

| Constraint | Value |
|------------|-------|
| ONNX opset | 17 (PyTorch default, supported by onnxruntime) |
| ONNX IR version | auto |
| Max file size | 1,509,949 bytes (1.44 MB) |
| Input tensor name | `"input"` |
| Output tensor name | `"output"` |
| Input/output shape | `(1, 10, 30, 30)` |
| Input/output dtype | float32 |
| Forbidden ops | LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION |

---

## Iteration Ideas (Post-Baseline)

1. **More capacity:** increase `hidden` channels if score is 0 due to wrong answers
2. **Smaller model:** reduce `hidden` to 8 or use 1×1 convs only for higher score
3. **Task-specific tuning:** some tasks may need deeper/wider networks
4. **Quantization:** INT8 weights (4× size reduction) via `torch.quantization`
5. **Architecture search:** try different kernel sizes per task
6. **ONNX slimming:** `onnxslim` / `onnx-simplifier` to reduce graph overhead

---

## Leaderboard Context (as of 2026-04-21)

| Rank | Team | Score |
|------|------|-------|
| 1 | Kaggle Agent | 6907.86 |
| 2 | keymoon | 6857.79 |
| 3 | chicm | 6272.96 |
| Top ~20 | various | 5000–6900 |

Competition launched 2026-04-15. Still very early.
