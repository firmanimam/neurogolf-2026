# NeuroGolf 2026

A Kaggle competition entry for [NeuroGolf 2026](https://www.kaggle.com/competitions/neurogolf-2026), organized by the Neurosynthetic Research Institute at IJCAI-ECAI 2026.

---

## What Is This?

NeuroGolf is a **model size minimization** competition built on top of ARC-AGI image transformation tasks. The challenge has two simultaneous goals:

1. **Correctness** — the model must produce the right output grid for every example in a task (train, test, and arc-gen splits).
2. **Smallness** — among correct models, smaller = higher score.

Each task gets its own dedicated `.onnx` file (one model per task, 400 tasks total).

### Scoring

```python
points = max(1.0, 25.0 - math.log(macs + memory_bytes + params))
```

- A task scores **0** if any example is wrong.
- A task scores up to **~25 points** based on model size (fewer MACs, bytes, and params = more points).
- Theoretical maximum: ~10,000 points across all 400 tasks.
- Hard file size limit: **1.44 MB** per `.onnx` file.

---

## Data Format

Tasks are 2D color grids (integers 0–9, max 30×30), encoded as one-hot float32 tensors:

```
shape: (1, 10, 30, 30)
tensor[0][color][row][col] = 1.0
```

Every ONNX model must accept and produce tensors of this exact shape with input name `"input"` and output name `"output"`. A cell is predicted as color `c` if `output[0][c][row][col] > 0.0`.

---

## Baseline Model — TinyCNN

A 3-layer CNN with ~4K parameters and ~16 KB on disk:

```
Input (1, 10, 30, 30)
  → Conv2d(10, 16, 3, padding=1) → ReLU
  → Conv2d(16, 16, 3, padding=1) → ReLU
  → Conv2d(16, 10, 1)
Output (1, 10, 30, 30)
```

- `padding=1` preserves the 30×30 spatial size through every layer.
- The final 1×1 conv maps hidden channels back to 10 color logits cheaply.
- One model is trained independently per task.

---

## Project Structure

```
neurogolf-2026/
├── requirements.txt             # Python dependencies
├── neurogolf_utils/
│   └── neurogolf_utils.py       # Official scoring + verification utilities
├── src/
│   ├── model.py                 # TinyCNN architecture
│   ├── get_dataset.py           # ARCDataset: JSON → one-hot tensors
│   ├── train.py                 # Per-task training script
│   ├── export.py                # Export PyTorch checkpoint → ONNX
│   └── verify.py                # Local verification via neurogolf_utils
├── notebooks/
│   └── train_kaggle.ipynb       # Self-contained Kaggle GPU training notebook
└── output/
    ├── checkpoints/             # taskXXX.pt — saved PyTorch weights
    └── onnx/                    # taskXXX.onnx — exported ONNX models
```

---

## Workflow

### 1. Train on Kaggle GPU

Training runs on Kaggle free-tier GPU (P100/T4).

1. Go to the [competition code tab](https://www.kaggle.com/competitions/neurogolf-2026/code) and create a new notebook.
2. Enable GPU accelerator and attach the `neurogolf-2026` dataset.
3. Upload `notebooks/train_kaggle.ipynb` and run all cells.
4. Models are saved to `output/onnx/`.

Data path on Kaggle: `/kaggle/input/neurogolf-2026/`

### 2. Download ONNX Files

Download the generated `taskXXX.onnx` files from the notebook output and place them in `output/onnx/` locally.

### 3. Verify Locally

```bash
python src/verify.py --task 1   # verify a single task
python src/verify.py --all      # verify all tasks
```

### 4. Package and Submit

```bash
cd output/onnx && zip ../../submit/submission.zip *.onnx
```

Submit `submit/submission.zip` at the [competition submission page](https://www.kaggle.com/competitions/neurogolf-2026/submit).

---

## Training Details

| Setting         | Value                           |
| --------------- | ------------------------------- |
| Loss            | BCEWithLogitsLoss               |
| Optimizer       | Adam, lr=0.01                   |
| LR schedule     | ReduceLROnPlateau (patience=50) |
| Max epochs      | 2000 per task                   |
| Early stop      | loss < 1e-6                     |
| Training data   | `train` + `arc-gen` splits      |
| Validation data | `test` split (held out)         |

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 2.0+.

---

## ONNX Constraints

| Constraint    | Value                                         |
| ------------- | --------------------------------------------- |
| Opset         | 17                                            |
| Max file size | 1,509,949 bytes (1.44 MB)                     |
| Input name    | `"input"`                                     |
| Output name   | `"output"`                                    |
| Shape         | `(1, 10, 30, 30)`                             |
| Dtype         | float32                                       |
| Forbidden ops | LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION |

## Ideas for Improving Score

- **More capacity:** increase `hidden` channels if a task's accuracy is 0.
- **Smaller model:** reduce `hidden` to 8 or use only 1×1 convs for higher points.
- **Quantization:** INT8 weights via `torch.quantization` for ~4× size reduction.
- **Architecture search:** try different kernel sizes per task type.
