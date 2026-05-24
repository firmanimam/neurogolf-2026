# NeuroGolf 2026 — Top-Score Strategy Analysis

Analysis of the highest-voted public kernels (Code tab) on Kaggle, distilling what the top public scorers (~5,700–6,225) are actually doing — and what the leader at ~6,907 is almost certainly doing too. The Discussion tab is JS-rendered and the available scrapers only return page titles, so this analysis is built from the **notebook source code itself** (pulled via `kaggle kernels pull`), where the methodology is documented directly.

Current state of our submission: **505.41** with a trained TinyCNN. The gap to the top is not a tuning problem — it's a paradigm problem.

---

## TL;DR — The Winning Paradigm

**Nobody at the top is training neural networks.** The competition rewards the smallest correct ONNX graph, and a hand-built ONNX graph encoding the exact ARC rule has 10–100× lower cost than any trained CNN. The top public pipeline is:

1. **Reverse-engineer each task's rule** from `train + test + arc-gen` examples (the rule is deterministic per task).
2. **Compile that rule into a static ONNX graph** using primitive ops (`Slice`, `Pad`, `Where`, `ScatterND`, `CumSum`, `ReduceMax`, `Conv` as a propagation kernel, etc.) — no learned weights.
3. **Per-task select the cheapest valid ONNX** across many public bundles + your own hand-builds.
4. **A/B test via "block-LB drilling"** — submission-based binary search to find which source is best per task.

The leaderboard is a pure optimization problem on top of a public artifact pool. Training is the wrong tool.

---

## Score Landscape (as of 2026-05-20)

| Rank | Approach | Score |
|---|---|---|
| #1 leader (Kaggle Agent, private) | Unknown, presumed full hand-build + LLM | 6907.86 |
| #2 keymoon | Unknown | 6857.79 |
| Top public artifact (afr1ste 6225 v.) | Hand-built static ONNX compilers | ~6225 |
| Per-Task Hand-Built (octaviograu) | 6029 base + 3 LLM-derived hand-builds | **6042.85** |
| Constraint Smart Logic Mix (jonathanchan) | ZIP blend → arc-nano → LLM rescue | ~5800–6000 |
| Block-LB Drilling (octaviograu) | Per-task swap between 2 public bundles | **5740.30** |
| Open-Solution (afr1ste) | Pure compiled-ONNX per task | **5689.51** |
| Our TinyCNN baseline | Trained 3-layer CNN, 400 task-specific models | **505.41** |

A trained TinyCNN scoring 505 means we got ~51 tasks correct (each correct task with our ~4K-param net scores ~9.92 pts). Top kernels are getting ~350+ tasks correct **and** doing it with order-of-magnitude smaller cost per task.

---

## The Five Key Techniques

### 1. Hand-Built Static ONNX Compilers (afr1ste, octaviograu)

This is the core technique. From the `5689.51 Current Rules Open Solution` notebook:

> *"Most of the new positive tasks are small fixed-window ARC rules. The implementation style is intentionally simple:*
> 1. *Slice only the active task window, usually 5×5, 6×6, 7×7, 9×9, 10×10, or 11×9.*
> 2. *Work in boolean masks or scalar color IDs instead of full float one-hot tensors.*
> 3. *Use `Slice`, `Pad`, `Or`, `And`, `Not`, `Where`, `CumSum`, `ReduceMax`, `Concat`.*
> 4. *Pad the final small result back to the required 30×30 static shape.*"

Concrete cost wins shown in their notebook:
- `task181` (marker-selected mirror copy): cost `54,784 → 1,194`, score `+3.83 LB`
- Each successful hand-build moves cost down by 10–100× vs. the trained-CNN baseline

The `[6042.85] Per-Task Hand-Built ONNX Solvers` notebook adds three more, all verified `266/266` on train+test+arc-gen before submission:
- `task277`: 8-connected components, recolor the unique-size component → uses label propagation + ScatterND histogram
- `task330`: connected-component recolor by cell count → uses `opset 17 ScatterND(reduction='add')` for O(cells) histogram
- `task364`: topology-based recolor by counting endpoints + turns

**Banned ops** they explicitly route around: `Loop`, `Scan`, `NonZero`, `Unique`, `Compress`, `Script`, `Function`. Label propagation that would naturally use `Loop` is unrolled into a fixed number of `Conv` or `Where` iterations.

### 2. The Local Cost Oracle (octaviograu)

> *"The grader scores each task ONNX with `cost = memory + params` and converts to points via `max(1, 25 - ln(max(1, cost)))`. Both terms are computable locally."*

- **`params`** = element count of every `initializer` + every `Constant` node value (incl. `value_floats`, `value_ints`)
- **`memory`** = sum over intermediate tensors of `max(static_inferred_size, ORT_runtime_size)`, from a single ORT profile trace
- **`macs`** estimated similarly per-op

Once your local cost matches the grader to the hundredth, you A/B locally and only submit known wins. Octavi's notebook shows their LB predictions match realized LB within **±0.02**, confirming the per-task additivity assumption of the public LB.

### 3. Per-Task Selection / Ensemble Blending (svanikkolli `arc-nano-engine`, jonathanchan)

The pattern, verbatim from `arc-nano-engine`:

```python
NOTEBOOK_SOURCES = [
    ('NGC_Mix',      ...ngc26-constraint-smart-logic-mix-blending/submission.zip),
    ('Konbu_341',    ...neurogolf-2026-blended-341-tasks-lb-4215/submission.zip),
    ('Magma_4200',   ...4200-v5-neurogolf-fix-for-new-system-soon/submission.zip),
    ('Afr1ste_6225', ...neurogolf-6225-51-public-score-open-solution/submission.zip),
]
ENSEMBLE_SOURCES = [...several CC0 datasets of pre-built ONNX bundles...]

for task_id in range(1, 401):
    candidates = []
    for src in all_sources:
        raw = all_sources[src].get(task_id)
        if raw is None: continue
        r = validate_official(task_id, raw)  # ORT inference vs. all 266 examples + local cost
        if r: candidates.append((r['cost'], r['score'], raw, src))
    if candidates:
        candidates.sort(key=lambda x: (x[0], -x[1]))   # cheapest valid wins
        solved[task_id] = candidates[0][2]
```

Key implementation details:
- Validate with `nu.verify_subset(sess, examples['train']+examples['test'])` AND `examples['arc-gen']` — must be zero failures
- Score with `nu.score_network(path)` → `(macs, mem, params)` → `cost = sum`
- Pick lowest cost. Ties broken by highest score (same thing). All bundles assumed CC0; this is the explicit norm on this competition.

This is the cheapest entry point into the top: you don't have to author anything, just smartly aggregate every public artifact.

### 4. Block-LB Drilling (octaviograu, 5740.30)

A submission-based binary search exploiting the fact that the public LB is exactly additive over per-task scores. Given two bundles A (5689) and B (5571):

- **Round 1**: Swap blocks of 40 tasks. Submit each variant. Delta per block reveals where B beats A.
- **Round 2**: Recurse into positive blocks at 10-task granularity.
- **Round 3**: Recurse into positive sub-blocks at 2-task granularity.

Result: from afr1ste's 5689.51 bundle, swapping in konbu17's versions for just **4 tasks** lifted LB to 5740.30. Predictions matched realized LB within **±0.02**, proving the per-task-additive grader.

Worth knowing because: you can drill at the granularity of any "additive" subset, you don't need source access to either bundle, and each submission costs only one slot. Octavi reached 5740 in ~10 submissions.

### 5. LLM-Assisted Rule Derivation (jonathanchan, octaviograu)

Both top notebooks describe a "Manual LLM Rescue" loop for unsolved tasks:

> *"1. Run the full pipeline and inspect unsolved tasks in CELL 8 / 2. For a target task, execute: `show_task_json(N)`, `show_py_solution(N)` / 3. Send the extracted JSON + python skeleton to an LLM (Claude/GPT) / 4. Take the returned rule, validate 266/266 locally, then translate to static ONNX."*

The LLM is used as a **rule deriver**, not a model — it reads the input/output grid pairs and outputs a Python function. You verify that function on all 266 examples, then hand-translate to ONNX primitives.

---

## Task Taxonomy (karnakbaevarthur — "Logic & Complexity Map")

Tasks are categorized by logic family before being solved. The author publishes a dataset (`neurogolf-2026-task-transformation-library`) mapping every one of the 400 tasks to:
- transformation type (color map, symmetry, fill, count, connected-component, etc.)
- difficulty score
- grid flags (size, palette, monochrome/multicolor)

This is the prioritization layer for the hand-build pipeline. You pick tasks where:
- the rule is simple enough that a human (or LLM) can derive it from a handful of examples;
- the rule maps cleanly to allowed ONNX primitives;
- a trained CNN can't beat the hand-build on cost.

---

## What Public Notebooks Explicitly Discourage

From afr1ste's open-solution writeup:

> *"Direct grafting of old dynamic-shape or constant-output tricks mostly stopped working after the rule refresh. A public model is a hint, not a submission. First infer the task rule from train+test+arc-gen, then write a strict static ONNX compiler."*

The evaluator was tightened mid-competition. Tactics that died:
- **Dynamic output shapes** — outputs must be static `(1, 10, 30, 30)`.
- **Constant-output banks** — embedding the visible example outputs and returning them by index.
- **Old parser tolerance** — looser ONNX IR/opset combos that the new grader rejects.
- Anything relying on hidden-test similarity to visible examples.

Take public ONNX as a hint, re-derive the rule, re-emit a clean static graph.

---

## Where Our Current Approach Sits

Our TinyCNN baseline (`src/model.py`) trains one ~4K-param 3-layer Conv net per task. Per the local cost formula:

```
cost ≈ macs (3.5M) + memory (~16KB) + params (~4K) ≈ 3.53M
score_per_correct = max(1, 25 - ln(3.53M)) ≈ 9.92
505.41 / 9.92 ≈ 51 tasks correct out of 400 (~13%)
```

Two compounding problems:
1. **Accuracy ceiling**: TinyCNN can't fit ~87% of the tasks (counting, symmetry over distance, recursion, etc.) even with arbitrary training time.
2. **Cost floor**: Even on tasks it does solve, the 3.5M-MAC cost caps each correct task at 9.92 pts. A hand-built graph for the same task can hit 20+ pts (cost ≤ 150).

To get to 5,000+ we need both: solve more tasks AND make each correct solution radically smaller. Training alone can't do either.

---

## Recommended Path Forward

Listed in order of effort × expected gain:

1. **Adopt the public-bundle ensemble pattern** (~1 day, instant +4,000+ LB):
   - Download the top 5–10 public submission.zip bundles as Kaggle datasets in our training notebook.
   - For each of the 400 tasks, validate every candidate ONNX against `train + test + arc-gen` using `neurogolf_utils.verify_subset` and `score_network`.
   - Emit the lowest-cost valid ONNX. All these bundles are CC0; this is the explicit norm on this competition.
   - This alone should land us in the 5,500–6,200 range.

2. **Set up the local cost oracle** (~½ day):
   - Wrap `neurogolf_utils.score_network` and ORT profiling so we can compute the exact grader cost locally.
   - Required to A/B any future change without burning submission slots.

3. **Block-LB drill on top of the ensemble** (~3 submissions):
   - With ~10 bundles in our pool, drill 40 → 10 → 2 to find per-task winners.

4. **Hand-build a few easy tasks** (open-ended, +10–50 LB each):
   - Use the task-complexity map to pick the easiest unsolved tasks.
   - Derive rule with LLM, validate 266/266 locally, emit static ONNX.

5. **Stop training the TinyCNN.** It contributes nothing above the public ensemble floor. Keep `src/model.py` only as a fallback for tasks where no bundle is correct (rare).

---

## Sources (high-voted public kernels analyzed)

| Score | Author / Slug | Approach |
|---|---|---|
| 6042.85 | `octaviograu/6042-85-per-task-hand-built-onnx-solvers` | Hand-built ONNX + local oracle methodology |
| 5800.55 | `kojimar/5800-55-lb-neurogolf-task-level-onnx-blend` | Per-task overrides table, SHA10-pinned |
| 5740.30 | `octaviograu/neurogolf-2026-block-lb-drilling-5740-30` | Block-LB binary search, per-task swap |
| 5689.51 | `afr1ste/neurogolf-5689-51-current-rules-open-solution` | Pure static-ONNX compilers, per-task |
| ~5500 | `svanikkolli/arc-nano-engine` | Cross-source ensemble (NOTEBOOK_SOURCES + ENSEMBLE_SOURCES) |
| ~5000 | `jonathanchan/ngc26-constraint-smart-logic-mix-blending` | Three-phase: blend → arc-nano → LLM rescue |
| n/a | `karnakbaevarthur/neurogolf-all-task-logic-complexity-map` | Task taxonomy and difficulty dataset |
| n/a | `yash9439/neurogolf-2026-improved-starter-notebook` | Clean reference scaffold with `score_network` validation |
| n/a | `hanifnoerrofiq/neurogolf-sparse-builder` | Sparse-tensor ONNX builders |
| n/a | `aliafzal9323/neurogolf-2026-tiny-onnx-solver` | Detector + learned-conv fallback hybrid |

Discussion tab content was not retrievable via WebFetch (JS-rendered, scraper returns title only). Strategy details above are extracted from the source code and markdown of the kernels themselves, which is more authoritative than the discussion summaries anyway.
