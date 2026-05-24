# NeuroGolf 2026 — Handover Strategy & Progress Log

**Updated: 2026-05-23, mid-session (paused by user).**

## Current State

- **Best LB: 6,044.76** (v11 — unchanged through this session) — rank 164 / 1,319 teams
- **Bronze threshold: 6,062.21** (moved up since last session)
- **Gap to bronze: +17.45 LB**
- **Deadline: 2026-07-15** (~7.5 weeks remaining)
- **Kaggle notebook (live):** https://www.kaggle.com/code/piririp/neurogolf-2026-ensemble-v1

## Session Summary (2026-05-23)

Goal: reach ~6063 LB. **Outcome: no progress; all new submissions regressed.** v11 is still the best.

Key findings from 4 new submissions (v12-v15) + 8 new ONNX builds:

1. **Every positional/spatial handbuild failed on the grader**, even ones with rules verified 265/265 locally and using "safe" op sets:
   - v12 (hb261, shift+recolor 5×5): **-5.89 LB**
   - v13 (hb095, 3×3 dilation 9×9): **-4.60 LB**
   - v14 (hb317, super-cell 9×9 static-mask): **-18.20 LB**
   - v15 (hb282, 3×3 ring stamp 9×9, conv-based, fully safe ops): **-3.61 LB**

2. **The "ConstantOfShape + ReduceMax-attribute" hypothesis was wrong.** task282 avoided both and STILL failed. The actual common factor: **all four hardcoded the example grid size (9×9 or 5×5) into the graph**. Strongest remaining hypothesis: the grader's hidden test set contains examples with grid sizes that don't appear in our train+test+arc-gen splits, so any solver that assumes a fixed shape silently produces wrong output for those cells.

3. **The ONLY confirmed-working handbuild remains task389** (+1.91 LB). It is a pure channel permutation: `output = Gather(input, perm, axis=1)` where `perm` is computed from a channel-level ArgMax. **No positional/spatial logic.** Works for any grid size up to 30×30 because the per-channel data flows untouched.

4. **Trivial constant permutations cannot beat the 6042 bundle.** Built and validated handbuilds for task016, task276, task309, task337 — all are single-Gather solutions at cost 36,090 (theoretical floor: 36KB output memory + 10-element index). Each ties the bundle exactly (+0.00 LB). The 6042 bundle is already at the floor for these.

5. **task129 (most-frequent-fill) LOSES vs bundle**: ours 66,863 vs bundle 37,170. Bundle likely uses a single Conv1×1 with crafted weights. **task298 rejected** (ring rotation requires positional logic, not pure permutation). task267 and task229 builders were stopped mid-run; their bundle baselines (11,606 and similar) are SUBSTANTIALLY below the single-Gather floor of 36,090 — meaning the bundle has clever sub-floor tricks we don't understand. **Unlikely to beat them with our current toolkit.**

6. **task004 builder produced an ONNX with dynamic grid handling** (cumsum-based) and predicted +0.79 LB. It was NOT submitted in this session. It's the only positional rule we built that uses fully dynamic shape derivation, so it's the best remaining candidate to test the "hardcoded shape = grader fail" hypothesis.

## Quick Resume — What To Do Next Session

1. **Submit task004 in isolation to test the "hardcoded shape" hypothesis cleanly.** Build a 400-file zip where only `task004.onnx` is our handbuild and the other 399 are the 6042 bundle. If it gains, the hypothesis is confirmed and we should aggressively rebuild 095/261/282/317 with dynamic shape derivation. If it loses, the grader has even more subtle rejection criteria and we should abandon positional handbuilds entirely.
   ```bash
   mkdir -p /tmp/v16 && \
   unzip -q -o /tmp/bundle-tests/best-score-the-2026-neurogolf-championship/submission.zip -d /tmp/v16/onnx_files && \
   cp output/handbuild_onnx/task389.onnx output/handbuild_onnx/task004.onnx /tmp/v16/onnx_files/ && \
   cd /tmp/v16/onnx_files && zip -q -r /tmp/v16/submission.zip *.onnx && cd - && \
   kaggle competitions submit -c neurogolf-2026 -f /tmp/v16/submission.zip -m "v16: 6042 + hb389 + hb004 (dynamic-shape handbuild)"
   ```

2. **Compute per-task 6042-bundle costs** (oracle estimate) and rank tasks by ratio of `bundle_cost / theoretical_floor`. Tasks where the bundle is far from optimal are our handbuild targets — not tasks that simply "look easy". `task389` worked because the bundle's task389 was 137K (3× our 45K). Most other "easy" permutation tasks are already at the floor. There should be a small subset where the bundle is wasteful AND the rule is channel-permutation-only. That's the gold seam.

3. **Re-attempt task267 and task229 with the Einsum trick** noted by the killed task229 builder: `Einsum("nchw,c->nhw", input, mask)` collapses an Mul-then-ReduceSum into one op with no (1,10,30,30) intermediate. May get below 36K floor that's tripping us up. But only attempt if step 2 shows their bundle baselines have actual room to beat.

## Submission Log Update (this session)

| ver | LB | Change | Lesson |
|----|----:|---|---|
| v12 | 6,038.87 | +hb261 | first grader failure of locally-valid handbuild |
| v13 | 6,040.16 | swap hb261→hb095 | second failure, also 9×9 hardcoded |
| v14 | 6,026.56 | swap hb095→hb317 | third failure; "safe ops" hypothesis disproved |
| v15 | 6,041.15 | swap hb317→hb282 | fourth failure; conv-based 9×9 also fails |

## Hand-Built ONNX Inventory Update

| Task | Cost | Pts | vs Bundle | Status | Grid logic |
|------|----:|----:|----:|---|---|
| task389 | 45,460 | 14.28 | +1.91 realized | ✅ confirmed on LB | pure channel permutation |
| task261 | 118,910 | 13.31 | -5.89 realized | ❌ failed on grader | 5×5 hardcoded |
| task095 | 150,489 | 13.08 | -4.60 realized | ❌ failed on grader | 9×9 hardcoded |
| task317 | 4,298 | 13.98 | -18.20 realized | ❌ failed on grader | 9×9 hardcoded (static in_grid mask) |
| task282 | 1,216 | 13.52 | -3.61 realized | ❌ failed on grader | 9×9 Slice+Pad |
| task004 | 457,277 | 11.97 | +0.79 predicted | ⏳ built, not submitted | **dynamic** (cumsum-based) |
| task276 | 36,090 | 14.51 | +0.00 (tie) | ⛔ don't ship — bundle already at floor | pure permutation |
| task309 | 36,090 | 14.51 | +0.00 (tie) | ⛔ don't ship | pure permutation |
| task337 | 36,090 | 14.51 | +0.00 (tie) | ⛔ don't ship | pure permutation |
| task016 | 36,090 | 14.51 | +0.00 (tie) | ⛔ don't ship | pure permutation |
| task129 | 66,863 | 13.89 | -0.59 predicted | ⛔ don't ship — we lose | runtime permutation |
| task267 | n/a | n/a | bundle is 11,606 (below floor!) | ⏸ builder killed mid-iteration | runtime permutation |
| task229 | n/a | n/a | bundle is similarly low | ⏸ builder killed mid-iteration | runtime permutation |

## Strategy Notes Going Forward

- **Stop chasing positional handbuilds blindly.** Until we either confirm (via task004) or refute the "dynamic shape = grader pass" hypothesis, every positional submission is a coin flip costing real LB.
- **6042 bundle is highly optimized for trivial cases.** Don't build a handbuild for any task without first inspecting the bundle's per-task cost — if it's already near the 36K floor, no gain is possible.
- **The gold seam is tasks where bundle cost > 100K AND rule is pure channel permutation.** That was task389. Find more like it.

---

## The Big Strategic Reversal (post-v9)

The original `strategy.md` plan was "ensemble + pick cheapest valid ONNX per task using our cost oracle". **This was wrong.** Our oracle has *source-dependent bias*:

| Bundle | Predicted (our oracle) | Realized (LB) | Delta | Ratio |
|---|---:|---:|---:|---:|
| 6042 hand-built | 5,081 | 6,042 | **+961** | 13× cheaper |
| 5689 open-solution | 5,302 | 5,689 | +387 | ~3× cheaper |
| ngc26 | 5,357 | 5,800 | +443 | ~3× cheaper |
| 5800 task-level | 5,290 | 5,800 | +510 | ~3.5× cheaper |

The 6042 bundle's ONNX scores ~13× cheaper on the grader than our `onnx_tool` predicts. When we "picked cheapest by oracle", we systematically skipped 6042's ONNX in favor of others that *looked* cheaper but were actually worse on the grader. We ended up worse than just using 6042 alone (5,678 vs 6,042).

**Corrected strategy: trust the bundle's known LB score, not our oracle.** Use 6042's ONNX for all 400 tasks unconditionally → guarantees ~6,042 floor. Layer hand-builds only when they have been *verified to win on the grader* (not just locally).

---

## Submission Log (chronological)

| ver | LB | What changed | Lesson |
|----|----:|---|---|
| TinyCNN baseline | 505.41 | Train ~4K-param CNN per task | Trained NNs cannot compete with hand-built ONNX |
| v1 ensemble | 5,678.23 | Cheapest-valid pick from 6 public bundles | Oracle-based picking has systematic bias |
| v8 + hb389 | 5,680.14 | Added handbuild389 to v1 ensemble | Handbuilds verified at 266/266 *can* gain on LB (+1.91) |
| v9 priority | 5,922.25 | 6042 preferred + ngc26 fallback for 21 tasks where our validator rejected 6042's ONNX | Bundles ranked by known LB beats oracle |
| v10 trust-6042 | **6,042.85** | All 400 ONNX from 6042 (skip local validation) | Local validator was over-rejecting; matches 6042 bundle solo |
| v11 trust + hb389 | **6,044.76** | v10 + handbuild389 | +1.91 LB matches v8→v1 delta exactly |
| v12 + hb261 | 6,038.87 | v11 + handbuild261 (also 265/265 local) | **Hand-build can FAIL on grader despite local pass** — net -5.89 LB |

### Bundle tests (other public submissions tried solo)

| Bundle | LB |
|---|---:|
| best-score (mirzayasirabdullah07) | 6,042.85 *(clone of 6042)* |
| 6029-09-lb (jsrdcht) | 6,029.09 |
| ngc26 (jonathanchan, in v1 too) | ~5,800 |
| 5800-55 (kojimar, in v1 too) | ~5,800 |
| 5740-30 drilling (octaviograu, in v1 too) | 5,740.30 |
| multi-source (vyankteshdwivedi) | 5,740.30 |
| may-8-updated (konbu17) | 5,571.69 |
| 4808 (thisray) | 4,866.75 |
| ldausl championship | 2,776.87 |
| kaggle-agent (jiweiliu), cross-source (karnakbaevarthur), emsembling (yash9439) | ERROR (submission invalid) |

**No public bundle scores above 6,042.** The path to higher must be hand-builds.

---

## Hand-Build Workflow (proven on task389)

**Per-task process** (~30–60 min each):

1. `python3 src/view_task.py <tid>` — render input/output grid pairs as ASCII
2. Derive a Python `solve(grid)` rule
3. Save as `src/handbuild/task<NNN>.py` and run it: must report **N/N pass**
4. Build the ONNX in `src/handbuild/build_task<NNN>.py` using ONLY these primitives (avoid the forbidden ops `LOOP`/`SCAN`/`NONZERO`/`UNIQUE`/`SCRIPT`/`FUNCTION`):
   - Shape-free: `Slice`, `Pad`, `Concat`, `Transpose`, `Reshape`, `Identity`, `Constant`, `ConstantOfShape`
   - Boolean/comparison: `Where`, `Equal`, `Greater`, `Less`, `And`, `Or`, `Not`
   - Arithmetic: `Add`, `Sub`, `Mul`, `Div`, `Clip`, `Min`, `Max`
   - Reductions: `ReduceSum` (axes as INPUT in opset 13+), `ReduceMax` (axes as ATTRIBUTE in opset 17 — **gotcha**, see Notes), `ReduceMean`
   - Movement: `Gather`, `ScatterND` (with `reduction='add'` for histograms via opset 17), `Tile`, `Expand`
   - Conv-like: `Conv`, `MaxPool`, `AvgPool`, `Resize` (for upsampling via nearest)
5. `python3 src/handbuild/test_onnx.py <tid>` — validates against all examples AND measures cost
6. If cost beats current winner: add to `HANDBUILD = {...}` dict in the notebook's Phase 4 cell
7. **NEW caveat (post-v12):** even at 265/265 local, the grader may still reject. Test EACH new handbuild in isolation (or in a small batch) before stacking.

### Key cost insights

- **Output tensor (1,10,30,30)** = 36KB memory minimum — unavoidable
- **ReduceSum/ReduceMax over input** = 9K MACs — needed if rule depends on cell-counting or "in-grid" mask
- **Floor for "permutation" hand-builds** ≈ 45K cost (task389 example) → 14.3 pts each
- **For lower cost**: avoid `ReduceSum`/`ReduceMax` entirely. Possible when:
  - The rule uses only fixed grid positions (no per-cell color counting)
  - The "in-grid" mask can be derived from a single channel slice (when all examples use one specific marker color)

### Hand-built ONNX inventory (in `output/handbuild_onnx/`)

| Task | Cost | Pts | vs Current | Status | File |
|------|----:|----:|----:|---|---|
| task389 | 45,460 | 14.28 | +1.11 pred / **+1.91 realized** | ✅ confirmed on LB (v8 & v11) | `task389.onnx` |
| task261 | 118,910 | 13.31 | +0.60 pred / **-5.89 realized** | ❌ **failed on grader** in v12 | `task261.onnx` |
| task095 | 150,489 | 13.08 | +1.74 pred | ⏳ built, not yet submitted | `task095.onnx` |

### Hand-build targets (sorted by potential gain)

Run `python3 src/prioritize_handbuild.py` for the full table. Top easy ones:

| Task | Current cost | Pts | Gain | Grid | Colors | Pattern |
|------|----:|----:|----:|----:|----:|---|
| task389 | 137,416 | 13.17 | +1.1 | 5×5 | 3 | **done** |
| task261 | 217,000 | 12.71 | +0.6 | 5×5 | 3 | shift+recolor 8→2 (built, failed) |
| task095 | 855,000 | 11.34 | +1.7 | 9×9 | 3 | 3×3 dilation around 5s + keep center |
| task317 | 850,500 | 11.35 | +1.7 | 9×9 | 3 | Per 3×3 super-cell containing a 5: fill with 1s |
| task282 | 846,000 | 11.35 | +1.6 | 9×9 | 3 | 3×3 "ring" pattern: corners=5, edges=1, center=0 |
| task366 | 20.9M | 8.14 | +6.3 | 30×30 | 4 | Too complex, skip |
| task344 | 2.3M | 10.35 | +3.9 | 9×9 | 5 | Unknown |
| task019 | 1.1M | 11.07 | +3.5 | 12×12 | 3 | Unknown |
| task004 | 1.0M | 11.18 | +3.4 | 14×14 | 2 | Unknown — only 2 colors, might be simple |
| task389/.. | various | various | each +1–2 | various | various | similar marker-swap patterns |

**To hit bronze (+10 LB more), need 5–7 more confirmed hand-builds.**

---

## Open Problem: task261 Failed on Grader

`handbuild261` passes 265/265 locally but lost 5.89 LB when added to v11. Two hypotheses:

1. **Grader has hidden test examples** beyond the 265 in train+test+arc-gen.
2. **Grader's ONNX runtime treats our ops differently** — possibly the combo `ReduceMax(axes attribute) + Pad + ConstantOfShape + ScatterND`. task389 uses different ops (`ReduceSum(axes input) + Gather`) and works fine.

To diagnose, next session should:
- **Isolation test**: submit a zip that contains ONLY task261.onnx (others as identity). If it scores 0, the grader rejects our ONNX entirely.
- **Alternative ONNX**: rewrite task261 using ReduceSum + Gather (task389 style) instead of ReduceMax + ConstantOfShape + ScatterND.
- **Check opset**: try ir_version=10 and opset=10 (matches `neurogolf_utils.py`'s defaults) instead of opset 17.

---

## Repository State (key files)

```
neurogolf-2026/
├── strategy.md                          # THIS FILE (handover)
├── analysis.md                          # Original analysis of top public kernels
├── CLAUDE.md                            # Project guide (deadline, format, scoring)
│
├── notebooks/
│   ├── ensemble_kaggle.ipynb            # The Kaggle notebook (10 versions pushed)
│   ├── ensemble-kernel-metadata.json    # 10 kernel_sources attached
│   └── train_kaggle.ipynb               # GRAVEYARD — TinyCNN training, do not run
│
├── src/
│   ├── analyze_candidates.py            # Phase 2 calibration analysis
│   ├── prioritize_handbuild.py          # Ranks tasks by gain potential
│   ├── view_task.py                     # ASCII task viewer
│   ├── handbuild/
│   │   ├── validate_rule.py             # Python rule → all-example validator
│   │   ├── test_onnx.py                 # ONNX validation + cost + delta calc
│   │   ├── task389.py + build_task389.py   # ✅ Working (+1.91 LB)
│   │   ├── task261.py + build_task261.py   # ❌ Failed on grader
│   │   └── task095.py + build_task095.py   # ⏳ Untested
│   ├── model.py, train.py, ...          # GRAVEYARD — TinyCNN, do not use
│
├── output/
│   ├── candidates.csv                   # Per-source per-task cost matrix
│   ├── calibration.md                   # Phase 2 report
│   └── handbuild_onnx/                  # The hand-built ONNX files
│
└── submit/                              # Old TinyCNN submissions (graveyard)

/tmp/                                    # Working directories from this session
├── bundle-tests/                        # Downloaded submission.zips from public kernels
│   ├── best-score-the-2026-neurogolf-championship/submission.zip  # = 6042 bundle (1.46MB)
│   ├── 6029-09-lb-neurogolf-all-task-onnx-solution/submission.zip
│   ├── neurogolf-multi-source-onnx-solver/submission.zip
│   └── the-2026-neurogolf-championship/submission.zip
├── bundle-tests2/                       # More bundles fetched
└── v8, v9, v10, v11, v12/               # Local working dirs for each version
```

---

## How To Resume Quickly

```bash
cd /Users/firmanimam/Kaggle/neurogolf-2026

# 1. See current state
kaggle competitions submissions neurogolf-2026 | head -15

# 2. Check leaderboard / bronze threshold
kaggle competitions leaderboard neurogolf-2026 -d -p /tmp/lb && \
  unzip -p /tmp/lb/*.zip | sed -n '132p'   # ~bronze cutoff (top 10%)

# 3. To run a new ensemble:
#    Edit notebooks/ensemble_kaggle.ipynb (Phase 4 inject cell for new hand-builds)
#    Push:
cp notebooks/ensemble_kaggle.ipynb /tmp/neurogolf-ensemble-push/
cp notebooks/ensemble-kernel-metadata.json /tmp/neurogolf-ensemble-push/kernel-metadata.json
cd /tmp/neurogolf-ensemble-push && kaggle kernels push -p .

# 4. To submit a specific assembly locally:
#    (Faster than re-running the notebook — just edit local submission.zip)
mkdir -p /tmp/vNEW
unzip -q -o /tmp/bundle-tests/best-score-the-2026-neurogolf-championship/submission.zip -d /tmp/vNEW/onnx_files
cp output/handbuild_onnx/task389.onnx /tmp/vNEW/onnx_files/  # add any handbuilds
cd /tmp/vNEW/onnx_files && zip -q -r /tmp/vNEW/submission.zip *.onnx && cd -
kaggle competitions submit -c neurogolf-2026 -f /tmp/vNEW/submission.zip -m "description"

# 5. To test a new hand-build locally:
python3 src/view_task.py <tid>                # see the task
# write src/handbuild/task<NNN>.py with solve()
python3 src/handbuild/task<NNN>.py            # validate Python rule
# write src/handbuild/build_task<NNN>.py with ONNX construction
python3 src/handbuild/build_task<NNN>.py      # build ONNX
python3 src/handbuild/test_onnx.py <tid>      # validate + measure cost
```

---

## Key Technical Notes

### ONNX opset 17 gotchas
- `ReduceSum`: takes `axes` as INPUT (changed in opset 13)
- `ReduceMax`: takes `axes` as ATTRIBUTE in opset 17, changed to input in opset 18 → if checker fails with "input size 2 not in range [1, 1]", switch to attribute syntax `helper.make_node("ReduceMax", ["input"], ["out"], axes=[1], keepdims=1)`
- `Pad`: takes `pads` as INPUT tensor (8 values for 4D: `[start_N, start_C, start_H, start_W, end_N, end_C, end_H, end_W]`)
- `Slice`: takes `starts`, `ends`, `axes` (optional `steps`) as INPUT tensors

### Kaggle environment quirks
- Don't install `onnxruntime-gpu` — it requires CUDA 11 but Kaggle has CUDA 12 → floods logs with errors, kills kernel
- `onnx==1.16.1` install removes pre-installed onnxruntime → must explicitly add `onnxruntime` to pip install
- The competition's `neurogolf_utils` module IS pre-installed but is an OLDER version missing `verify_subset` and `score_network` → inline these in the notebook
- Some ONNX files segfault during ORT loading or `onnx_tool` profiling → **subprocess isolation** is mandatory (see `_Isolator` class in notebook cell `cell-utils`)

### Submission limits
- 100 submissions per day (NOT 5 as initially assumed)
- This means experimental submissions are cheap; don't be conservative

### Bundle slugs that worked
```
octaviograu/6042-85-per-task-hand-built-onnx-solvers       # 6042.85 - BEST single bundle
mirzayasirabdullah07/best-score-the-2026-neurogolf-championship  # 6042.85 - clone of 6042
jsrdcht/6029-09-lb-neurogolf-all-task-onnx-solution        # 6029.09
jonathanchan/ngc26-constraint-smart-logic-mix-blending     # ~5800
kojimar/5800-55-lb-neurogolf-task-level-onnx-blend         # ~5800
octaviograu/neurogolf-2026-block-lb-drilling-5740-30       # 5740.30
afr1ste/neurogolf-5689-51-current-rules-open-solution      # 5689.51
vyankteshdwivedi/neurogolf-multi-source-onnx-solver        # 5740
konbu17/neurogolf-2026-may-8-updated                       # 5571
thisray/neurogolf-4808-21-post-apr-28-update               # 4866
svanikkolli/arc-nano-engine                                # meta — no wins
```

### Bundles that errored on submission (corrupt or invalid)
```
jiweiliu/kaggle-agent-ensemble-with-yash9439
karnakbaevarthur/cross-source-ensemble
yash9439/neurogolf-emsembling
```

---

## Final Snapshot — Where We Stand

```
Rank:              ~340 / 1,319 teams
Public score:      6,044.76 (v11)
Bronze cutoff:     6,055.00  (need +10.24 LB)
Silver cutoff:     ~6,225    (need +180 LB)
Gold cutoff:       ~7,300    (need +1,255 LB)
Leader:            7,516
Top 10% (bronze):  132 teams
Top 5% (silver):   66 teams
Top 1% (gold):     14 teams
```

**Most efficient path to bronze:** 5–7 working hand-builds (each ~+1–2 LB) layered on the 6042 base. Each hand-build takes 30–60 min including the unknown grader-pass risk.

**Risk to mitigate FIRST next session:** the task261 mystery. We must understand why a locally-validated hand-build can fail on the grader before building 5 more that might all fail.
