# amax7 queue (adaptive host)

amax7 is the adaptive host — runs critical-path experiments where the
result might change the experiment plan. Tighter feedback loops; one
cell at a time; replan after each result.

**vllm:** Qwen 3.5 27B at `localhost:8927`.

## In progress

### `[→]` rvlm_skeletal n=1 val (task #32) — started 2026-05-29T11:??+03

Drops the 3 doc-shape pattern bullets from `rvlm_minimal` (high-density,
many-page, counting). Keeps APPROACH + verify-under-VLM-stochasticity
principle. Tests whether the patterns were doing work or were
window-dressing.

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_skeletal \
  data.split=val data.num_samples=null \
  max_concurrency=32 \
  run_id=rvlm-skeletal-val-t1
```

- Expected wall: ~50 min
- Compare to: rvlm_minimal n=8 mean 42.03% (SD 2.21pp).
- Decision rule: if within ±2pp of minimal → run rvlm_naked next; if
  >3pp drop → patterns were load-bearing, halt further stripping.

### `[ ]` rvlm_naked n=1 val (task #33) — auto-starts after skeletal

Strips everything except DATA + TOOLS + faithfulness + OUTPUT FORMAT.
The pure "give the agent a recursive-perception tool and let it figure
it out" test. Strongest possible result if Δ ≈ 0.

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_naked \
  data.split=val data.num_samples=null \
  max_concurrency=32 \
  run_id=rvlm-naked-val-t1
```

- Auto-chains after skeletal in the same tmux session `rvlm-strip-chain`.
- Decision rule: if naked ≈ skeletal ≈ minimal → naked becomes the
  headline; absolute floor of "what the method needs to work" is just
  the tool API.

## Queued

### 1. `[ ]` rvlm_ocr n=1 val (task #14)

Locks the clean OCR-extension number. Current `rvlm_full` legacy data is
confounded with `look()` ergonomic wrapper. This is the clean cell.

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_ocr \
  data.split=val data.num_samples=null \
  max_concurrency=16 \
  run_id=rvlm-ocr-val-t1
```

- Expected wall: ~50min
- Compare to: `rvlm` val (OCR-free); expect approximately equal on
  DocVQA-2026 (moderate-length docs).

### 2. `[ ]` direct_vlm n=1 val (task #19)

Alternative architecture data point. Tests whether single-multimodal-model
REPL can match the recursive sub-call structure. Important for the
paper's "the recursive sub-call is the load-bearing mechanism" claim.

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=direct_vlm \
  data.split=val data.num_samples=null \
  max_concurrency=16 \
  run_id=direct-vlm-val-t1
```

- Expected wall: ~30-50min (depends on display() bandwidth)
- Compare to: `rvlm` headline + `raw_vlm_multi` baseline (20.0% val SC-8)

## Done

### `[✓]` rvlm_minimal n=8 val — generality test (task #31) — 2026-05-29

run_ids: `rvlm-minimal-val-t{1..8}` · **n=8 mean 42.03% ± 2.21pp**
(range 38.75–45.00) on Qwen 3.5 27B local, c=32.

Paired vs `rvlm_unified` t1..t8: **Δ = +1.09pp [CI95: −3.14, +5.33]**,
paired t = 0.611, df = 7 (n.s.). Lands cleanly in the pre-set
"≈ 0pp / within paired noise" band → **`rvlm_minimal` is the proposed
method.** The 10.7 kB of hand-crafted DocVQA-2026 category tips and
the engineering-drawing-specific VLM sub-call signature are not
load-bearing; the recursive-perception mechanism is.

Variance note worth surfacing for the paper: minimal σ = 2.21pp,
unified σ = 4.05pp — almost 2× tighter trial-to-trial. Plausibly the
agent stops being yanked between competing per-category prescriptions
when the dispatch guess is wrong. Per-trial table + paired analysis:
`docs/experiments/rvlm-minimal-generality.md`.

Refill notes: 4/8 trials needed pass-2 refill — `science_paper_1`
systematically hits the 4h task_timeout (~50% per-attempt success
rate, agent-loop / vllm-hang specific to this doc). All cleared on
the sequential refill pass.

### `[✓]` unified-tips n=8 val (tasks #25 + #28) — 2026-05-28

run_ids: `rvlm-unified-val-t{1..8}` · **n=8 mean 40.94% ± 4.05pp**
(range 35.0–47.5) on Qwen 3.5 27B local, c=32.

Paired vs amax1's `rvlm` baseline at t1..t7: Δ mean = +0.71pp,
SE 1.72pp, 95% CI [−3.50, +4.93]pp — well inside noise. Per-trial
table + paired analysis: `docs/experiments/unified-category-tips-ablation.md`.

Variance asymmetry worth flagging for the promote-to-default
decision: unified σ=4.05pp vs rvlm σ=2.38pp (~1.7×). Final paired
analysis lands when amax1's t8 commits.

## Decision rules (set in advance)

- **unified-tips Δ ≈ 0pp** → promote unified to default; replace rvlm
  cells with rvlm_unified in subsequent cells.
- **rvlm_ocr ≈ rvlm on val** → OCR neutral on moderate-length docs;
  paper's §B doc-length-axis claim holds. Push to MMLongBench-Doc next.
- **direct_vlm < rvlm by 5+pp** → recursive sub-call is load-bearing;
  paper §C prediction 3 supported.
- **direct_vlm ≈ rvlm** → architecture-agnostic; reframing needed
  (the sub-call may not be the load-bearing piece — context-rationing
  is, regardless of architecture).
