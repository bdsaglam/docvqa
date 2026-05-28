# amax7 queue (adaptive host)

amax7 is the adaptive host — runs critical-path experiments where the
result might change the experiment plan. Tighter feedback loops; one
cell at a time; replan after each result.

**vllm:** Qwen 3.5 27B at `localhost:8927`.

## In progress

### `[→]` rvlm_minimal n=8 val — generality test (task #31) — started 2026-05-28T16:??+03

Tests whether the proposed method's score depends on benchmark-tuned
category tips. Strip the 8 DocVQA-2026 category tip blocks; keep only
4 generic document-shape patterns in the solver body. Paired
comparison vs `rvlm_unified` (amax7 t1..t8, already done) by trial.

```bash
tmux new-session -d -s rvlm-minimal-chain bash scripts/run_rvlm_minimal_chain.sh
# Resumable: bash scripts/run_rvlm_minimal_chain.sh (skips completed run_ids).
```

- Expected wall: 8 × ~50min ≈ 7h (Qwen 3.5 27B vllm, c=32)
- run_ids: `rvlm-minimal-val-t{1..8}`
- Decision gate + paired analysis: `docs/experiments/rvlm-minimal-generality.md`
- Strongest paper outcome: Δ ≈ 0pp → `rvlm_minimal` becomes the
  proposed method; `rvlm` / `rvlm_unified` become engineering
  ablations showing what tip-tuning buys.

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
