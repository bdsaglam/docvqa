# amax7 queue (adaptive host)

amax7 is the adaptive host — runs critical-path experiments where the
result might change the experiment plan. Tighter feedback loops; one
cell at a time; replan after each result.

**vllm:** Qwen 3.5 27B at `localhost:8927`.

## In progress

### `[→]` unified-tips n=2..n=8 val (task #28) — started 2026-05-28T02:10+03

User opted to skip the D-008 n=2 escalation step and run all 8 trials
directly. t1 = 45.0%. The paired-conditions rvlm baseline runs on amax1
in parallel (see coordination/amax1.md cell #1) — same model, prompts,
c, paired by trial number.

```bash
tmux new-session -d -s unified-tips-chain bash scripts/run_unified_tips_chain.sh
# Resumes-on-crash: bash scripts/run_unified_tips_chain.sh (it skips
# completed run_ids).
```

- Expected wall: ~7 × ~50min ≈ 6h (Qwen 3.5 27B vllm, c=32)
- Compare per-trial-pair against `rvlm-val-t{1..8}` on amax1
- Decision gate: see `docs/experiments/unified-category-tips-ablation.md`

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

### `[✓]` unified-tips n=1 val (task #25) — 2026-05-28

run_id: `rvlm-unified-val-t1` · **45.0%** (36/80) on Qwen 3.5 27B local,
c=32. Per-category: infographics 80%, comics 60%, eng_drawing 50%,
slide 50%, science_poster 50%, business_report 40%, science_paper 30%,
**maps 0%**.

Δ vs per-category-dispatch baseline per-trial mean (rvlm legacy
`leanest_solo` n=8 = 42.8%): **+2.2pp**, well inside trial-noise band
(~3pp std). Lands in the "Δ ≈ 0pp → promote unified to default" cell of
the decision rules; needs n=2 to confirm before promoting.

Writeup updated at `docs/experiments/unified-category-tips-ablation.md`.
Next: file n=2 trial if promotion is the direction the paper wants.

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
