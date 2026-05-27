# amax7 queue (adaptive host)

amax7 is the adaptive host — runs critical-path experiments where the
result might change the experiment plan. Tighter feedback loops; one
cell at a time; replan after each result.

**vllm:** Qwen 3.5 27B at `localhost:8927`.

## In progress

### `[→]` unified-tips n=1 val (task #25)  — started 2026-05-28T22:30+03

Tmux session `unified-tips-t1`. `max_concurrency=32` (per c=32 dispatch).
Tests whether per-document category metadata is required. If Δ ≈ 0pp vs
rvlm headline, we **promote unified to default** and many downstream
cells get cleaner (no `doc_category` dependence).

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_unified \
  data.split=val data.num_samples=null \
  max_concurrency=32 \
  run_id=rvlm-unified-val-t1
```

- Expected wall: ~30-40min (c=32 faster than the earlier c=16 estimate)
- Compare to: `rvlm` (proposed method) headline val per-trial mean ~42.8%

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

(none yet under new naming)

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
