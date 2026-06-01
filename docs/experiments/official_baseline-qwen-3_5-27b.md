# official_baseline — Qwen 3.5 27B (val)

## Hypothesis / question

The **external anchor**: the literal DocVQA-2026 competition baseline —
all document pages sent to the VLM in one multi-image chat-completion
request, using the kit's verbatim `MASTER_PROMPT` (mandatory reasoning
protocol + `FINAL ANSWER:` format). No REPL, no recursive perception, no
category tips.

Role in the matrix: ties our comparison table to the competition's
published numbers, and shows our minimized parity prompt (used by
`raw_vlm_multi_baseline`) isn't sandbagging the no-scaffold floor —
`official_baseline` is essentially `raw_vlm_multi_baseline` with the
official prompt instead of our parity-stripped one. A large gap between
the two would mean prompt, not scaffold, drives part of the lift.

## Setup

- Solver: `official_baseline` (multi-image VLM, competition `MASTER_PROMPT`)
- Model: Qwen 3.5 27B local vllm 8927 (vlm), `enable_thinking=false`
- Profile: DocVQA-2026 (default)
- `max_pages: 10`, `max_image_pixels: 1500000` (per the solver yaml — Qwen
  vLLM lacks the closed-frontier image-API downscaling the kit assumes;
  without this, native-res pages overflow Qwen's 131k context on test).
- Added 2026-06-01 as an extra baseline (user request); n=3.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=official_baseline \
  data.split=val data.num_samples=null \
  max_concurrency=8 \
  run_id=official-cmp-val-tN
```

## Per-trial table

| Trial | run_id | Score | Correct | Wall | Notes |
|---|---|---|---|---|---|
| — | (queued) | — | — | — | n=3 queued |

Prior reference (kit-faithful, max_pages=null, no downscale): **21.67% ±
1.91pp, n=3** (recorded in `official_baseline.yaml`). The downscaled
`max_pages=10` config here may differ; this re-run establishes the
comparison-matrix number.

## Summary

Queued (n=0). Mean ± std at n=8.

## Comparison

External anchor — not a paired Δ vs `rvlm` per se, but bounds where the
competition's own VLM baseline sits relative to our `raw_vlm_multi_baseline`
(same shape, parity prompt). Close agreement validates the parity prompt.

## Status

queued (n=0 of 8)
