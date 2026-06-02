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
| t1 | `official-cmp-val-t1` | 15.00% | 12/80 | — | business_report/comics/maps 0/10; below our parity-prompt raw_vlm_multi |
| t2 | `official-cmp-val-t2` | 18.75% | 15/80 | — | business_report/comics 0/10, maps 2/10 |
| t3 | `official-cmp-val-t3` | 18.75% | 15/80 | — | comics/maps 0/10 |

Per-category (t2): business_report 0/10, comics 0/10, engineering_drawing
3/10, infographics 4/10, maps 2/10, science_paper 2/10, science_poster
1/10, slide 3/10.

Per-category (t1): business_report 0/10, comics 0/10, engineering_drawing
3/10, infographics 4/10, maps 0/10, science_paper 2/10, science_poster
1/10, slide 2/10.

Per-category (t3): business_report 1/10, comics 0/10, engineering_drawing
2/10, infographics 4/10, maps 0/10, science_paper 2/10, science_poster
3/10, slide 3/10.

Prior reference (kit-faithful, max_pages=null, no downscale): **21.67% ±
1.91pp, n=3** (recorded in `official_baseline.yaml`). The downscaled
`max_pages=10` config here may differ; this re-run establishes the
comparison-matrix number.

## Summary

n=3 so far (target n=8): 15.00 / 18.75 / 18.75 → running mean **17.50% ± 2.17pp**, below our parity-prompt `raw_vlm_multi_baseline` (20.83%, n=3), so the minimized prompt is NOT crippling the baseline (if anything the official prompt + max_pages=10 downscale is weaker). Mean ± std at n=8.

## Comparison

External anchor — not a paired Δ vs `rvlm` per se, but bounds where the
competition's own VLM baseline sits relative to our `raw_vlm_multi_baseline`
(same shape, parity prompt). Close agreement validates the parity prompt.

## Status

in progress (n=3 of 8)
