# raw_vlm_multi_baseline — Qwen 3.5 27B (val)

## Hypothesis / question

The stronger raw-VLM baseline: feed all page images to the VLM in one
multi-image prompt, no REPL / no recursive perception. Quantifies the
"no-scaffold" floor the `rvlm` method lifts off — the prediction-3
active-perception delta. Minimized prompt (parity with `rvlm`).

## Setup

- Solver: `raw_vlm_multi_baseline` (multi-image, single VLM call, no REPL)
- Model: Qwen 3.5 27B local vllm 8927 (lm + vlm), `enable_thinking=false`
- Profile: DocVQA-2026 (default)
- max_concurrency: 24
- Part of the 7-solver comparison re-run (val, n=3), 2026-06-01.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=raw_vlm_multi_baseline \
  data.split=val data.num_samples=null \
  max_concurrency=24 \
  run_id=raw-vlm-multi-cmp-val-tN
```

## Per-trial table

| Trial | run_id | Score | Correct | Wall | Notes |
|---|---|---|---|---|---|
| t1 | `raw-vlm-multi-cmp-val-t1` | 18.75% | 15/80 | — | collapses on business_report 0/10, comics 0/10, engineering_drawing 1/10 |
| t2 | `raw-vlm-multi-cmp-val-t2` | 22.50% | 18/80 | — | business_report 0/10, comics 0/10 again |
| t3 | `raw-vlm-multi-cmp-val-t3` | 21.25% | 17/80 | — | business_report 0/10, comics 0/10 (third time) |

Per-category (t1): business_report 0/10, comics 0/10, engineering_drawing
1/10, infographics 4/10, maps 1/10, science_paper 3/10, science_poster
3/10, slide 3/10.

Per-category (t2): business_report 0/10, comics 0/10, engineering_drawing
2/10, infographics 5/10, maps 1/10, science_paper 3/10, science_poster
4/10, slide 3/10.

Per-category (t3): business_report 0/10, comics 0/10, engineering_drawing
2/10, infographics 5/10, maps 1/10, science_paper 3/10, science_poster
3/10, slide 3/10.

## Summary

**n=3 so far (target n=8): 20.83% ± 1.91pp** (t1 18.75%, t2 22.50%, t3 21.25%).
business_report and comics are **0/10 in all three trials** — a single
multi-image read cannot resolve their fine print, the exact failure the
recursive scaffold targets.

## Comparison

Baseline floor. **rvlm (38.75% ± 1.25pp) − this (20.83% ± 1.91pp) =
+17.92pp** scaffold lift at n=3 — biggest gaps in the zoom-then-read
categories (business_report, comics near-zero here vs rvlm's 4-6/10).

## Observations / caveats

- Near-zero on detail-heavy categories where a single multi-image read
  can't resolve fine print — the failure the recursive scaffold targets.

## Status

in progress (n=3 of 8)
