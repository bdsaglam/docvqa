# direct_vlm — Qwen 3.5 27B (val)

## Hypothesis / question

The "see it yourself" baseline: the REPL agent `display()`s page images
directly into its **own** context (no VLM sub-call, no `batch_look`).
Isolates whether the lift in `rvlm` comes from *recursive* perception (a
dedicated VLM sub-call answering focused questions) vs simply giving the
reasoning agent the raw pixels. If direct image access were enough,
`direct_vlm` would match `rvlm`; the gap measures the value of the
sub-call indirection.

## Setup

- Solver: `direct_vlm` (display images into own context; no VLM sub-call)
- Model: Qwen 3.5 27B local vllm 8927 (lm + vlm), `enable_thinking=false`
- Profile: DocVQA-2026 (default)
- max_concurrency: 24
- Part of the 7-solver comparison re-run (val, n=3), 2026-06-01.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=direct_vlm \
  data.split=val data.num_samples=null \
  max_concurrency=24 \
  run_id=direct-vlm-cmp-val-tN
```

## Per-trial table

| Trial | run_id | Score | Correct | Wall | Notes |
|---|---|---|---|---|---|
| t1 | `direct-vlm-cmp-val-t1` | 21.25% | 17/80 | ~4h+ | very slow (long-image contexts); recovered once at 24/25; business_report/comics/maps 0/10 |

Per-category (t1): business_report 0/10, comics 0/10, engineering_drawing
3/10, infographics 6/10, maps 0/10, science_paper 2/10, science_poster
3/10, slide 3/10.

## Summary

n=1 so far (t2 running, t3 queued). Mean ± std at n=8.

## Comparison

`rvlm` t1 (40.00%) − this (21.25%) = **+18.75pp** at n=1 — putting raw
pixels in the agent's own context recovers far less than routing focused
questions through a dedicated VLM sub-call. Close to the
`raw_vlm_multi_baseline` floor (18.75% t1): same collapse on detail
categories (business_report, comics, maps all 0/10), confirming that
direct image access without recursive querying can't resolve fine print.
Lock paired Δ at n=3.

## Observations / caveats

- Very slow: displaying full pages into the agent's own context makes
  each turn a long-image inference; t1 ran ~4h+ and died once at 24/25
  (auto-recovered, resumable).

## Status

in progress (n=1 of 8)
