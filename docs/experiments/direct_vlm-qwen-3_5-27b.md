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
| t2 | `direct-vlm-cmp-val-t2` | 21.25% | 17/80 | — | identical to t1; comics/maps 0/10; recovered once at 23/25 |
| t3 | `direct-vlm-cmp-val-t3` | 18.75% | 15/80 | — | business_report 0/10, slide 1/10; **died+resumed 9×** on slow long-image docs |
| t4 | `direct-vlm-cmp-val-t4` | 26.25% | 21/80 | — | best trial; science_poster 5/10, infographics 6/10; comics 0/10; **died+resumed ~8×**, stuck at doc-23 for ~5 ticks |
| t5 | `direct-vlm-cmp-val-t5` | 21.25% | 17/80 | — | maps 0/10, slide 4/10; died+resumed several× |
| t6 | `direct-vlm-cmp-val-t6` | 26.25% | 21/80 | — | ties t4 (best); science_poster 5/10, infographics 6/10; died+resumed several× |
| t7 | `direct-vlm-cmp-val-t7` | 20.00% | 16/80 | — | comics/maps 0/10; died+resumed several× |

Per-category (t4): business_report 1/10, comics 0/10, engineering_drawing
1/10, infographics 6/10, maps 1/10, science_paper 3/10, science_poster
5/10, slide 4/10.

Per-category (t3): business_report 0/10, comics 1/10, engineering_drawing
2/10, infographics 5/10, maps 1/10, science_paper 2/10, science_poster
3/10, slide 1/10.

Per-category (t1): business_report 0/10, comics 0/10, engineering_drawing
3/10, infographics 6/10, maps 0/10, science_paper 2/10, science_poster
3/10, slide 3/10.

Per-category (t2): business_report 1/10, comics 0/10, engineering_drawing
3/10, infographics 6/10, maps 0/10, science_paper 2/10, science_poster
2/10, slide 3/10.

## Summary

n=7 so far (target n=8): 21.25 / 21.25 / 18.75 / 26.25 / 21.25 / 26.25 / 20.00 → running mean **22.14% ±
2.95pp**. ≈ `raw_vlm_multi` (20.50%) — putting raw pixels in the agent's
own context buys no more than a single multi-image read. vs `rvlm`
(39.79%): **Δ +17.91pp**. **Operational note:** by far the slowest
solver — each trial dies and resumes ~5–9× on long-image docs (the slow
last few pages), making it the run-time bottleneck of the matrix. Mean ±
std at n=8.

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
