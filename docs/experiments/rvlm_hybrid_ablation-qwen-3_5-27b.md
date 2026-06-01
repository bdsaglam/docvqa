# rvlm_hybrid_ablation — Qwen 3.5 27B (val)

## Hypothesis / question

The hybrid perception ablation: `rvlm` plus a `display()` channel that
puts page images into the agent's *own* context alongside the
`batch_look` VLM sub-call (`ask_vlm`). Tests whether giving the REPL
agent direct image access on top of recursive perception helps, hurts,
or is neutral vs the OCR-free `rvlm` reference (which perceives only
through the VLM sub-call).

## Setup

- Solver: `rvlm_hybrid_ablation` (display + ask_vlm)
- Model: Qwen 3.5 27B local vllm 8927 (lm + vlm), `enable_thinking=false`
- Profile: DocVQA-2026 (default)
- max_concurrency: 24
- Part of the 7-solver comparison re-run (val, n=3), 2026-06-01.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_hybrid_ablation \
  data.split=val data.num_samples=null \
  max_concurrency=24 \
  run_id=rvlm-hybrid-cmp-val-tN
```

## Per-trial table

| Trial | run_id | Score | Correct | Wall | Notes |
|---|---|---|---|---|---|
| t1 | `rvlm-hybrid-cmp-val-t1` | 40.00% | 32/80 | — | ties `rvlm` t1; science_poster 7/10, comics 1/10 |
| t2 | `rvlm-hybrid-cmp-val-t2` | 38.75% | 31/80 | — | ties `rvlm` t2 exactly (38.75%) |

Per-category (t1): business_report 5/10, comics 1/10, engineering_drawing
4/10, infographics 6/10, maps 1/10, science_paper 3/10, science_poster
7/10, slide 5/10.

Per-category (t2): business_report 4/10, comics 1/10, engineering_drawing
3/10, infographics 6/10, maps 1/10, science_paper 4/10, science_poster
6/10, slide 6/10.

## Summary

n=2 so far (t3 queued). t1 40.00%, t2 38.75% — tracks `rvlm` trial-for-trial
(40.00/38.75 both), reinforcing that the direct `display()` channel is
redundant on top of the VLM sub-call. Mean ± std at n=3.

## Comparison

`rvlm` t1 (40.00%) vs this (40.00%) = **0.00pp** at n=1 — adding a direct
`display()` image channel on top of the VLM sub-call neither helps nor
hurts here. Per-category mix shifts slightly (hybrid +4 science_poster,
−2 comics vs `rvlm` t1) but overall is identical. Lock paired Δ at n=3;
if it stays ~0 it supports the OCR-free `rvlm` framing — recursive
perception via the sub-call already captures the signal, direct image
access is redundant.

## Observations / caveats

- (none yet)

## Status

in progress (n=1 of 3)
