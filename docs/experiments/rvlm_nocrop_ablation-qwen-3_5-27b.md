# rvlm_nocrop_ablation — Qwen 3.5 27B (val)

## Hypothesis / question

A one-affordance ablation of `rvlm` that removes **cropping/zooming**. In
`rvlm`, `batch_look(requests)` takes `(image, query)` tuples where the
image may be a full page *or* an arbitrary crop (`pages[i].crop(...)`);
here it takes `(page_index, query)` — whole pages only, no PIL objects in
scope to crop. The prompt is byte-close to `rvlm` with crop references
removed (it neither encourages nor forbids cropping). Everything else —
LeanRLM REPL, recursive VLM sub-call, profile/answer rules — is identical.

The **paired Δ vs `rvlm` isolates cropping**: how much of `rvlm`'s score
comes from the agent's ability to zoom into a region vs reading whole
pages. Hypothesis: it bites hardest on detail-dense categories
(engineering_drawing, business_report, maps) where a whole-page read
can't resolve fine print; the agent may also take *more* iterations
(re-reading pages with different queries to compensate for losing crop).

## Setup

- Solver: `rvlm_nocrop_ablation` (`batch_look` by page index; no crop)
- Model: Qwen 3.5 27B local vllm 8927 (lm + vlm), `enable_thinking=false`
- Profile: DocVQA-2026 (default); max_iterations 25 (same as `rvlm`)
- max_concurrency: 24 (high-concurrency phase, GPU free post-matrix)
- Added 2026-06-03 (user request); n=8.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_nocrop_ablation \
  data.split=val data.num_samples=null \
  max_concurrency=24 \
  run_id=rvlm-nocrop-cmp-val-tN
```

## Per-trial table

| Trial | run_id | Score | Correct | Wall | Notes |
|---|---|---|---|---|---|
| — | (running) | — | — | — | n=8 in progress |

## Summary

n=0 (running). Mean ± std at n=8; paired Δ vs `rvlm` (39.38% ± 1.49)
isolates the contribution of cropping.

## Comparison

vs `rvlm` (vision + crop) — same scaffold, only cropping removed. Also
informative vs `rvlm` on per-category breakdown (expect the largest drop
on detail-dense categories) and vs the efficiency metric (does losing
crop raise iteration count?).

## Status

in progress (n=0 of 8)
