# rvlm — Qwen 3.5 27B (val)

## Hypothesis / question

Anchor for the 7-solver comparison re-run: the proposed OCR-free
recursive-perception method (`rvlm`, VLM sub-call via `batch_look`) under
the post-cleanup code (minimized prompts, parity-stripped, per-call
`num_retries=5` only — whole-agent `@retry` removed). This is the
reference every other solver's Δ is measured against.

## Setup

- Solver: `rvlm` (proposed method; OCR-free)
- Model: Qwen 3.5 27B local vllm 8927 (lm + vlm), `enable_thinking=false`
- Profile: DocVQA-2026 (default)
- max_concurrency: 24
- Part of the 7-solver comparison re-run (val, n=3), 2026-06-01.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm \
  data.split=val data.num_samples=null \
  max_concurrency=24 \
  run_id=rvlm-cmp-val-tN
```

## Per-trial table

| Trial | run_id | Score | Correct | Wall | Notes |
|---|---|---|---|---|---|
| t1 | `rvlm-cmp-val-t1` | 40.00% | 32/80 | ~3.5h | last doc (science_paper dense table) churned to iter cap; recovered once at 24/25 |
| t2 | `rvlm-cmp-val-t2` | 38.75% | 31/80 | — | recovered once at 24/25 (long-tail doc) |
| t3 | `rvlm-cmp-val-t3` | 37.50% | 30/80 | — | comics 4/10 (best), maps 1/10 |

Per-category (t1): business_report 6/10, comics 3/10, engineering_drawing
5/10, infographics 7/10, maps 0/10, science_paper 3/10, science_poster
3/10, slide 5/10.

Per-category (t2): business_report 4/10, comics 3/10, engineering_drawing
6/10, infographics 7/10, maps 0/10, science_paper 2/10, science_poster
4/10, slide 5/10.

Per-category (t3): business_report 5/10, comics 4/10, engineering_drawing
5/10, infographics 5/10, maps 1/10, science_paper 2/10, science_poster
4/10, slide 4/10.

## Summary

**n=3 complete: 38.75% ± 1.25pp** (t1 40.00%, t2 38.75%, t3 37.50%) —
tight variance. This is the reference the matrix is measured against.
maps is the consistent weak spot (0/0/1).

## Comparison

Reference method — others compared against this. Paired Δ vs each solver
(both at n=3) computed in `docs/results.md` once all 27 cells land. Known
so far: vs `rvlm_ocr_ablation` (37.08% ± 2.60pp) → **+1.67pp** — OCR adds
nothing over the OCR-free method (slightly below). vs the n=1 baselines:
raw_vlm_multi 18.75% (+~20pp), repl_only 7.5% (+~31pp).

## Observations / caveats

- One pathological science_paper doc (dense dataset table) looped on
  VLM-read disagreements to the 32-iteration cap — dominated wall time.
  Its process died once at 24/25 and was auto-recovered (resumable).

## Status

complete (n=3 of 3)
