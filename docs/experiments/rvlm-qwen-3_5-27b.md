# rvlm — Qwen 3.5 27B (val)

## Hypothesis / question

Anchor for the 8-solver comparison: the proposed OCR-free
recursive-perception method (`rvlm`, VLM sub-call via `batch_look`) under
the post-cleanup code (minimized prompts, parity-stripped, per-call
`num_retries=5` only — whole-agent `@retry` removed). This is the
reference every other solver's Δ is measured against.

## Setup

- Solver: `rvlm` (proposed method; OCR-free)
- Model: Qwen 3.5 27B local vllm 8927 (lm + vlm), `enable_thinking=false`
- Profile: DocVQA-2026 (default)
- max_concurrency: 24
- Headline cell of the 8-solver comparison matrix (val, n=8).

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

*Original measurement batch (per-category texture; per-trial artifacts since
deleted → not the pass@k source). Canonical n=8 = the re-run in Summary below.*

| Trial | run_id | Score | Correct | Wall | Notes |
|---|---|---|---|---|---|
| t1 | `rvlm-cmp-val-t1` | 40.00% | 32/80 | ~3.5h | last doc (science_paper dense table) churned to iter cap; recovered once at 24/25 |
| t2 | `rvlm-cmp-val-t2` | 38.75% | 31/80 | — | recovered once at 24/25 (long-tail doc) |
| t3 | `rvlm-cmp-val-t3` | 37.50% | 30/80 | — | comics 4/10 (best), maps 1/10 |
| t4 | `rvlm-cmp-val-t4` | 41.25% | 33/80 | — | best trial; engineering_drawing 6/10, science_paper 4/10; recovered once at 23/25 |
| t5 | `rvlm-cmp-val-t5` | 41.25% | 33/80 | — | ties t4 (best); science_paper 5/10, infographics 7/10, maps 0/10 |
| t6 | `rvlm-cmp-val-t6` | 40.00% | 32/80 | — | business_report 6/10, comics 4/10, maps 0/10; recovered once at 24/25 |
| t7 | `rvlm-cmp-val-t7` | 37.50% | 30/80 | — | science_poster 6/10, infographics 6/10, maps 1/10; recovered once at 24/25 |
| t8 | `rvlm-cmp-val-t8` | 38.75% | 31/80 | — | comics 6/10 (best), infographics 7/10, maps 0/10; recovered once at 23/25 |

Per-category (t6): business_report 6/10, comics 4/10, engineering_drawing
5/10, infographics 5/10, maps 0/10, science_paper 2/10, science_poster
5/10, slide 5/10.

Per-category (t5): business_report 5/10, comics 2/10, engineering_drawing
5/10, infographics 7/10, maps 0/10, science_paper 5/10, science_poster
5/10, slide 4/10.

Per-category (t4): business_report 4/10, comics 2/10, engineering_drawing
6/10, infographics 6/10, maps 1/10, science_paper 4/10, science_poster
5/10, slide 5/10.

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

**n=8 (retained per-trial artifacts): 41.88% ± 5.79, pass@8 68.75, SC@8 47.50**
(per-trial 30.0 / 43.8 / 45.0 / 50.0 / 43.8 / 42.5 / 41.2 / 38.8). The relatively
high std is driven by the single **t1 = 30.0 low outlier**; the other 7 trials
cluster 38.8–50.0. `science_paper_3` / `comics_2` are the recurring
degenerate-loop traps (rvlm has no exec-timeout; cleared via kill+resume). This
is the canonical headline cell and the reference all other Δ are measured against.

> **Provenance.** An earlier measurement batch of the same cell (n=8, **39.38% ±
> 1.49**) had its per-trial `submission.json`s deleted in a disk cleanup, so it
> carried no pass@k/SC@k — the run above re-measures with retained artifacts and
> is canonical. The per-trial **table above** (with per-category texture) is that
> original batch, kept for the category-level detail; the re-roll lands +2.5pp
> higher with wider variance (the original's ±1.49 vs 41.88's ±5.79).

## Comparison

Reference method — every other solver's Δ is measured against this. The full
n=8 matrix and per-cell Δ live in `docs/results.md`: the rvlm-tier ablations
(`rvlm_ocr` 36.56, `rvlm_subagent` 36.72, `rvlm_nocrop` 35.78) sit ~5pp below
the `rvlm` mean but within combined std (rvlm ±5.79) — OCR and sub-call
generalization add nothing over the minimal OCR-free `batch_look`. The
no-recursion baselines collapse to 21–27% and the OCR-only control to 14.7%.

## Observations / caveats

- One pathological science_paper doc (dense dataset table) looped on
  VLM-read disagreements to the 32-iteration cap — dominated wall time.
  Its process died once at 24/25 and was auto-recovered (resumable).

## Status

complete (n=8 of 8)
