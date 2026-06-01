# rvlm_ocr_ablation — Qwen 3.5 27B (val)

## Hypothesis / question

Does adding OCR + BM25 search to the OCR-free `rvlm` method help on
DocVQA-2026? This is the OCR-extension ablation (D-006 paper framing:
OCR/search as an *extension* of the proposed OCR-free RLM). Compared
head-to-head with `rvlm` under the post-cleanup code (minimized prompts,
parity-stripped, per-call `num_retries=5` only).

## Setup

- Solver: `rvlm_ocr_ablation` (OCR-free RLM + `look()`/OCR + BM25 search)
- Model: Qwen 3.5 27B local vllm 8927 (lm + vlm), `enable_thinking=false`
- Profile: DocVQA-2026 (default)
- max_concurrency: 24
- Part of the 7-solver comparison re-run (val, n=3), 2026-06-01.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_ocr_ablation \
  data.split=val data.num_samples=null \
  max_concurrency=24 \
  run_id=rvlm-ocr-cmp-val-tN
```

## Per-trial table

| Trial | run_id | Score | Correct | Wall | Notes |
|---|---|---|---|---|---|
| t1 | `rvlm-ocr-cmp-val-t1` | 36.25% | 29/80 | ~1h14m | maps 0/10, science_paper 1/10 |
| t2 | `rvlm-ocr-cmp-val-t2` | 35.00% | 28/80 | — | maps 0/10, eng_drawing 3/10 |
| t3 | `rvlm-ocr-cmp-val-t3` | 40.00% | 32/80 | — | business_report 7/10, maps 3/10 (best trial) |

Per-category (t1): business_report 50%, comics 40%, engineering_drawing
50%, infographics 50%, maps 0%, science_paper 10%, science_poster 40%,
slide 50%.

Per-category (t2): business_report 5/10, comics 2/10, engineering_drawing
3/10, infographics 6/10, maps 0/10, science_paper 3/10, science_poster
4/10, slide 5/10.

Per-category (t3): business_report 7/10, comics 3/10, engineering_drawing
3/10, infographics 5/10, maps 3/10, science_paper 2/10, science_poster
3/10, slide 6/10.

## Summary

**n=3 so far (target n=8): 37.08% ± 2.60pp** (t1 36.25%, t2 35.00%, t3 40.00%). maps
volatile (0/0/3). vs `rvlm` (39.4% mean at n=2) — OCR adds nothing over
the OCR-free method, trending slightly below. Lock paired Δ once `rvlm`
reaches n=3.

## Comparison

Compare against `rvlm` (same matrix, `rvlm-qwen-3_5-27b.md`). Δ computed
once both reach n=3.

## Observations / caveats

- Builds BM25 indexes per doc (expected for the OCR extension).

## Status

in progress (n=3 of 8)
