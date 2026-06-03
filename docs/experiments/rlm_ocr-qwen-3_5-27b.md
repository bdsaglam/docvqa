# rlm_ocr — Qwen 3.5 27B (val)

## Hypothesis / question

The **text-perception variant** (RLM scaffold + OCR, no vision) — the
control for the OCR-free headline. The proposed
method `rvlm` is OCR-free recursive *visual* perception. The obvious
skeptic question: *could a cheap OCR-text pipeline match it?* This solver
answers exactly that — it holds the scaffold constant (same LeanRLM REPL,
same minimal prompt, same `search` tool) and swaps the perception
**modality** from visual (`batch_look`) to textual (OCR `page_texts` +
BM25 `search`), with **no image access at all**.

Reading of the result:
- `rvlm` >> this ⇒ visual perception does real work OCR text can't
  replace — supports the OCR-free framing.
- they tie ⇒ the visual story weakens; OCR text is sufficient.

Placement vs the rest of the matrix:
- vs `rvlm` (vision, no OCR): isolates perception modality, scaffold held constant.
- vs `rvlm_ocr_ablation` (vision + OCR): this is that solver minus the vision sub-call.

## Setup

- Solver: `rlm_ocr` (LeanRLM REPL + OCR `page_texts` + BM25 `search`; no vision)
- Model: Qwen 3.5 27B local vllm 8927 (lm only; no vlm used), `enable_thinking=false`
- Profile: DocVQA-2026 (default)
- max_concurrency: 8 (per the c=8 setting; aggregate cap MAXCONC=3)
- Added 2026-06-01 as an extra baseline (user request); n=3.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rlm_ocr \
  data.split=val data.num_samples=null \
  max_concurrency=8 \
  run_id=rlm-ocr-cmp-val-tN
```

## Per-trial table

| Trial | run_id | Score | Correct | Wall | Notes |
|---|---|---|---|---|---|
| t1 | `ocr-only-cmp-val-t1` | 12.50% | 10/80 | — | engineering_drawing 0/10, maps 0/10 — OCR extracts nothing for figures/drawings |
| t2 | `ocr-only-cmp-val-t2` | 13.75% | 11/80 | — | engineering_drawing 0/10, maps 0/10 again |
| t3 | `ocr-only-cmp-val-t3` | 13.75% | 11/80 | — | engineering_drawing 0/10, maps 0/10 (third time) |
| t4 | `ocr-only-cmp-val-t4` | 15.00% | 12/80 | — | engineering_drawing 0/10, maps 0/10 (fourth time); comics 3/10 |
| t5 | `ocr-only-cmp-val-t5` | 13.75% | 11/80 | — | engineering_drawing 0/10, maps 0/10, science_poster 0/10 (fifth time) |
| t6 | `ocr-only-cmp-val-t6` | 15.00% | 12/80 | — | engineering_drawing 0/10, maps 0/10 (sixth time); comics 2/10 |
| t7 | `ocr-only-cmp-val-t7` | 11.25% | 9/80 | — | lowest trial; engineering_drawing/maps/science_poster 0/10 (seventh time) |
| t8 | `ocr-only-cmp-val-t8` | 16.25% | 13/80 | — | best trial; engineering_drawing/maps 0/10 (eighth time); business_report 3/10, comics 3/10 |

(run_id prefix is `ocr-only-cmp-val-*`, kept distinct from `rvlm-ocr-cmp-val-*`
to avoid one-letter confusion; solver is `rlm_ocr`.)

Per-category (t1): business_report 2/10, comics 1/10, engineering_drawing
0/10, infographics 2/10, maps 0/10, science_paper 1/10, science_poster
1/10, slide 3/10.

Per-category (t2): business_report 2/10, comics 2/10, engineering_drawing
0/10, infographics 2/10, maps 0/10, science_paper 1/10, science_poster
1/10, slide 3/10.

Per-category (t3): business_report 2/10, comics 1/10, engineering_drawing
0/10, infographics 2/10, maps 0/10, science_paper 2/10, science_poster
1/10, slide 3/10.

## Summary

**n=8 COMPLETE: 13.91% ± 1.56pp** (12.50 / 13.75 / 13.75 / 15.00 / 13.75 / 15.00 / 11.25 / 16.25)
— tight variance. The OCR-text-only control lands **far below** the
OCR-free visual method `rvlm` (39.38% ± 1.49pp): **Δ +25.47pp**. It is the
**lowest solver in the entire matrix** — below the no-scaffold visual
baselines `raw_vlm_multi` (20.47%) and `direct_vlm` (~22%), and below the
competition `official` anchor (17.81%). **engineering_drawing and maps
are 0/10 in all eight trials** (OCR captures none of the figure/drawing
content). **Decisive support for the OCR-free framing:** holding the
LeanRLM scaffold constant and swapping visual perception for OCR text
collapses the score by ~25pp — recursive *visual* perception does work
OCR text cannot replace.

## Comparison

Paired Δ vs `rvlm` at n=3 — the load-bearing control for the OCR-free
claim. Expect weakness on figure/chart/drawing-heavy categories
(engineering_drawing, comics, maps) where OCR extracts little; relative
strength on text-dense categories (business_report, science_paper,
slide) where OCR text is reliable.

## Observations / caveats

- Smoke test (1 doc, engineering_drawing) completed end-to-end and scored
  0/1 — expected for the OCR-weakest category; validated the scoring path.
- OCR-only tends to churn iterations (no visual fallback to resolve
  ambiguity), so wall-time per doc can be high.

## Status

complete (n=8 of 8)
