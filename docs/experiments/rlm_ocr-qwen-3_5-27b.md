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
| — | (queued) | — | — | — | n=3 queued behind the official baseline cells |

## Summary

Queued (n=0). Mean ± std at n=8.

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

queued (n=0 of 8)
