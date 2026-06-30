# Dataset axis — MMLongBench-Doc (long-doc stress test)

Cross-benchmark check of the visual-context-budget hypothesis (D-006) on a
**long-document** benchmark. MMLongBench-Doc docs average ~47 pages (the sample
here spans short reports to 80-page guidebooks), vs DocVQA-2026's mostly ≤10-page
docs. Prediction: the recursive-perception method (`rvlm`) holds up on long docs
where a raw-VLM baseline (fixed page budget) cannot see the evidence.

## Setup (current code, 2026-06-29)

- **Model:** Qwen 3.5 27B (lm + vlm), `enable_thinking=false`, local vLLM `:8927`.
- **Sample:** 20 docs, **stratified-random by `doc_type`** (seed 0) via
  `scripts/stratified_sample.py` → `data/mmlongbench-doc/val/sample_doc_ids.txt`
  (Academic paper 4 · Research report 5 · Guidebook 3 · Tutorial 3 · Brochure 2 ·
  Financial 2 · Admin 1). **n=3** (mean ± std; run_ids `mmlb-<S>-t{1,2,3}`). Run
  **one trial at a time** — the 468pg `mmdetection` doc takes ~26min solo but
  1hr+ under concurrent contention (long-doc-concurrency trap).
- **Scoring:** profile scorer = **Qwen judge** (`_mmlb_judge_score`), run on the
  live 27B via `QWEN_JUDGE_BASE_URL=http://localhost:8927/v1`. Verified
  discriminating (scores "Not answerable" / semantic matches correctly; marks
  wrong values wrong).
- **Page budget:** recursive solvers navigate pages via tools (`batch_look`), so
  they're not page-budget-limited. Multi-image baselines need `solver.max_pages`
  raised + a server allowing >1 image/prompt (`--limit-mm-per-prompt {"image":N}`).

Run pattern:
```
QWEN_JUDGE_BASE_URL=http://localhost:8927/v1 uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=<S> data.dataset=yubo2333/MMLongBench-Doc data.use_profile_scoring=true \
  data.doc_ids_file=data/mmlongbench-doc/val/sample_doc_ids.txt \
  data.split=val data.num_samples=null max_concurrency=4 run_id=mmlb-<S>-t1
```

## Results (n=3, judge-scored)

| Solver | role | acc (n=3) | per-trial | Unknown (t1) |
|---|---|---|---|---|
| **`rvlm`** (proposed) | REPL + recursive `batch_look` | **66.6% ± 2.15** | 66.5 / 68.8 / 64.5 | 0% (19 docs) |
| **`codeact_chat`** (proposed twin) | chat-MDP, recursive `batch_look` | **63.8% ± 2.55** | 65.8 / 60.9 / 64.6 | 0% (20 docs) |
| `official_baseline` | MASTER_PROMPT, no scaffold, `max_pages=20` | **53.8% ± 3.58** | 49.7 / 55.9 / 55.9 | **36%** (20 docs) |
| `raw_vlm_multi_baseline` | raw multi-image, no scaffold, `max_pages=20` | **24.2% ± 0.60** | 24.2 / 23.6 / 24.8 | **87%** (20 docs) |

- **`rvlm` 66.6% > `official_baseline` 53.8% (+12.8pp) ≫ `raw_vlm_multi` 24.2%
  (+42.4pp)** on the long-doc subset — the recursive−baseline gap spans
  **~13–42pp**. The baselines are capped by their Unknown rates (36% / 87%) — the
  **page-budget signature**: with `max_pages=20` (downscaled) on ~47-page docs,
  evidence on later pages is unseen, so the model returns "Unknown" (and the 20
  images can overflow the 32k server context outright). `rvlm` navigates the full
  document via `batch_look`, so a 32k-context reasoner answers over docs it could
  never fit in-context (0% Unknown). This is the doc-length axis's load-bearing
  result: the recursive-perception advantage *widens* on long docs (cf.
  DocVQA-2026's mostly ≤10-page docs, where the gaps are smaller). Judge-scored
  (Qwen judge — likely more lenient than the official GPT-4o protocol; the **gap**,
  not the absolute, is the claim); rvlm on 19 docs vs the others on 20 (the 20th, a
  Pew report, reliably crashes rvlm's REPL tail).

- **`codeact_chat` (63.8%, 0% Unknown) ties `rvlm` (66.6%) within combined std**
  on long docs — the proposed twin holds on the doc-length axis exactly as on the
  model axis. Both recursive-navigation methods sit at ~64–67% / 0% Unknown; both
  raw-VLM baselines sit far below with high Unknown. So the split is **tier-level**
  (navigate-the-doc vs fixed-page-budget), not solver-specific. The baselines also
  carry the higher variance (official ±3.58) — they swing with which evidence
  pages happen to fall inside the 20-page budget.

- **Unknown-rate ladder is the cleanest read: 0% (rvlm / codeact_chat) → 36%
  (official) → 87% (raw_vlm_multi).** As page-navigation is removed, the baseline
  increasingly cannot reach later-page evidence and falls back to "Unknown".
  `raw_vlm_multi` (no MASTER_PROMPT scaffold, raw multi-image, page-capped)
  collapses hardest (24.2%, 87% Unknown). This monotone ladder — navigation
  ability ∝ accuracy, inverse ∝ Unknown — is the long-doc instance of the
  perception-budget thesis.

## Summary

On MMLongBench-Doc (long docs, ~47pg avg; 20-doc stratified subset, **n=3**,
judge-scored), the **two recursive-perception methods cluster at ~64–67% / 0%
Unknown** (rvlm 66.6 ± 2.15, codeact_chat 63.8 ± 2.55) while the **raw-VLM
baselines fall to 24–54% with 36–87% Unknown** (official 53.8 ± 3.58, raw_vlm
24.2 ± 0.60). The **~13–42pp gap** is the doc-length instance of the
perception-budget thesis: when the document exceeds the in-context page budget,
the ability to *navigate* it (recursive `batch_look`) is decisive. Main solvers
only (per scope); OCR-extension (`rvlm_ocr`) skipped — it's an ablation and
MMLongBench ships no OCR. The moderate-length axis point is
[`dataset-axis-mp-docvqa.md`](dataset-axis-mp-docvqa.md) (≤20pg), where the gap
narrows to ~2–6pp — the scaling that completes the axis.

> Status: **n=3 complete for all four main solvers** (table above). rvlm trials
> score over 19 docs (the 20th, a Pew report, reliably crashes rvlm's REPL tail);
> cc/official/raw_vlm over 20. Run one trial at a time (long-doc-concurrency trap).
