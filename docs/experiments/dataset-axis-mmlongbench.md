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
  Financial 2 · Admin 1). n=1 (exploratory, D-008).
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

## Results (n=1, judge-scored)

| Solver | role | acc | notes |
|---|---|---|---|
| **`rvlm`** (proposed) | REPL + recursive `batch_look` | **66.5%** (103/155) | 19 docs, 0% Unknown |
| `official_baseline` | MASTER_PROMPT, no scaffold, `max_pages=20` | **49.7%** (80/161) | 20 docs, **36% Unknown** |
| `raw_vlm_multi_baseline` | raw multi-image, no scaffold | _running_ | |
| `codeact_chat` | chat-MDP twin | _queued (heavy)_ | |
| `rvlm_ocr_ablation` | + OCR/search | _queued (heavy)_ | |

- **`rvlm` 66.5% > `official_baseline` 49.7% (≈ +17pp)** on the long-doc subset.
  The baseline is capped by **36% Unknown** — the **page-budget signature**: with
  `max_pages=20` (downscaled) on ~47-page docs, evidence on later pages is unseen,
  so the model returns "Unknown". `rvlm` navigates the full document via
  `batch_look`, so a 32k-context reasoner answers over docs it could never fit
  in-context (0% Unknown). This is the doc-length axis's load-bearing result: the
  recursive-perception advantage *widens* on long docs (cf. DocVQA-2026's mostly
  ≤10-page docs, where `rvlm`≈OCR/baseline gaps are smaller). n=1, judge-scored;
  rvlm on 19 docs vs official on 20 (the 20th, a Pew report, crashed rvlm's tail).
  Absolute values use the Qwen judge (likely more lenient than the official GPT-4o
  protocol) — the **gap**, not the absolute, is the claim.

> Status: rvlm done (19/20; 20th doc — a Pew report — repeatedly crashed the
> process at the tail, dropped for the n=1 read). Baselines pending a 27B restart
> with `--limit-mm-per-prompt {"image":32}`. MP-DocVQA (moderate-length axis
> point) deprioritized — streaming its image-laden val parquet timed out; needs
> a real download.
