# Dataset axis — MP-DocVQA (moderate-length)

The moderate-length point of the document-length axis (D-006). MP-DocVQA docs
are **≤20 pages (mean 5.3)** — between DocVQA-2026 (mostly ≤10pg) and
MMLongBench-Doc (~47pg avg). Prediction: the recursive-perception advantage over
raw-VLM baselines is **smaller here than on MMLongBench**, because a ≤20-page doc
largely fits a raised page budget — so the baselines can see most evidence.

## Setup (current code, 2026-06-29)

- **Model:** Qwen 3.5 27B (lm + vlm), `enable_thinking=false`, local vLLM `:8927`
  (`--limit-mm-per-prompt {"image":32}` so multi-image baselines run).
- **Sample:** 40 docs, **stratified-random by page-count bin** (seed 0) via
  `scripts/stratified_sample.py` → `data/mp-docvqa/val/sample_doc_ids.txt`
  (1-2pg 20 · 3-5pg 8 · 6-10pg 5 · 11-20pg 7, proportional to the 927-doc val).
  n=1 (exploratory, D-008).
- **Scoring:** **ANLS** (MP-DocVQA's metric; the profile uses the default ANLS
  scorer — no judge). `data.use_profile_scoring=true` for the MP-DocVQA profile
  prompt/formatting.
- ANLS is character-level and strict on format ('3' vs gold 'three' scores ~0);
  this hits all solvers equally, so the **relative** ranking is what's load-bearing.

Run pattern:
```
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false solver=<S> data.dataset=lmms-lab/MP-DocVQA \
  data.use_profile_scoring=true data.doc_ids_file=data/mp-docvqa/val/sample_doc_ids.txt \
  data.split=val data.num_samples=null max_concurrency=4 run_id=mpdoc-<S>-t1
```

## Results (n=1, ANLS, 40-doc stratified subset)

| Solver | role | ANLS | notes |
|---|---|---|---|
| **`codeact_chat`** (twin) | chat-MDP, recursive | **64.4%** (125/194) | 40 docs |
| **`rvlm`** (proposed) | REPL + recursive `batch_look` | **60.8%** (118/194) | 40 docs |
| `official_baseline` | MASTER_PROMPT, multi-image, `max_pages=20` | **58.8%** (114/194) | 8% Unknown |
| `raw_vlm_multi_baseline` | raw multi-image, no scaffold, `max_pages=20` | **58.2%** (113/194) | 22% Unknown |

(Main solvers only, per scope — no ablations.)

## Read — the gap nearly vanishes on moderate docs

All four solvers cluster at **58–64% ANLS**: the recursive methods
(codeact_chat 64.4, rvlm 60.8) lead the baselines (official 58.8, raw_vlm 58.2)
by only **~2–6pp** — and the baselines' Unknown rates are low (8% / 22%). On
≤20-page docs the raw-VLM baseline can fit most of the document in its page
budget, so it *sees the evidence* and the navigation advantage is small.

This is the **moderate-length contrast that completes the document-length axis**.
Against MMLongBench-Doc (~47pg; `dataset-axis-mmlongbench.md`):

| | MP-DocVQA (≤20pg) | MMLongBench-Doc (~47pg) |
|---|---|---|
| codeact_chat | 64.4% | 65.8% |
| rvlm | 60.8% | 66.5% |
| official_baseline | 58.8% (8% Unk) | 49.7% (36% Unk) |
| raw_vlm_multi | 58.2% (22% Unk) | 24.2% (87% Unk) |
| **recursive − baseline gap** | **~2–6pp** | **~16–42pp** |

**The recursive-perception advantage scales with document length** (D-006). The
recursive methods are flat across the axis (~61–66%, Unknown ≈ 0%) — they
navigate the doc regardless of length. The baselines *degrade with length* as the
fixed page budget increasingly misses evidence (Unknown 8/22% → 36/87%). When the
document fits the budget the scaffold buys little; when it doesn't, recursive
navigation is decisive. (n=1 exploratory, stratified subsets, ANLS vs Qwen-judge
across the two datasets — the **within-dataset gaps and their scaling**, not the
cross-dataset absolutes, are the claim.)
