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
| **`rvlm`** (proposed) | REPL + recursive `batch_look` | **60.8%** (118/194) | 40 docs |
| **`codeact_chat`** (twin) | chat-MDP, recursive | _running_ | |
| `official_baseline` | MASTER_PROMPT, multi-image | _queued_ | |
| `raw_vlm_multi_baseline` | raw multi-image, no scaffold | _queued_ | |

(Main solvers only, per scope — no ablations. The doc-length-axis read is the
rvlm-vs-baseline gap *here* vs on MMLongBench: see `dataset-axis-mmlongbench.md`.)
