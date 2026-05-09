# Flat Solo — search tool off

**Hypothesis:** the BM25 `search()` tool contributes meaningful
retrieval beyond what the agent gets from having `page_texts` directly
in scope. Removing only `search()` (keeping all OCR text in scope, all
VLM tools, cropping, tips, m=30) isolates "BM25 retrieval" from the
broader OCR-channel question.

**Setup:** Flat Solo solver, lean RLM, no thinking, default m=30,
cropping on, tips on. Qwen 3.5 27B (vLLM local 8927) for both LM and
VLM. Val 80q. `solver.use_search=false`. 8 trials.

## Implementation note

Added 2026-05-08: `solver.use_search` flag in `flat_solo_solver.py`.
When `False`:
- `search(query, k)` line stripped from `TASK_INSTRUCTIONS` via
  `_strip_search_tool()`.
- `def search()` wrapper omitted from sandbox code.
- `_search` excluded from tools list passed to RLM.

`page_texts` (the raw OCR text per page) remains in scope so the agent
can scan it manually. This is the "BM25 retrieval ablation", not an
"OCR channel ablation".

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local \
  vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=flat_solo \
  solver.rlm_type=lean \
  solver.use_search=false \
  data.split=val data.num_samples=null \
  max_concurrency=24 \
  run_id=flat-solo-no-search-3_5-27b-val-t${i}
```

## Per-trial scores

| Trial | Score | Sandbox errors |
|---|---|---|
| t1 | _running_ | — |
| t2 | — | — |
| t3 | — | — |
| t4 | — | — |
| t5 | — | — |
| t6 | — | — |
| t7 | — | — |
| t8 | — | — |

## Summary

(awaiting trials)

## Comparison to baseline

- **Baseline (flat_solo full, search ON):** 44.69% ± 2.81pp (n=8)
- Reference: leanest m=40 (no OCR at all) is 43.75% ± 4.33pp.
- Search-off should sit somewhere between leanest (no OCR) and
  flat_solo full. If close to baseline → the search tool is mostly
  redundant given page_texts already in scope. If close to leanest
  → BM25 retrieval is doing real work.

## Status

**In progress.** t1 running on Host A's local 8927 lane.
