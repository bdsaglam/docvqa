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

| Trial | Score | Sandbox errors | Notes |
|---|---|---|---|
| t1 | 42.50% | 0 | clean. Agent manually scans `page_texts` with `re.search()` (regex). 8 Unknown. |
| t2 | 41.25% | 0 | clean |
| t3 | 48.75% | 0 | clean (high outlier) |
| t4 | 46.25% | 0 | clean |
| t5 | 40.00% | 0 | clean |
| t6 | 37.50% | 0 | clean (low) |
| t7 | 38.75% | 0 | clean (low) |
| t8 | 45.00% | 0 | clean |

## Summary (n=8, all clean — 0 sandbox errors)

- per-trial: [42.5, 41.2, 48.8, 46.2, 40.0, 37.5, 38.8, 45.0]
- mean = **42.50% ± 3.90pp**
- range 37.5–48.8

## Comparison to baseline

- **Baseline (flat_solo full, search ON):** 44.69% ± 2.81pp (n=8).
- **Gap: −2.19pp**, SE on the difference = √(3.90²/8 + 2.81²/8) = 1.70pp
- **t-stat: 2.19 / 1.70 ≈ 1.29 → NOT significant.**
- Reference: leanest m=40 (no OCR at all): 43.75 ± 4.33pp; leanest
  m=25 default: 40.47 ± 4.86pp.

## Observations

- **BM25 search is largely redundant on this dataset.** Removing it
  while keeping `page_texts` in scope doesn't move the mean
  significantly. The agent compensates by manually grepping with
  `re.search(...)` over `page_texts` (visible in trajectories).
- **Variance bumps up modestly** (2.81 → 3.90 pp std). Not as dramatic
  as the OCR-off comparison (leanest), but suggests the BM25 tool was
  acting as a small stability anchor — when present, it pulls the
  agent toward consistent retrieval starting points.
- Implication for the paper: this is a **"scaffold component that
  didn't matter much"** finding. Worth reporting honestly — RLM
  paper-style ablations include "X surprisingly didn't help" too.
- Related ablation: leanest (no OCR at all) gets 40.47–43.75% ±
  ~4.5pp. Going from leanest → search-off (adding `page_texts` but
  not BM25) → full (adding BM25) gives an OCR contribution gradient:
  - **leanest → search-off: +2.0pp** (OCR text in scope adds modest lift)
  - **search-off → full: +2.2pp** (BM25 retrieval adds another modest lift)
  Neither step is individually significant, but together they're
  ~+4pp. The text-being-in-scope and the BM25 retrieval each
  contribute roughly equally to the small OCR effect.

## Status

**Done.** Headline: BM25 search ablation is statistically n.s.
(−2.19pp, t=1.29). Paper framing: BM25 is largely redundant on
DocVQA-2026 given OCR text in agent scope.
