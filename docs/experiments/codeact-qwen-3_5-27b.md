# codeact — Qwen 3.5 27B (val)

## Hypothesis / question

`codeact` is `rvlm`'s twin — **identical tools, names, and prompt body**
(recursive VLM perception via `batch_look` in a Python REPL) — with one
change: a **strictly append-only** transcript. Every `(reasoning, code,
stdout)` step is appended and shown in full next step; there is no
variable sidecar and no `RESET_HISTORY` compaction. The agent's
observation is its complete history — a clean, fully-observable
trajectory (MDP, not the POMDP that `rvlm`'s LeanRLM compaction creates).
This is the property we want for an **RL fine-tuning target**.

Question: does removing compaction (append-only MDP) cost accuracy vs
`rvlm`'s managed/compacted context, and what iteration budget does the
append-only agent need?

## Setup

- Solver: `codeact` (append-only CodeAct loop; rvlm's tools/prompt/`batch_look`)
- Model: Qwen 3.5 27B local vllm 8927 (lm + vlm), `enable_thinking=false`
- Profile: DocVQA-2026 (default)
- max_concurrency: 8
- **Budget knob:** `solver.max_iterations` (default 40 vs rvlm's 25 — the
  append-only context grows with no compaction, so the agent gets more
  steps before the extract fallback fires). Effective budget =
  max_iterations + page_bonus (same as rvlm).
- Added 2026-06-03 (user request); budget sweep first, then n=8.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=codeact solver.max_iterations=<B> \
  data.split=val data.num_samples=null \
  max_concurrency=8 \
  run_id=codeact-b<B>-val-tN
```

## Budget sweep (n=1 pilots)

| Budget (max_iterations) | run_id | Score | Correct | Notes |
|---|---|---|---|---|
| 24 | `codeact-b24-val-t1` | 37.50% | 30/80 | low budget (≈ rvlm's 25); business_report 6/10, infographics 7/10 |
| 40 | `codeact-b40-val-t1` | 33.75% | 27/80 | default; lowest pilot (likely noise — comics 3, eng_drawing 3, slide 3) |
| 56 | `codeact-b56-val-t1` | 40.00% | 32/80 | high budget; best pilot (comics 4, science_paper 4, slide 5) |

**n=2 confirm** (b24 vs b56, to disambiguate the top two before n=8):

| Budget | run_id | Score | n=2 mean |
|---|---|---|---|
| 24 | `codeact-b24-val-t2` | 43.75% | **40.6%** (37.5, 43.75) |
| 56 | `codeact-b56-val-t2` | 33.75% | **36.9%** (40.0, 33.75) |

**Verdict: budget is noise-dominated.** The n=2 confirm *flipped* the n=1
ranking (n=1: b56 40.0 > b24 37.5; n=2: b24 40.6 > b56 36.9), and the two
budgets' trials interleave completely (b24: 37.5/43.75; b56: 33.75/40.0).
The 5 pilots across all budgets average **37.75%** — squarely in the
visual-recursive tier (≈ `rvlm_ocr` 37.81, `rvlm` 39.38). Efficiency
(`iter_stats.py`) confirms the cap never binds (≤1% @cap at any budget),
so a lower budget loses nothing. **Chosen budget for n=8: 24** — slightly
ahead at n=2, cheapest/fastest, never caps, and the cleanest narrative
("append-only CodeAct matches `rvlm` even at `rvlm`'s own ~25-step
budget"). The 2 b24 pilots become t1, t2 of the n=8 run.

Per-category (b24): business_report 6/10, comics 4/10, engineering_drawing
3/10, infographics 7/10, maps 0/10, science_paper 2/10, science_poster
3/10, slide 5/10.

Per-category (b56): business_report 5/10, comics 4/10, engineering_drawing
4/10, infographics 6/10, maps 0/10, science_paper 4/10, science_poster
4/10, slide 5/10.

**Read:** 37.5 / 33.75 / 40.0 — a 6.25pp spread that is **non-monotonic**
(default-40 is the *worst*), i.e. n=1 noise (~3–5pp/trial) dominates and
budget is not a strong lever in [24, 56]. All three sit in the
visual-recursive tier (≈ rvlm 39.4 / rvlm_ocr 37.8): **append-only
CodeAct does not need extra budget and does not lose accuracy vs the
compacted rvlm.** Top two (b56 40.0, b24 37.5) are 2.5pp apart — within
noise — so the n=8 budget pick is essentially free; leaning b56 (most
headroom for the append-only context; scored best) pending the human's
call.

## n=8 per budget (in progress — running all 3 budgets, the pilots are t1/t2)

Given the high trial variance, all three budgets are taken to n=8 so
"budget" is a proper axis with error bars rather than a single noisy pick.

**budget 24** (`codeact-b24-val-t*`)
| t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 |
|---|---|---|---|---|---|---|---|
| 37.50 | 43.75 | 40.00 | 28.75 | 43.75 | 33.75 | 37.50 | — |

**budget 40** (`codeact-b40-val-t*`)
| t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 |
|---|---|---|---|---|---|---|---|
| 33.75 | 43.75 | 40.00 | 32.50 | 36.25 | 42.50 | — | — |

**budget 56** (`codeact-b56-val-t*`)
| t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 |
|---|---|---|---|---|---|---|---|
| 40.00 | 33.75 | 35.00 | 36.25 | 35.00 | 36.25 | — | — |

## Summary

Budget sweep in progress (b40, b56 pending). Early read: **b24 = 37.5%**
— even at the low rvlm-like budget, append-only CodeAct is in the
visual-recursive tier (≈ `rvlm` 39.4 / `rvlm_ocr` 37.8), i.e. dropping
compaction does **not** collapse accuracy. Final budget pick + n=8 mean
± std pending the other two pilots.

## Comparison

vs `rvlm` (39.38% ± 1.49pp, n=8) — same scaffold/tools, the only
difference is append-only (MDP) vs compacted (POMDP) context. A small or
zero gap means the append-only trajectory (the RL-friendly form) is
nearly free; a large gap means compaction is doing real work.

## Status

budget sweep complete (n=1: b24 37.5 / b40 33.75 / b56 40.0); awaiting budget pick for n=8
