# react_baseline — Qwen 3.5 27B (val)

## Hypothesis / question

The no-REPL baseline: `dspy.ReAct` with the same VLM perception tools as
`rvlm` but **no Python REPL / code execution**. Isolates whether the
LeanRLM code-REPL is load-bearing — react keeps perception, drops the
programmatic scaffold. Complement of `repl_only_baseline` (keeps REPL,
drops perception). Minimized prompt (parity with `rvlm`).

## Setup

- Solver: `react_baseline` (ReAct + VLM tools, no REPL)
- Model: Qwen 3.5 27B local vllm 8927 (lm + vlm), `enable_thinking=false`
- Profile: DocVQA-2026 (default)
- max_concurrency: 24
- Part of the 7-solver comparison re-run (val, n=3), 2026-06-01.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=react_baseline \
  data.split=val data.num_samples=null \
  max_concurrency=24 \
  run_id=react-cmp-val-tN
```

## Per-trial table

| Trial | run_id | Score | Correct | Wall | Notes |
|---|---|---|---|---|---|
| t1 | `react-cmp-val-t1` | 17.50% | 14/80 | — | science_paper 0/10, maps 0/10 |
| t2 | `react-cmp-val-t2` | 30.00% | 24/80 | — | +12.5pp over t1 — high variance |

Per-category (t1): business_report 1/10, comics 2/10, engineering_drawing
2/10, infographics 4/10, maps 0/10, science_paper 0/10, science_poster
2/10, slide 3/10.

Per-category (t2): business_report 2/10, comics 3/10, engineering_drawing
4/10, infographics 4/10, maps 2/10, science_paper 2/10, science_poster
2/10, slide 5/10.

## Summary

n=2 so far (t3 queued). t1 17.50%, t2 30.00% → running mean ~23.8% with
large spread (±~9pp) — the no-REPL react baseline is high-variance; t3
matters for a stable mean. Mean ± std at n=3.

## Comparison

`rvlm` t1 (40.00%) − this (17.50%) = **+22.50pp** — the REPL is
load-bearing (consistent with the prior n=8 react ablation, Δ ≈ −10.5pp
vs rvlm; this re-run's n=1 gap is larger but same direction — lock at
n=3). Note react (perception, no REPL) 17.50% vs repl_only (REPL, no
perception) 7.50%: with only one of the two, perception buys more than
the bare REPL, but `rvlm` needs **both**.

## Observations / caveats

- (none yet)

## Status

in progress (n=1 of 3)
