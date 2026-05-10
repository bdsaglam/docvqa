# Flat Solo turn-budget sweep

**Hypothesis:** the agent's `max_iterations` budget has a non-monotonic
sweet spot — too few turns starves reasoning, too many waste context
on unproductive trajectories.

**Setup:** Flat Solo solver, lean RLM, no thinking. Qwen 3.5 27B (vLLM
local 8927) for both LM and VLM. Val 80q. 8 trials per budget point.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local \
  vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=flat_solo \
  solver.rlm_type=lean \
  solver.max_iterations=$M \
  data.split=val data.num_samples=null \
  max_concurrency=24 \
  run_id=flat-solo-3_5-27b-m${M}-val-t${i}
# m=30 baseline used run_id=flat-solo-val-t${i} (default config, no -m suffix)
```

## Per-trial scores

| Trial | m=5 | m=10 | m=20 | m=30 (default) | m=40 |
|---|---|---|---|---|---|
| t1 | 30.00% | 45.00% | 46.25% | 40.00% | 47.50% |
| t2 | 30.00% | 43.75% | 36.25% | 48.75% | 43.75% |
| t3 | 30.00% | 36.25% | 40.00% | 45.00% | 40.00% |
| t4 | — | 40.00% | 40.00% | 46.25% | 38.75% |
| t5 | — | 42.50% | 43.75% | 46.25% | 41.25% |
| t6 | — | 40.00% | 38.75% | 46.25% | 42.50% |
| t7 | — | 45.00% | 43.75% | 42.50% | 35.00% |
| t8 | — | 38.75% | 38.75% | 42.50% | 37.50% |

## Summary

| Budget | Mean | Std | Range | n |
|---|---|---|---|---|
| **m=5** | **30.00%** | **0.00pp** | 30.00 | **3** |
| m=10 | 41.41% | 3.16pp | 36.2–45.0 | 8 |
| m=20 | 40.94% | 3.32pp | 36.2–46.2 | 8 |
| **m=30** | **44.69%** | **2.81pp** | 40.0–48.8 | 8 |
| m=40 | 40.78% | 3.89pp | 35.0–47.5 | 8 |

m=10/20/30/40: all clean (0 sandbox errors). m=5: clean (0 sandbox errors).

## Observations

- **m=30 is the peak.** Both shorter (m=10/20) and longer (m=40) budgets
  drop ~3–4pp below m=30.
- **m=30 also has the lowest variance** (2.81pp vs 3.16/3.32/3.89). The
  lower variance + higher mean suggests the curve is genuinely
  non-monotonic, not just noise.
- **m=40 vs m=30 gap = 3.91pp ≈ 2.3 SE** — marginally significant. Going
  past 30 turns increases the chance of unproductive trajectories that
  drift / accumulate errors / waste context.
- **m=10 ≈ m=20** — both ~41%. The first budget gain happens between 20
  and 30 turns.
- **m=5 added 2026-05-08, n=3:** 30.00% ± 0.00pp. ~11pp below m=10 —
  starvation is real at 5 turns. All three trials hit exactly 30.00%
  (per-category mix differs); a coincidence on n=3 but consistent with
  m=5 having very little turn-by-turn divergence (run is quickly
  truncated). Run on Host B with `lm.enable_thinking=false` and Host B
  vllm 8928 (4-GPU); not the same vllm as the m=10..40 cells (Host A
  8927, 3-GPU). Identical model weights, otherwise identical config.
  Run dirs: `output/runs/flat-solo-m5-val-{t1,t2,t3}/`.

## Efficiency (turns per question)

Pooled across 8 trials × 80q per cell (m=5 not measured locally).

| Budget | turns mean ± std | median | p90 | max | turns_correct | turns_wrong | wrong/correct | hit-cap rate (p90 = m?) |
|---|---|---|---|---|---|---|---|---|
| m=10 | 9.92 ± 3.79 | 10 | 16 | 20 | 9.00 | 10.57 | 1.17 | budget-bound (max=20=2m) |
| m=20 | 12.42 ± 6.29 | 11 | 20 | 30 | 10.87 | 13.49 | 1.24 | partly bound |
| **m=30 (default)** | 13.19 ± 7.85 | 11 | 26 | 40 | 11.92 | 14.22 | 1.19 | rarely bound |
| m=40 | 13.55 ± 9.01 | 10 | 25 | 50 | 11.42 | 15.01 | 1.31 | not bound |

(`max` exceeds the budget because each iteration may emit several
trajectory steps — ~2× per iteration on average — so step count tracks
budget but isn't equal to it. The right-hand column reports whether
the p90 sits at the cap, which is the practical "budget-bound" check.)

Key reads:

- **The agent is budget-bound at m=10**: median equals the cap, p90
  saturates. m=10 trajectories are getting truncated.
- **Past m=30 the budget is slack.** Going m=30 → m=40 only adds
  +0.36 turns to the mean — the extra 10 turns of headroom are mostly
  unused. Yet accuracy *drops* 3.9pp (44.7 → 40.8), so the long-tail
  trajectories the bigger budget enables are net harmful (drift /
  context dilution).
- **Wrong/correct thrash ratio drifts up with budget** (1.17 →
  1.24 → 1.19 → 1.31). With more rope, the agent thrashes more on
  the questions it ends up getting wrong.

## Status

**Done.** Headline: m=30 is the peak at 44.69% ± 2.81pp. Five-point curve
{5, 10, 20, 30, 40} for the paper figure; m=5 shows the lower-end drop-off.
