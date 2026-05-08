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

| Trial | m=10 | m=20 | m=30 (default) | m=40 |
|---|---|---|---|---|
| t1 | 45.00% | 46.25% | 40.00% | 47.50% |
| t2 | 43.75% | 36.25% | 48.75% | 43.75% |
| t3 | 36.25% | 40.00% | 45.00% | 40.00% |
| t4 | 40.00% | 40.00% | 46.25% | 38.75% |
| t5 | 42.50% | 43.75% | 46.25% | 41.25% |
| t6 | 40.00% | 38.75% | 46.25% | 42.50% |
| t7 | 45.00% | 43.75% | 42.50% | 35.00% |
| t8 | 38.75% | 38.75% | 42.50% | 37.50% |

## Summary (n=8 each, all clean — 0 sandbox errors)

| Budget | Mean | Std | Range |
|---|---|---|---|
| m=10 | 41.41% | 3.16pp | 36.2–45.0 |
| m=20 | 40.94% | 3.32pp | 36.2–46.2 |
| **m=30** | **44.69%** | **2.81pp** | 40.0–48.8 |
| m=40 | 40.78% | 3.89pp | 35.0–47.5 |

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
- m=5 not run (queued as defensive lower-end check, low priority — the
  curve already shapes clearly).

## Status

**Done.** Headline: m=30 is the peak at 44.69% ± 2.81pp.
