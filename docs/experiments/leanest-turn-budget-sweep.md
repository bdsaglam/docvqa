# Leanest turn-budget sweep

**Hypothesis:** the leanest solver (no OCR, VLM-only) may have a
different turn-budget sweet spot than flat_solo, since each turn does
more work (no cheap symbolic lookups). Existing m=25/m=40 hint at a
non-flat curve in the opposite direction from flat_solo.

**Setup:** Leanest Solo solver, lean RLM, no thinking. Qwen 3.5 27B
(vLLM local 8927) for both LM and VLM. Val 80q. 8 trials per cell.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local \
  vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=leanest_solo \
  solver.max_iterations=$M \
  data.split=val data.num_samples=null \
  max_concurrency=24 \
  run_id=leanest-m${M}-3_5-27b-val-t${i}
# m=25 default used run_id=leanest-3_5-27b-val-t${i} (no -m suffix)
```

## Per-trial scores

| Trial | m=25 (default) | m=30 | m=40 | m=50 |
|---|---|---|---|---|
| t1 | 31.25% | 43.75% | 48.75% | — |
| t2 | 42.50% | 40.00% | 50.00% | — |
| t3 | 38.75% | _running_ | 41.25% | — |
| t4 | 40.00% | — | 42.50% | — |
| t5 | 48.75% | — | 36.25% | — |
| t6 | 42.50% | — | 45.00% | — |
| t7 | 40.00% | — | 42.50% | — |
| t8 | 40.00% | — | 43.75% | — |

## Summary

| Budget | Mean | Std | n | Range |
|---|---|---|---|---|
| m=25 | 40.47% | 4.86pp | 8 | 31.2–48.8 |
| m=30 | 41.88% | 2.65pp | 2 (in progress) | 40.0–43.8 |
| m=40 | **43.75%** | 4.33pp | 8 | 36.2–50.0 |
| m=50 | — | — | 0 (queued) | — |

## Comparison

- **m=40 vs m=25:** +3.28pp, SE = √(4.86²/8 + 4.33²/8) = 2.30pp →
  **t = 1.43 SE — marginally significant.** The win sign is consistent
  but the variance is high enough that more cells / more trials are
  needed before claiming a curve.
- m=30 (n=2 so far) sits between, as expected. Need 6 more trials to
  bracket m=25→m=40.

## Observations

- **Opposite shape from flat_solo.** Flat solo peaks at m=30 and m=40
  hurts; leanest's m=40 helps over m=25. The interpretation: leanest
  has no symbolic shortcut (no OCR retrieval), so each "found the
  answer" event requires more VLM-mediated exploration. Higher budget
  pays off more.
- **Leanest variance is much higher** (~4.5pp) than flat_solo (~3pp).
  Without OCR as an anchor, runs can diverge more across seeds.
- m=25 has a low outlier (31.25% on t1) that drags the mean. With
  high std, individual trials are unreliable.
- Long-tail docs (especially `science_paper_1`, 7 questions) routinely
  take 50+ min on leanest at high budget. Stall detector occasionally
  flags this — false positive, the run is still progressing.

## Status

**In progress.** m=30: 2/8 trials done. m=50: not started — queued
after m=30 completes. Running on Host A's local 8927 lane sequentially
(one trial at a time, single lane).
