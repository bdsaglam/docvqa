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
| t1 | 31.25% | 43.75% | 48.75% | 38.75% |
| t2 | 42.50% | 40.00% | 50.00% | 47.50% |
| t3 | 38.75% | 40.00% | 41.25% | 46.25% |
| t4 | 40.00% | 47.50% | 42.50% | 38.75% |
| t5 | 48.75% | 48.75% | 36.25% | 33.75% |
| t6 | 42.50% | 36.25% | 45.00% | 41.25% |
| t7 | 40.00% | 35.00% | 42.50% | 42.50% |
| t8 | 40.00% | 48.75% | 43.75% | 42.50% |

## Summary

| Budget | Mean | Std | n | Range |
|---|---|---|---|---|
| m=25 | 40.47% | 4.86pp | 8 | 31.2–48.8 |
| m=30 | 42.50% | 5.51pp | 8 | 35.0–48.8 |
| m=40 | **43.75%** | 4.33pp | 8 | 36.2–50.0 |
| m=50 | 41.41% | 4.40pp | 8 | 33.8–47.5 |

## Comparison

- **m=40 is the peak** at 43.75% ± 4.33pp.
- **m=40 vs m=25:** +3.28pp, SE = 2.30pp → t = 1.43 (marginal).
- **m=40 vs m=50:** +2.34pp, SE = 2.16pp → t = 1.08 (n.s. but a real
  drop in mean and similar std).
- **m=40 vs flat_solo m=30 (44.69 ± 2.81):** −0.94pp, SE = 1.82 →
  t = 0.52 — leanest at peak budget is **statistically
  indistinguishable** from flat_solo at peak budget on the mean.
- **Variance comparison:** all leanest cells (4.33–5.51pp) have
  meaningfully higher std than flat_solo m=30 (2.81pp). Across the
  pooled 32 leanest trials, std is ~4.8pp vs flat_solo's 2.81pp on
  n=8. Variance reduction by OCR is the more robust finding than
  any mean lift.

## Observations

- **Same non-monotonic shape as flat_solo, shifted right by 10.**
  Flat solo peaks at m=30; leanest peaks at m=40. Both curves drop
  on either side of the peak. Interpretation: leanest needs more
  turns because it lacks the symbolic shortcut, but past the peak
  long trajectories drift / accumulate errors / waste context — same
  failure mode in both solvers.
- **Leanest variance is consistently higher** across all four cells
  (std 4.33–5.51pp vs flat_solo's 2.81pp at m=30). Without OCR as a
  deterministic anchor, runs diverge more across seeds.
- **m=50 verdict: NOT an improvement.** Mean drops 2.34pp from m=40
  with similar std. Higher budget alone doesn't compensate for
  missing OCR.
- m=25 had a low outlier (31.25% on t1) that drags the mean; m=30
  and m=50 also had low outliers (35.0%, 33.75%). High-tail trials
  are common — variance is the dominant story.
- Long-tail docs (especially `science_paper_1`, 7 questions, and
  `maps_2` for spatial reasoning) routinely take 50+ min on leanest
  at high budget. Stall detector often flags these — false positives,
  runs are still progressing.

## Status

**Done.** Headline: leanest peaks at m=40 (43.75% ± 4.33pp), close
to flat_solo's peak at m=30 (44.69% ± 2.81pp) on the mean but with
~50% higher std. OCR's contribution looks more like variance
reduction than capability lift.
