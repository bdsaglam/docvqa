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

## Efficiency (turns per question)

Pooled across 8 trials × 80q per cell.

| Budget | turns mean ± std | median | p90 | max | turns_correct | turns_wrong | wrong/correct |
|---|---|---|---|---|---|---|---|
| m=25 (default) | 12.82 ± 7.18 | 11 | 24 | 35 | 11.54 | 13.69 | 1.19 |
| m=30 | 13.29 ± 7.65 | 11 | 25 | 40 | 11.51 | 14.61 | 1.27 |
| **m=40 (peak)** | 13.97 ± 9.08 | 11 | 27 | 50 | 11.89 | 15.59 | 1.31 |
| m=50 | 14.21 ± 9.81 | 11 | 28 | 60 | 12.11 | 15.73 | 1.30 |

Cross-comparison: at the same nominal budget (m=30), leanest uses
**13.29** vs flat_solo's **13.19** turns/question — essentially
identical. So the OCR channel isn't saving the agent turns; the two
solvers do the same volume of work, just in different channels (OCR
+ BM25 vs more `look()` calls).

Other patterns:

- **Median is flat at 11 across all four leanest cells.** The
  budget mostly affects the long tail (p90 climbs 24 → 28; max
  climbs 35 → 60). Increasing the cap mostly buys longer wrong
  trajectories, not more correct ones.
- **Correct trajectories barely lengthen** (11.54 → 12.11) while
  wrong trajectories grow much faster (13.69 → 15.73). Thrash
  ratio rises 1.19 → 1.31. Same shape as the flat_solo budget
  sweep — extra turns mostly go to the agent thrashing on
  unsolvable questions.
- m=40 being the accuracy peak while m=50 has higher mean turns
  matches the "headroom past the peak is wasted" reading from the
  flat_solo sweep.

## Status

**Done.** Headline: leanest peaks at m=40 (43.75% ± 4.33pp), close
to flat_solo's peak at m=30 (44.69% ± 2.81pp) on the mean but with
~50% higher std. OCR's contribution looks more like variance
reduction than capability lift.
