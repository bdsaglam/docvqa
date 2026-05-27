# Flat Solo — VLM cropping ablation (D-004)

**Hypothesis:** the active-perception channel (VLM accepts arbitrary
PIL crops, not just whole pages) contributes meaningfully. Restricting
the `look()` tool to whole-page images by index isolates the cropping
contribution from the broader VLM-on/off question.

**Setup:** Flat Solo solver, lean RLM, no thinking, default m=30.
Qwen 3.5 27B (vLLM remote 8928, 4-GPU) for both LM and VLM. Val 80q.
`solver.vlm_cropping=false` (uses `TASK_INSTRUCTIONS_PAGE_ONLY` and
`look(page_idx, query)` instead of `look(image, query)`).
8 trials. Concurrency=32 (remote 4-GPU has more headroom than local).

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-remote \
  vlm=qwen-3_5-27b-vllm-remote \
  lm.enable_thinking=false \
  solver=flat_solo \
  solver.rlm_type=lean \
  solver.vlm_cropping=false \
  data.split=val data.num_samples=null \
  max_concurrency=32 \
  run_id=flat-solo-page-only-3_5-27b-val-t${i}
```

## Per-trial scores

| Trial | Score | Sandbox errors |
|---|---|---|
| t1 | 35.00% | 0 |
| t2 | 32.50% | 0 |
| t3 | 40.00% | 0 |
| t4 | 37.50% | 0 |
| t5 | 40.00% | 0 |
| t6 | 36.25% | 0 |
| t7 | 36.25% | 0 |
| t8 | 37.50% | 0 |

## Summary (n=8, all clean)

- mean = **36.88% ± 2.50pp**
- range 32.5–40.0
- per-trial: [35.0, 32.5, 40.0, 37.5, 40.0, 36.2, 36.2, 37.5]

## Comparison to baseline

- **Baseline (cropping enabled):** 44.69% ± 2.81pp (n=8) — see
  `flat-solo-turn-budget-sweep.md` m=30 column.
- **Gap: −7.81pp**, SE on the difference = √(2.50²/8 + 2.81²/8) = 1.33pp
- **t-stat: 7.81 / 1.33 ≈ 5.88 → highly significant.**

## Observations

- Cropping is the **largest single ablation effect** measured so far on
  Flat Solo (−7.81pp), beating tips (−5.94pp).
- Variance is the **lowest** of any cell tested (2.50pp). The page-only
  variant is a more deterministic regime: fewer degrees of freedom in
  what the VLM gets to look at.
- This validates the "active perception" framing for the paper —
  the agent's ability to crop and zoom contributes ~8pp on top of
  whole-page-only VLM access.
- **D-004 ablation in `decisions.md`** is now backed by an 8-trial run.

## Efficiency (turns per question)

Pooled across 8 trials × 80q (n=633 — a few questions failed before
producing a trajectory).

| Cell | turns mean ± std | median | p90 | max | turns_correct | turns_wrong | wrong/correct |
|---|---|---|---|---|---|---|---|
| flat_solo m=30 (baseline, cropping ON) | 13.19 ± 7.85 | 11 | 26 | 40 | 11.92 | 14.22 | 1.19 |
| flat_solo no-cropping | 10.90 ± 7.32 | 8 | 21 | 40 | 8.82 | 12.14 | 1.38 |

The page-only variant uses **−2.3 turns/question** on average — the
biggest turn-count drop of any ablation, *and* it loses 7.8pp
accuracy. The agent finishes faster but cheaper: with no cropping
tool, there are simply fewer productive moves to make on hard
visual-reasoning questions, so it stops earlier. Wrong/correct ratio
also widens (1.38 vs 1.19): the wrong answers it does produce after
truncation are still expensive. This is the strongest case in the
data of "fewer turns, worse outcome" — turn-count alone is not a
quality signal.

## Status

**Done.** Cropping contributes ~7.8pp; effect is highly significant
(5.88 SE).
