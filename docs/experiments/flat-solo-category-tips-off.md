# Flat Solo — category tips ablation

**Hypothesis:** the per-category prompt tips in `src/docvqa/prompts.py`
contribute meaningfully to accuracy. Removing them tests whether the
handcrafted hints carry weight or are decoration.

**Setup:** Flat Solo solver, lean RLM, no thinking, default m=30.
Qwen 3.5 27B (vLLM local 8927) for both LM and VLM. Val 80q.
`solver.use_category_tips=false`. 8 clean trials targeted (t1
contaminated, replaced with t9).

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local \
  vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=flat_solo \
  solver.rlm_type=lean \
  solver.use_category_tips=false \
  data.split=val data.num_samples=null \
  max_concurrency=24 \
  run_id=flat-solo-no-tips-3_5-27b-val-t${i}
```

## Per-trial scores

| Trial | Score | Sandbox errors | Notes |
|---|---|---|---|
| t1 | 18.75% | **2064** | **EXCLUDED** — sandbox subprocess crashed repeatedly during run, agent abstained 55/80q with "Unknown". Triggered by concurrent 4-GPU vllm spin-up on host. Run dir renamed `_excluded.flat-solo-no-tips-3_5-27b-val-t1`. |
| t2 | 38.75% | 0 | clean |
| t3 | 37.50% | 0 | clean |
| t4 | 35.00% | 0 | clean |
| t5 | 36.25% | 0 | clean |
| t6 | 42.50% | 0 | clean |
| t7 | 43.75% | 0 | clean |
| t8 | 36.25% | 0 | clean |
| t9 | 40.00% | 0 | clean — replacement for excluded t1 |

## Summary (n=8 clean, t1 excluded)

- mean = **38.75% ± 3.13pp**
- range 35.0–43.8
- per-trial: [38.8, 37.5, 35.0, 36.2, 42.5, 43.8, 36.2, 40.0]

## Comparison to baseline

- **Baseline (full tips):** 44.69% ± 2.81pp (n=8) — see
  `flat-solo-turn-budget-sweep.md` m=30 column.
- **Gap: −5.94pp**, SE on the difference = √(3.13²/8 + 2.81²/8) = 1.49pp
- **t-stat: 5.94 / 1.49 ≈ 3.99 → highly significant.**

## Observations

- Tips contribute ~6pp net. Less than VLM cropping (~8pp) but on the
  same order — both are real components of the scaffold.
- The contaminated t1 (18.75%) was a near-loss-of-signal event: 55/80
  questions abstained because the python sandbox kept dying.
  See `sandbox_subprocess_errors` memory entry for diagnosis pattern.
- Variance (3.13pp) similar to baseline (2.81pp); turning off tips
  doesn't appear to destabilize the agent, it just makes it slightly
  worse on average.

## Efficiency (turns per question)

Pooled across 8 clean trials × 80q = 640 questions (`_excluded.` t1
sandbox-contaminated trial omitted).

| Cell | turns mean ± std | median | p90 | max | turns_correct | turns_wrong | wrong/correct |
|---|---|---|---|---|---|---|---|
| flat_solo m=30 (baseline, tips ON) | 13.19 ± 7.85 | 11 | 26 | 40 | 11.92 | 14.22 | 1.19 |
| flat_solo no-tips | 12.78 ± 7.98 | 10 | 25 | 40 | 10.50 | 14.21 | 1.35 |

Tips do **not** materially change the total number of turns the agent
spends (−0.4 mean, n.s.). What they shift is **where** turns land:
without tips, correct trajectories are 1.4 turns shorter on average
(10.5 vs 11.9) — the agent either gets it quickly or gives up. Wrong
trajectories are unchanged. Net: thrash ratio jumps from 1.19 to
1.35. Tips appear to encourage more careful (longer) verification on
correct answers rather than reducing total work.

## Test set (overfit check)

8 trials at the same config on test, SC-8 voted:

| Setup | Val (per-trial) | Test (SC-8) | Val gap | Test gap |
|---|---|---|---|---|
| flat_solo full (tips ON) | 44.69 ± 2.81pp (n=8) | **38.75%** | — | — |
| flat_solo no-tips | 38.75 ± 3.13pp (n=8) | **35.00%** | **−5.94pp** | **−3.75pp** |

**Tips are NOT val-overfit.** They help on test too, just by a smaller
margin (−3.75pp vs val's −5.94pp). That ~2pp attenuation is normal
val→test shrinkage on a harder split, not overfitting. If tips had
truly been memorising val-specific question patterns, the no-tips
test score would have been at or above the full-tips test score.
The directional consistency rules that out.

Even without tips, the scaffold on Qwen 3.5 27B (35.00% test SC-8)
still beats Gemini 3 Flash (33.75%) and GPT-5 Mini (22.5%) — only
GPT-5.2 (35.0%) and Gemini Pro (37.5%) are competitive.

Submissions:
- `submissions/flat-solo-no-tips-3_5-27b-test-sc8.json` (scored 35.00%)
- per-trial: `submissions/flat-solo-no-tips-3_5-27b-test-t{1..8}.json`

## Status

**Done.** Tips contribute ~6pp on val (highly significant, 3.99 SE)
and ~4pp on test (SC-8). Directionally consistent — tips generalise,
not overfit.
