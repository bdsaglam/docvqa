# No-loop baseline (raw VLM, single composite image)

**Hypothesis:** the agent loop + tool channel contributes meaningfully on
top of a raw VLM call. A single forward pass with all pages stacked into
one composite image is the most defensible "raw model" point — no REPL,
no tools, no agent.

**Setup:** `solver=no_loop`. Pages concatenated vertically into one
composite image, capped at 16384px tall (proportionally rescaled if
exceeded). Single VLM call per question. Qwen 3.5 27B serves as the VLM
via Host B vllm 8928 (4-GPU); LM unused by this solver. Val 80q,
`lm.enable_thinking=false`, max_concurrency=16, question_concurrency=4.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-remote \
  vlm=qwen-3_5-27b-vllm-remote \
  lm.enable_thinking=false \
  solver=no_loop \
  data.split=val data.num_samples=null \
  max_concurrency=16 \
  run_id=no-loop-3_5-27b-val-t${i}
# Note: trials launched 2026-05-07/08 used run_id=no-loop-val-t${i}
# (without the model tag); run dirs are output/runs/no-loop-val-t{1,2,3}.
```

## Per-trial scores

| Trial | run_id | Score | Correct | Wall | Sandbox errors |
|---|---|---|---|---|---|
| t1 | `no-loop-val-t1` | 20.00% | 16/80 | ~12 min | 0 (n/a — no REPL) |
| t2 | `no-loop-val-t2` | 15.00% | 12/80 | ~12 min | 0 |
| t3 | `no-loop-val-t3` | 16.25% | 13/80 | ~12 min | 0 |

## Summary (n=3, all clean)

- mean = **17.08% ± 2.60pp** (sample std, n−1)
- range 15.00–20.00%
- per-trial: [20.00, 15.00, 16.25]

## Per-category (mean over 3 trials)

| Category | t1 | t2 | t3 | Mean |
|---|---|---|---|---|
| business_report | 0/10 | 0/10 | 0/10 | 0.0/10 |
| comics | 2/10 | 2/10 | 2/10 | 2.0/10 |
| engineering_drawing | 2/10 | 2/10 | 2/10 | 2.0/10 |
| infographics | 5/10 | 4/10 | 3/10 | 4.0/10 |
| maps | 2/10 | 0/10 | 0/10 | 0.7/10 |
| science_paper | 0/10 | 0/10 | 0/10 | 0.0/10 |
| science_poster | 3/10 | 3/10 | 4/10 | 3.3/10 |
| slide | 2/10 | 1/10 | 2/10 | 1.7/10 |
| **Overall** | **20.0%** | **15.0%** | **16.25%** | **17.08%** |

## Comparison to scaffold

- **Scaffold (Flat Solo, m=30, full tools):** 44.69% ± 2.81pp (n=8) —
  see `flat-solo-turn-budget-sweep.md` m=30 column.
- **Gap: −27.61pp.** SE on the difference = √(2.60²/3 + 2.81²/8) =
  1.74pp. **t-stat: 27.61 / 1.74 ≈ 15.9 → highly significant.**
- The scaffold lifts raw-VLM accuracy by ~28pp on val. This is the
  headline number for claim 3 (component contributions).

## Observations

- **business_report and science_paper at 0% across all 3 trials.** Long
  docs (105–181 pages, 19–44 pages) get crushed by the 16384px composite-
  image cap; per-page resolution drops to ~80–150px tall after rescaling
  — far below legibility. This is a structural limit of the composite
  baseline, not a model failure. See `no-loop-multi-image.md` for the
  multi-image variant that addresses this.
- **comics and engineering_drawing at exact 2/10 every trial.** Suggests
  the raw VLM gets a deterministic subset of "easy" questions right and
  loses the rest the same way each time — low entropy in the wrong-
  answer distribution.
- **infographics is the strongest category (4.0/10 mean).** Single-page,
  text-heavy layouts survive the composite path; this is where raw VLM
  is closest to the scaffolded agent.
- **maps high variance on a small base.** t1 hit 2/10, t2/t3 0/10 —
  variance on 10 questions is large; not a real signal.
- Variance (2.60pp std) is comparable to the scaffold's 8-trial std
  (2.81pp), so the lift is not noise.
- Each trial is fast (~12 min wall) because there's no agent loop —
  one VLM forward pass per question.
- Solver source: `src/docvqa/solvers/no_loop_solver.py`. Bug fix
  during this experiment: `evals.py:80` was silently ignoring
  `data.num_samples`; now passes it through.

## Status

**Done.** 3 clean trials. Lift figure (−27.6pp vs scaffold) supports
claim 3 strongly. Composite-image rescaling artifact on long docs is a
known limit; addressed by the multi-image variant (separate experiment).
