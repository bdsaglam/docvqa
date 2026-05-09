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
**Default `use_category_tips=true`** — see "Fairness fix" below.

## Fairness fix (2026-05-08, late)

Original n=3 run shipped without category tips, while the agent solvers
(flat_solo, leanest_solo) did receive tips. To make the comparison fair,
we added a baseline-adapted tips dict (`BASELINE_CATEGORY_TIPS` in
`prompts.py`) that strips agent-only verbs (crop / Python / batch_look /
search) but keeps semantic guidance (PART vs ITEM number, "Width" vs
"Length", I↔1/O↔0 OCR confusion, "broken down into" semantics, etc.),
and re-ran 3 trials with `use_category_tips=true`. The headline number
below uses the tips-on result; the no-tips number is reported as a
secondary cell to isolate the tips contribution.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-remote \
  vlm=qwen-3_5-27b-vllm-remote \
  lm.enable_thinking=false \
  solver=no_loop \
  data.split=val data.num_samples=null \
  max_concurrency=16 \
  run_id=no-loop-tips-3_5-27b-val-t${i}
# Tips-on trials launched 2026-05-08 used run_id=no-loop-tips-val-t${i}.
# Tips-off trials (older) used run_id=no-loop-val-t${i}.
```

## Per-trial scores — tips ON (headline)

| Trial | run_id | Score | Correct | Wall | Sandbox errors |
|---|---|---|---|---|---|
| t1 | `no-loop-tips-val-t1` | 20.00% | 16/80 | ~12 min | 0 (n/a — no REPL) |
| t2 | `no-loop-tips-val-t2` | 21.25% | 17/80 | ~12 min | 0 |
| t3 | `no-loop-tips-val-t3` | 22.50% | 18/80 | ~12 min | 0 |

**Mean: 21.25% ± 1.25pp | range 20.00–22.50%**

## Per-trial scores — tips OFF (ablation)

| Trial | run_id | Score | Correct |
|---|---|---|---|
| t1 | `no-loop-val-t1` | 20.00% | 16/80 |
| t2 | `no-loop-val-t2` | 15.00% | 12/80 |
| t3 | `no-loop-val-t3` | 16.25% | 13/80 |

**Mean: 17.08% ± 2.60pp | range 15.00–20.00%**

## Summary

| Configuration | Mean | Std | n |
|---|---|---|---|
| **no_loop + tips (headline)** | **21.25%** | **1.25pp** | **3** |
| no_loop − tips (ablation) | 17.08% | 2.60pp | 3 |
| Δ tips contribution | **+4.17pp** | (SE 1.67, t=2.50, marginally significant) | — |

## Per-category (mean over 3 trials, tips ON vs tips OFF)

| Category | tips-on t1 | t2 | t3 | tips-on mean | tips-off mean | Δ from tips |
|---|---|---|---|---|---|---|
| business_report | 0/10 | 0/10 | 0/10 | 0.0/10 | 0.0/10 | 0 |
| comics | 2/10 | 2/10 | 2/10 | 2.0/10 | 2.0/10 | 0 |
| engineering_drawing | 4/10 | 5/10 | 4/10 | 4.3/10 | 2.0/10 | **+2.3** |
| infographics | 3/10 | 3/10 | 4/10 | 3.3/10 | 4.0/10 | −0.7 |
| maps | 0/10 | 0/10 | 1/10 | 0.3/10 | 0.7/10 | −0.4 |
| science_paper | 0/10 | 0/10 | 0/10 | 0.0/10 | 0.0/10 | 0 |
| science_poster | 4/10 | 4/10 | 3/10 | 3.7/10 | 3.3/10 | +0.4 |
| slide | 3/10 | 3/10 | 4/10 | 3.3/10 | 1.7/10 | **+1.6** |
| **Overall** | **20.0%** | **21.25%** | **22.5%** | **21.25%** | **17.08%** | **+4.17pp** |

Tips concentrated their lift on engineering_drawing (+2.3 / 10) and
slide (+1.6 / 10) — the categories with the densest domain-specific
tip content (PART vs ITEM numbers, OCR digit confusions, page-
navigation guidance). Infographics regressed slightly (−0.7) — the
"enumerate ALL items" tip may have led the model to longer / less-
focused answers on simple lookups.

## Comparison to scaffold (using fair tips-on baseline)

- **Scaffold (Flat Solo, m=30, full tools):** 44.69% ± 2.81pp (n=8) —
  see `flat-solo-turn-budget-sweep.md` m=30 column.
- **Gap: −23.44pp.** SE on the difference = √(1.25²/3 + 2.81²/8) =
  1.23pp. **t-stat: 23.44 / 1.23 ≈ 19.1 → highly significant.**
- The scaffold lifts the *fair* raw-VLM (composite + tips) accuracy
  by ~23pp on val. This is the corrected headline for claim 3
  (down from −27.6pp when the baseline lacked tips).

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

**Done.** 3 clean trials with tips ON (headline: 21.25% ± 1.25pp) plus
3 clean trials with tips OFF (ablation: 17.08% ± 2.60pp). Fair lift
figure (−23.4pp vs scaffold, 19.1 SE) supports claim 3 strongly.
Composite-image rescaling artifact on long docs (business_report,
science_paper at 0% across all configurations) is a known limit;
addressed by the multi-image variant (separate experiment).
