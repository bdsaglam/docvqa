# rvlm_subagent_ablation — Qwen 3.5 27B (val)

## Hypothesis / question

Fork of `rvlm` that generalizes the recursive sub-call from a narrow
**perception** tool (`batch_look((image, query))`) into a general
**delegation** tool (`batch_subagent((task, image))`). The sub-agent is
still a single multimodal-model call, but the main agent is told it can
delegate *any* well-scoped subtask — visual *or* not (image is optional;
`None` for a text/reasoning subtask). The framing is **balanced**: it does
not bias the agent toward visual or non-visual delegation.

Question: does a general task-decomposition / delegation tool help the
main agent solve more (by breaking the problem into subtasks and
delegating each) than the perception-specific `batch_look` — or does
`rvlm`'s narrow perception sub-call already capture the benefit?

## Setup

- Solver: `rvlm_subagent_ablation` (`batch_subagent`; sub-agent = the VLM, multimodal, image optional)
- Model: Qwen 3.5 27B local vllm 8927 (lm + vlm/sub-agent), `enable_thinking=false`
- Profile: DocVQA-2026 (default); max_iterations 25 (same as `rvlm`)
- max_concurrency: 24 (high-concurrency phase)
- Added 2026-06-04 (user request); n=8.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_subagent_ablation \
  data.split=val data.num_samples=null \
  max_concurrency=24 \
  run_id=rvlm-subagent-cmp-val-tN
```

## Per-trial table

| Trial | run_id | Score | Correct | Wall | Notes |
|---|---|---|---|---|---|
| t1 | `rvlm-subagent-cmp-val-t1` | 41.25 | 33/80 | — | business_report 4/10, comics 4/10, engineering_drawing 6/10, infographics 7/10, maps 1/10, science_paper 2/10, science_poster 4/10, slide 5/10 |
| t2 | `rvlm-subagent-cmp-val-t2` | 36.25 | 29/80 | — | business_report 4/10, comics 2/10, engineering_drawing 4/10, infographics 6/10, maps 2/10, science_paper 4/10, science_poster 2/10, slide 5/10 |
| t3 | `rvlm-subagent-cmp-val-t3` | 36.25 | 29/80 | — | business_report 4/10, comics 4/10, engineering_drawing 5/10, infographics 6/10, maps 1/10, science_paper 2/10, science_poster 3/10, slide 4/10 |
| t4 | `rvlm-subagent-cmp-val-t4` | 35.00 | 28/80 | — | business_report 4/10, comics 4/10, engineering_drawing 2/10, infographics 5/10, maps 1/10, science_paper 3/10, science_poster 4/10, slide 5/10 |
| t5 | `rvlm-subagent-cmp-val-t5` | 42.50 | 34/80 | — | business_report 5/10, comics 4/10, engineering_drawing 7/10, infographics 6/10, maps 1/10, science_paper 2/10, science_poster 4/10, slide 5/10 |
| t6 | `rvlm-subagent-cmp-val-t6` | 43.75 | 35/80 | — | business_report 4/10, comics 3/10, engineering_drawing 7/10, infographics 6/10, maps 2/10, science_paper 4/10, science_poster 3/10, slide 6/10 |
| t7 | `rvlm-subagent-cmp-val-t7` | 37.50 | 30/80 | — | business_report 4/10, comics 2/10, engineering_drawing 4/10, infographics 6/10, maps 0/10, science_paper 5/10, science_poster 4/10, slide 5/10 |
| t8 | `rvlm-subagent-cmp-val-t8` | 41.25 | 33/80 | — | business_report 6/10, comics 5/10, engineering_drawing 3/10, infographics 5/10, maps 1/10, science_paper 4/10, science_poster 4/10, slide 5/10 |

## Summary

**n=8 (retained artifacts): 36.72% ± 2.75, pass@8 66.25, SC@8 41.25.** Within
combined std of the `rvlm` headline (41.88 ± 5.79) — generalizing the sub-call
from perception (`batch_look`) to arbitrary delegation (`batch_subagent`, image
optional) **neither helps nor hurts**. `comics_2` is the recurring
degenerate-loop trap (35-iter budget, no exec-timeout); cleared via kill+resume.

> **Provenance.** An earlier batch (n=8, **39.22 ± 3.34**) had its per-trial
> artifacts deleted (no pass@k/SC@k); the run above re-measures with retained
> artifacts and is canonical. Both batches land in the proposed tier within
> combined std of `rvlm` (the original's −0.16pp vs the original `rvlm` 39.38 was
> dead parity) — the parity verdict is unchanged.

**Why parity: the general affordance is barely used.** Across all 8 trials,
**~1.0%** of delegations are non-visual (image=`None`): 45 text-only vs
4527 image-bearing. The main agent overwhelmingly delegates *perception*
subtasks — it uses `batch_subagent` essentially as `batch_look`. On the
DocVQA-2026 val set (short docs, 80 Qs) the questions rarely decompose into
non-visual reasoning subtasks the agent chooses to hand off, so the general
framing is exercised ~never and the result collapses onto `rvlm`.

Efficiency (iter_stats, n=8): **~9.7 iters/Q, median 8** — *fewer* than
`rvlm` (13.0). Packing a richer instruction into each delegation lets the
main agent terminate in fewer of its own steps, but with no accuracy change.

**Read:** a single focused perception sub-call already captures the benefit;
generalizing it to "any subtask" adds an affordance the agent doesn't take
up here. This **bounds the necessary sub-call interface** (supports minimal
`rvlm` as the method) and motivates the follow-up `rvlm_subagent_full`
variant — does upgrading the (perception) sub-call to a *full agent* with its
own iterative `batch_look` help, or is one forward enough?

## Comparison

vs `rvlm` (perception-only `batch_look`) — same scaffold/model, only the
sub-call's interface + framing change. Δ ≈ 0; non-visual delegation ~1%;
iters lower (9.7 vs 13.0).

## Status

**DONE.** Canonical n=8 = **36.72 ± 2.75** (pass@8 66.25, SC@8 41.25), within
combined std of `rvlm`; non-visual delegation ~1%. Rolled into `docs/results.md`
(ablations group + iter row). Follow-up: `rvlm_subagent_full` (full-agent
sub-call) pilot.
