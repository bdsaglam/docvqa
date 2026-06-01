# ReAct baseline (REPL-vs-no-REPL ablation)

## Hypothesis / question

Is the LeanRLM code-REPL in `rvlm` *load-bearing*, or would a plain
ReAct-style iterative tool-use agent (no Python execution) reach the
same accuracy?

`rvlm` (proposed method) lets the agent write Python that calls a VLM
tool (`batch_look`) and arbitrary `PIL` ops (`page.crop(...)`, list
comprehensions over results, arithmetic on extracted values). Drop the
REPL and the agent can still call the VLM — just not compose results
programmatically. A reviewer could reasonably ask: *"the code-REPL is
expensive engineering effort; does it actually help, or is ReAct
enough?"*

If react ≈ rvlm: keep the simpler design (no sandbox/IPC infrastructure
to maintain). If react ≪ rvlm: the REPL is doing real work — defend it.

## Expected outcomes (set in advance)

| Δ (react − rvlm), paired n=8 | Reading | Paper action |
|---|---|---|
| ≈ 0pp (±1.5pp noise band) | REPL is convenience, not load-bearing | Demote the REPL; switch baseline to ReAct |
| −1 to −5pp | REPL gives a measurable but modest lift | Report both; lead with rvlm + note ReAct as cheaper variant |
| ≤ −5pp | REPL is load-bearing | Keep rvlm; ablation establishes REPL as a *contributor*, not a luxury |

## Implementation

New solver: `src/docvqa/solvers/react_solver.py` (commit `96246ca`).
Wraps `dspy.ReAct` with two VLM tools, no Python execution:

```python
look(page_index: int, query: str) -> str
look_many(page_indices: list[int], query: str) -> list[str]
```

`look_many` preserves the parallel-VLM-call capability of `rvlm`'s
`batch_look`; **PIL crops and arithmetic on retrieved values are
intentionally absent** — that's the no-REPL constraint.

Dataset-aware via the same `DatasetProfile` mechanism as `rvlm`
(D-009 single source of truth for prompt content). Per-question rate-
limit retry, trajectory persistence, and runner integration all mirror
`rvlm_solver`.

Hydra config: `configs/solver/react.yaml`. Chain script:
`scripts/run_react_chain.sh`.

## Operational note: timeout calibration mid-chain

The default `LMConfig.timeout = 600s` (per-HTTP-request) was sized for
`rvlm`/`LeanRLM` whose per-iteration LLM call emits a short reasoning +
code block. `dspy.ReAct`'s trajectory accumulates ~4 entries per
iteration (thought + tool_name + tool_args + observation, with VLM
observations often hundreds of tokens), so late-iteration completions
on hard docs can exceed 10 min on Qwen 27B and hit `litellm.Timeout`.

Observed in this run: 3 timeouts across t2 (×2) and t3 (×1), all on
`science_poster_2` and `business_report_2`. **First time the new
runner-skip-and-retry behavior (commit `8309710`) was exercised in
anger** — failed docs were left unpersisted so they could be backfilled
without polluting the score with `prediction="Unknown"` placeholders.

Fix (commit `3acee78`): `LMConfig.timeout` is now a configurable field
with default 600s; `run_react_chain.sh` overrides to 1800s
(30 min) for react invocations. Other solvers stay at 600s. The 30-min
override let `science_poster_2`'s slow q4 finish on the third backfill
attempt, giving the chain full 80/80 coverage for all 8 trials.

## Per-trial results (n=8)

### react (this chain, amax1)

| Trial | run_id | Score | Correct/total | Notes |
|---|---|---|---|---|
| t1 | `react-val-t1` | 31.25% | 25/80 | clean |
| t2 | `react-val-t2` | 28.75% | 23/80 | needed 2 backfills (`science_poster_2` timeouts; finally succeeded with `lm.timeout=1800`) |
| t3 | `react-val-t3` | 32.50% | 26/80 | needed 1 backfill (`science_poster_2`) |
| t4 | `react-val-t4` | 27.50% | 22/80 | low outlier |
| t5 | `react-val-t5` | 25.00% | 20/80 | low outlier |
| t6 | `react-val-t6` | 32.50% | 26/80 | |
| t7 | `react-val-t7` | 33.75% | 27/80 | high so far |
| t8 | `react-val-t8` | 32.50% | 26/80 | |
| **n=8 mean** | — | **30.47%** | **195/640** | **std 3.06pp**, range [25.00, 33.75] |

### Per-trial paired comparison

| Trial | react | rvlm | Δ (r − rvlm) | rvlm_unified | Δ (r − unif) |
|---|---|---|---|---|---|
| t1 | 31.25 | 40.00 | **−8.75** | 45.00 | **−13.75** |
| t2 | 28.75 | 36.25 | **−7.50** | 41.25 | **−12.50** |
| t3 | 32.50 | 41.25 | **−8.75** | 38.75 | −6.25 |
| t4 | 27.50 | 42.50 | **−15.00** | 35.00 | **−7.50** |
| t5 | 25.00 | 41.25 | **−16.25** | 42.50 | **−17.50** |
| t6 | 32.50 | 43.75 | **−11.25** | 47.50 | **−15.00** |
| t7 | 33.75 | 40.00 | −6.25 | 40.00 | −6.25 |
| t8 | 32.50 | 42.50 | **−10.00** | 37.50 | −5.00 |

**Every trial is negative on both comparisons. No sign flip in 16
paired observations.**

### Paired Δ stats

| Comparison | mean Δ | std | SE | 95% CI [t₇=2.365] | contains 0? |
|---|---|---|---|---|---|
| react − rvlm (n=8) | **−10.47pp** | 3.53pp | 1.249pp | **[−13.42, −7.52]pp** | **No** |
| react − rvlm_unified (n=8) | **−10.47pp** | 4.77pp | 1.686pp | **[−14.46, −6.48]pp** | **No** |

(Same mean Δ for both anchors because rvlm n=8 mean = rvlm_unified n=8
mean = 40.94% — see `docs/experiments/unified-category-tips-ablation.md`.)

**By the pre-set decision table, this lands squarely in the
"−5pp or worse → REPL is load-bearing" cell.**

## Per-category breakdown (n=8 mean ± std)

| Category | react | rvlm | Δ (rvlm − react) |
|---|---:|---:|---:|
| business_report | 26.2% ± 16.9 | 50.0% ± 10.7 | **+23.8pp** |
| comics | 23.8% ± 9.2 | 37.5% ± 8.9 | **+13.7pp** |
| engineering_drawing | 35.0% ± 13.1 | 58.8% ± 6.4 | **+23.8pp** |
| infographics | 53.8% ± 9.2 | 57.5% ± 7.1 | +3.7pp |
| maps | 3.8% ± 5.2 | 3.8% ± 5.2 | 0.0pp |
| science_paper | 32.5% ± 12.8 | 32.5% ± 8.9 | 0.0pp |
| science_poster | 26.2% ± 13.0 | 40.0% ± 9.3 | **+13.8pp** |
| slide | 42.5% ± 7.1 | 47.5% ± 7.1 | +5.0pp |

Where the REPL helps most (`business_report`, `engineering_drawing` at
+23.8pp each, `comics` and `science_poster` at +13–14pp) are exactly
the categories the per-category tips highlight as **PIL-crop-heavy**:
small-text reading from a precise region requires zoom-then-read,
which ReAct can't do without `pages[i].crop((l,t,r,b))`. Categories
where ReAct stays competitive (`infographics`, `slide`) tend to be
full-page-readable; categories with floor effects (`maps`,
`science_paper`) limit both arms equally.

## Summary

**REPL is load-bearing.** Paired n=8 confirms a robust, large (~10pp)
accuracy gap between `rvlm` (LeanRLM + Python REPL + VLM tools) and a
matched `dspy.ReAct` baseline using the same VLM tools without code
execution. The 95% CI on the paired Δ is **[−13.42, −7.52]pp** vs rvlm
and **[−14.46, −6.48]pp** vs rvlm_unified — both cleanly outside the
±1.5pp noise band and well past the −5pp "load-bearing" threshold set
in advance.

**Pattern of the gap.** The REPL advantage is concentrated in
categories that reward zoom-then-read (business_report, engineering
drawings, comics, science posters) — exactly where `pages[i].crop(...)`
in the rvlm sandbox lets the agent extract text from a precise region.
On full-page-readable categories (infographics, slides) the gap shrinks
to a few pp; on floor-effect categories (maps, science_paper) the
arms tie.

**Paper framing.** The REPL is not a luxury — it is a perception
multiplier on small-text / region-localization questions. The
proposed method's `batch_look + PIL.crop + Python composition` lets
the agent zoom *adaptively* per page; the ReAct baseline can only
ask the VLM about whole pages. This ablation establishes the REPL
as a contributor to method capability.

**Variance.** react std (3.06pp) sits between rvlm (2.29pp) and
rvlm_unified (4.05pp) — no special variance pathology. The chain
hit 3 `litellm.Timeout`s on the original run; the new
runner-skip-and-retry path (commit `8309710`) plus a per-solver
timeout bump (`lm.timeout=1800` in `run_react_chain.sh`, via the
config-driven `LMConfig.timeout` field added in `3acee78`) recovered
all 6 missing questions cleanly, giving full 80/80 coverage on every
trial.

## Status

**n=8 complete** (2026-05-28 → 2026-05-29; amax1, c=24). Cell marked
done in `coordination/amax1.md`. Comparison anchors:

- `rvlm` paired baseline (cf. cell 1 of amax1) — 40.94% n=8 mean,
  paired Δ = +0.00pp vs `rvlm_unified` (see
  `docs/experiments/unified-category-tips-ablation.md`).
- `rvlm_unified` n=8 (amax7) — 40.94% n=8 mean.

Decision: **keep `rvlm` family as the proposed method.** ReAct
baseline appears in the ablation table as the "−REPL" row.
