# rvlm_subagent_full — Qwen 3.5 27B (val) — n=1 PILOT

## Hypothesis / question

`rvlm_subagent_full` climbs one rung above `rvlm_subagent_ablation`: each
`batch_subagent` delegation is handed to a **full LeanRLM sub-agent** with
its own `batch_look` over the passed images (its own survey/crop/zoom/
re-read/verify loop), **recursion depth hard-capped at 1** (the sub-agent
has `batch_look` only, no `batch_subagent`). Sub-agent budget 6;
fan-out capped (`batch_concurrency=3`); the main-agent prompt was rewritten
to **delegate sparingly** — one substantial subtask at a time, NOT a swarm
of parallel sub-agents (each is now a whole agent run, not a cheap forward).

**Refined hypothesis (user, 2026-06-04):** an agentic sub-call earns its
cost on **many-page documents** — where the main agent can delegate "explore
pages 5–12 for X" and the sub-agent works that page range with its own loop
— but **not on single-page large-image docs**, where `batch_look` cropping
already suffices and the extra agency is pure overhead.

## Setup

- Solver: `rvlm_subagent_full`, `solver.subagent_max_iterations=6`,
  `batch_concurrency=3`, sparse-delegation prompt.
- Model: Qwen 3.5 27B local vllm 8927 (main lm + sub-agent reasoning + sub-agent
  `batch_look` VLM, all 27B), `enable_thinking=false`.
- **8-doc val subset, contrast design** (paired vs `rvlm`/`subagent` on the
  SAME docs — their per-doc results from the n=8 matrix):
  - **MANY-PAGE** (lift predicted): science_paper_1 (44pp), comics_2 (52pp),
    slide_3 (18pp), slide_2 (32pp), science_paper_3 (30pp).
  - **SINGLE-PAGE control** (null predicted): infographics_2, science_poster_1,
    science_poster_2 (all 1pp).
- **n=1** for `rvlm_subagent_full`; `rvlm`/`subagent` columns are their **n=8
  means** restricted to these docs.
- run_id `subagent-full-mp8-t1`. The earlier full-val attempt was killed
  (nested-loop cost ~10–15h); this scoped subset is the bounded pilot.

## Result — paired contrast (split by page count)

| Group | full-agent (n=1) | rvlm (n=8) | subagent (n=8) | Δ vs rvlm | Δ vs subagent |
|---|---|---|---|---|---|
| **MANY-PAGE** (5 docs, 21 Q) | **47.6** | 38.7 | 41.7 | **+8.9pp** | **+6.0pp** |
| **SINGLE-PAGE** (3 docs, 18 Q) | 38.9 | 49.3 | 44.4 | **−10.4pp** | **−5.6pp** |
| **ALL 8** (39 Q) | 43.6 | 43.6 | 42.9 | **0.0pp** | +0.6pp |

**A crossover interaction, exactly as predicted** — full-agent helps where
the doc needs cross-page decomposition and hurts on single images (the
agentic overhead adds noise where a crop was enough). The aggregate is
**0.0pp** — the effect is *completely hidden* without the page-count split,
which is the methodological point: a single-number DocVQA-val comparison
would have called this variant "no different from rvlm" and missed the real
structure.

Per-doc (many-page), full-agent vs rvlm-n8:

| Doc | pp | Q | full | rvlm | Δ |
|---|---|---|---|---|---|
| comics_2 | 52 | 4 | 100 | 44 | **+56** |
| science_paper_3 | 30 | 2 | 50 | 38 | +12 |
| slide_3 | 18 | 5 | 60 | 55 | +5 |
| slide_2 | 32 | 3 | 33 | 29 | +4 |
| science_paper_1 | 44 | 7 | 14 | 29 | **−14** |

## Honest caveats (this is a PILOT, not a result to quote)

1. **n=1, tiny per-group samples** (21 / 18 Q). Trial noise is ~±several pp;
   the *direction and crossover* are trustworthy, the *magnitudes* are not.
2. **The many-page lift is concentrated and non-uniform.** 4 of 5 many-page
   docs improve, but the headline is driven heavily by **comics_2 (+56pp on
   just 4 Q)** — plausibly partly luck. And the **largest/hardest many-page
   doc, science_paper_1 (44pp, 7Q), regressed −14pp** — on the hardest doc
   the full-agent did worse, not better. So "helps on many-page" holds on
   average and in direction but is not a clean per-doc monotonic effect.
3. **Cost & churn.** Many-page × full-agent is the expensive combination
   (~3h for 8 docs at c=4). 72 `reached max iterations` warnings in the run
   — mostly the inner sub-agents capping at budget 6 (expected), but a
   churning main agent on the hard docs (science_paper_1 @ 14%) can't be
   ruled out as a confound from the log alone.

## Read & next step

The pilot **supports the hypothesis directionally** (clean crossover, hidden
in aggregate) and is promising enough to test properly — but n=1 on 39
questions, with the lift leaning on one 4-Q doc and the hardest doc
regressing, **cannot lock it**. Two ways to firm it up:

- **(a) n=8 on a many-page val set** — adds error bars, controls the
  comics_2-luck / science_paper_1-regression concern. Cheaper, reuses
  paired baselines, but DocVQA-val has only ~13 many-page docs total.
- **(b) MMLongBench-Doc** (the long-document benchmark) — many-page is the
  norm and the framing ("long-doc cross-page decomposition") is the
  variant's natural home. Needs a `rvlm_subagent_full_da` dataset-aware
  variant + profile + fair page budget (per the cross-benchmark rule), but
  it's the right test bed for what this pilot is pointing at.

## Status

**PILOT DONE (2026-06-05).** n=1 8-doc contrast: many-page **+8.9pp** vs
rvlm, single-page **−10.4pp**, aggregate 0.0pp. Direction confirms the
many-page hypothesis; magnitude unlocked (n=1, comics_2-driven,
science_paper_1 regressed). Awaiting user call on n=8-many-page vs
MMLongBench-Doc.
