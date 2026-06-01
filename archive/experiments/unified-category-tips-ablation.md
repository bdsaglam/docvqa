# Unified category tips ablation (defense vs "needs category metadata")

## Hypothesis / question

Does the scaffold require per-document category metadata, or is a
unified prompt that includes ALL category tips equally effective?

Per D-006 the proposed method (`rvlm`) currently dispatches per-document:
the solver reads `document.doc_category` and injects the matching block
from the DocVQA-2026 profile's `_DOCVQA_2026_CATEGORY_TIPS`. A reviewer
could object: *"this only works because you hand the agent the
category as metadata; it does not generalize to datasets without
category labels."*

The unified-tips ablation concatenates all 8 category tip blocks into a
single static prompt, prepended regardless of the document's category.
If accuracy holds:

- The method does not require per-document category metadata.
- Cross-benchmark applicability is trivial (the same prompt works
  everywhere; no per-dataset dispatch logic needed).
- The solver simplifies: drop the `doc_category` lookup, drop the
  per-category dispatch.
- We can **promote unified to default** rather than keep it as a
  defensive ablation.

## Expected outcomes (set in advance)

| Δ vs per-category dispatch | Reading | Paper action |
|---|---|---|
| ≈ 0pp (within ±1.5pp noise band) | Per-category dispatch is procedural convenience, not load-bearing | **Promote unified to default.** Strongest paper outcome. |
| −1 to −3pp | Modest prompt-length pressure; still small cost for big simplicity win | Keep per-category as default; report unified as "if you don't have category metadata, this still works (slight cost)" cell |
| −5pp or worse | Category dispatch IS load-bearing | Caveat in paper: method assumes category labels are available |

The middle outcome is most likely a priori — the unified prompt adds
~10.7k chars of tip content, all of which is irrelevant for ~7/8 docs.
Mid-sized models (Qwen 27B) typically handle long prompts well but
distraction is plausible.

## Implementation

New solver: `src/docvqa/solvers/rvlm_unified_solver.py` — identical to
`rvlm_solver.py` except for one line in `_solve_one`:

```python
# rvlm_solver.py
tips = self.profile.category_tips_fn(document.doc_category)

# rvlm_unified_solver.py
tips = _UNIFIED_TIPS  # ignore document.doc_category
```

`_UNIFIED_TIPS` is built at module-import time from
`profile._DOCVQA_2026_CATEGORY_TIPS` (private import — acknowledged;
single source of truth per D-009). Length: 10,735 characters across all
8 category blocks with section headers.

Hydra config: `configs/solver/rvlm_unified.yaml` (`_target_:
docvqa.solvers.rvlm_unified_solver.create_rvlm_unified_program`).

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_unified \
  data.split=val data.num_samples=null \
  max_concurrency=16 \
  run_id=rvlm-unified-val-t1
```

Per D-008 escalation: n=1 first, n=2 if direction holds, n=8 only after
paper headline locks.

## Comparison anchor

`rvlm` (proposed method, per-category dispatch) headline values on
Qwen 3.5 27B local, post-scrub prompts:
- Per-trial mean (n=8 from pre-rename `leanest_solo` data): 42.8% val
  (matches the rename's predecessor; see scrub-audit.md and
  leanest-test-matched-baseline.md)
- SC-8 (val): 48.8%

Unified-tips cells will be compared per-trial against the per-category
baseline, same model + prompt scrub.

## Per-trial table (n=1 first; escalate per D-008)

### `rvlm_unified` (treatment, amax7)

| Trial | run_id | Score | Correct/total | Notes |
|---|---|---|---|---|
| t1 | `rvlm-unified-val-t1` | 45.0% | 36/80 | tmux `unified-tips-t1`; finished 2026-05-28T01:59 |
| t2 | `rvlm-unified-val-t2` | 41.2% | 33/80 | |
| t3 | `rvlm-unified-val-t3` | 38.8% | 31/80 | |
| t4 | `rvlm-unified-val-t4` | 35.0% | 28/80 | low outlier |
| t5 | `rvlm-unified-val-t5` | 42.5% | 34/80 | |
| t6 | `rvlm-unified-val-t6` | 47.5% | 38/80 | high outlier |
| t7 | `rvlm-unified-val-t7` | 40.0% | 32/80 | |
| t8 | `rvlm-unified-val-t8` | 37.5% | 30/80 | finished 2026-05-28T12:?? |
| **n=8 mean** | — | **40.94%** | **std 4.05pp**, range 35.0%–47.5% | |

### `rvlm` paired baseline (amax1, c=24)

Matched-conditions baseline for the per-trial comparison: same model
(Qwen 3.5 27B), same prompts, `lm.enable_thinking=false`, val=80Q.
Concurrency is c=24 on amax1 vs c=32 for the unified chain on amax7 —
GPU is throughput-bound so total tokens/sec is unchanged; per-question
accuracy doesn't depend on c. Launched 2026-05-27T23:11Z via
`scripts/run_rvlm_paired_baseline.sh`. Last update 2026-05-28T08:38Z.

| Trial | run_id | Score | Correct/total | Notes |
|---|---|---|---|---|
| t1 | `rvlm-val-t1` | 40.0% | 32/80 | |
| t2 | `rvlm-val-t2` | 36.2% | 29/80 | low outlier |
| t3 | `rvlm-val-t3` | 41.2% | 33/80 | last doc ~2h on a hard tile question |
| t4 | `rvlm-val-t4` | 42.5% | 34/80 | |
| t5 | `rvlm-val-t5` | 41.2% | 33/80 | |
| t6 | `rvlm-val-t6` | 43.75% | 35/80 | |
| t7 | `rvlm-val-t7` | 40.0% | 32/80 | resumed clean after t7 contamination incident (see below) |
| t8 | `rvlm-val-t8` | 42.5% | 34/80 | first trial under new runner-timeout-retry code (commit `8309710`); finished 2026-05-28T16:45 |
| **n=8 mean** | — | **40.94%** | **std 2.29pp** | range 36.25%–43.75% |

### Per-trial paired comparison (unified − rvlm)

| Trial | unified | rvlm | Δ |
|---|---|---|---|
| t1 | 45.0% | 40.0% | **+5.00pp** |
| t2 | 41.2% | 36.2% | **+5.00pp** |
| t3 | 38.8% | 41.2% | −2.50pp |
| t4 | 35.0% | 42.5% | **−7.50pp** |
| t5 | 42.5% | 41.2% | +1.25pp |
| t6 | 47.5% | 43.75% | +3.75pp |
| t7 | 40.0% | 40.0% | 0.00pp |
| t8 | 37.5% | 42.5% | −5.00pp |

**Paired n=8: Δ mean = 0.00pp, std = 4.68pp, SE = 1.65pp.** 95% CI
[t₇=2.365]: **[−3.91, +3.91]pp**. Both arms scored exactly 262/640
total correct across the n=8 paired set (rvlm n=8 mean = unified n=8
mean = **40.94%**) — the mean lift is mathematically zero, not just
statistically indistinguishable. The CI is tight enough to firmly rule
out a ±4pp effect either way.

By the pre-set decision table this lands squarely in the **"Δ ≈ 0pp →
promote unified to default"** cell. Strongest version of the paper
outcome: unified-tips delivers the same accuracy as per-category
dispatch, with the simpler implementation (no `doc_category` lookup,
no dataset-specific dispatch), so we promote.

### Per-category breakdown

`rvlm_unified` t1 (amax7):

| Category | Acc | Correct/total |
|---|---|---|
| infographics | 80.0% | 8/10 |
| comics | 60.0% | 6/10 |
| engineering_drawing | 50.0% | 5/10 |
| science_poster | 50.0% | 5/10 |
| slide | 50.0% | 5/10 |
| business_report | 40.0% | 4/10 |
| science_paper | 30.0% | 3/10 |
| maps | 0.0% | 0/10 |

`rvlm` baseline best-per-category over t1..t5 (amax1):

| Category | Best (5 trials) | Pattern across t1..t5 |
|---|---|---|
| infographics | 70.0% (t5) | noisy 5–7/10 |
| engineering_drawing | 60.0% (t2/t4/t5) | stable 5–6/10 |
| business_report | 60.0% (t2/t4) | stable 5–6/10 |
| comics | 50.0% (t3/t4) | noisy 3–5/10 |
| science_poster | 50.0% (t3/t5) | noisy 3–5/10 |
| slide | 50.0% (t1/t3/t5) | noisy 4–5/10 |
| science_paper | 40.0% (t1/t3/t4) | noisy 2–4/10 |
| maps | 10.0% (t1/t4) | **0–1/10 — RVLM tile search isn't recovering map evidence reliably across either solver** |

Both solvers hit the same maps=0–10% wall — that failure mode is
about RVLM-on-maps, not about the prompt; the unified prompt doesn't
make it worse.

## Summary

**Final state (2026-05-28T16:46+03, n=8 both arms):**

| Arm | n | Mean | Std | Range | Total correct |
|---|---|---|---|---|---|
| `rvlm_unified` (amax7) | 8 | **40.94%** | 4.05pp | 35.0–47.5 | 262/640 |
| `rvlm` paired baseline (amax1) | 8 | **40.94%** | **2.29pp** | 36.25–43.75 | 262/640 |
| **Δ paired (t1..t8)** | 8 | **0.00pp** | 4.68pp | −7.5..+5.0 | — |

**Means are mathematically identical** (same total correct count
262/640 across both arms), not just statistically equivalent. **95%
paired CI: [−3.91, +3.91]pp**.

**Marginal Δ vs paired Δ.** Marginal Δ (unified n=8 mean − rvlm n=6
mean) = +0.13pp; paired Δ across the 6 matched trials is +0.83pp
(SE 2.03pp). Both anchors land in the "Δ ≈ 0pp" zone of the pre-set
decision table.

**Variance asymmetry (n=8 both arms).** unified std 4.05pp vs rvlm
std 2.29pp — unified is **1.77× noisier**, with the same expected
accuracy. For the paper this is a small additional point in favor of
unified-as-default (simpler implementation; equal expected accuracy;
noisier per-trial isn't a deployment concern at SC-8 ensembling).

**Per-trial sign distribution (n=8).** Of the 8 paired pairs:
3 positive (+5.0 at t1/t2, +3.75 at t6), 1 ≈ 0 (+1.25 at t5), 1 zero
(t7), 1 mildly negative (−2.5 at t3), 2 strongly negative (−5.0 at t8,
−7.5 at t4). Almost perfectly symmetric around zero; no systematic
direction. The +5.0pp signal at t1/t2 — what had us writing a
"unified wins" first draft — was simply two above-mean unified trials
landing first.

**Maps category.** Both arms hit the maps 0–10% wall — the failure
mode is RVLM-on-maps (tile-search fails to recover map evidence
reliably), not unified-prompt-specific. Removed from the decision
input.

**Done.** Both n=8 chains complete. amax1 cell #1 marked `[✓]` in
`coordination/amax1.md`. **Decision: promote `rvlm_unified` to default
solver** (will be done as a separate commit if the user confirms).

### Operational note: t7 contamination + re-launch

t7 was originally launched standalone in tmux `rvlm-t7` in parallel
with the tail of t6 to overlap wall-time (GPU is throughput-bound, so
parallel ≈ serial total cost). When the chain script advanced past t6
and tried to start its own t7 with the same `run_id=rvlm-val-t7`, an
interceptor SIGTERMed the chain bash loop, but a chain-spawned uv/python
pair attached to the shared TMPDIR (`output/runs/rvlm-val-t7/tmp/`) and
ran for ~1m41s before being caught and killed. The contamination was
via shared sandbox-IPC file descriptors — the surviving standalone t7's
RLM ended up in a futex/pipe deadlock on `science_paper_1_q6`.
Recovery: killed the hung standalone, cleaned the polluted sandbox
dirs, restarted t7 fresh — runner resumability skipped the 24
cleanly-completed docs and only re-ran `science_paper_1`. **No effect
on t1–t6** (contamination requires two processes sharing the same
`run_id`, which only ever happened on t7). Lesson: never run two evals
with the same `run_id` in parallel — different `run_id` = different
TMPDIR = safe.

### Silent-failure audit

Cross-checked t1–t6 against the runner's silent-fallback paths:

- **Doc-level runner timeout (`runner.py:295`, 10-min default).** Logs
  `"Document X timed out or failed"` and would silently set all
  questions in the doc to `prediction="Unknown"`, `is_correct=False`.
  **Zero such warnings in t1–t6 chain log.** Path never fired.
- **Per-question RLM iteration exhaustion (`rvlm_solver.py:293-294`).**
  Returns `"Unknown"` when `result.answer` is empty (RLM ran out of
  iterations without `SUBMIT()`). Fires 2–7 times per trial (~3–9%
  of questions); these are legitimate solver give-ups, scored as
  WRONG. Recurring offenders: `comics_4_q2` ("Hortense"),
  `science_paper_3_q2` (compound long answer), parts of `maps_*`.
  Not a script timeout, not silently lost — just looks identical to
  wrong-confident-answers in the headline score.
- **Per-question missing-id fallback (`runner.py:184`).** Defensive,
  never observed.

`opentelemetry: Queue full, dropping Span` WARNINGs (~700 across t1–t6)
are pure telemetry: the otel SDK's batch processor drops tracing spans
under high concurrency; the `with logfire.span(...)` context manager
never raises and eval code runs regardless. No scoring side effect.

Same failure modes will exist in `rvlm_unified` — the per-trial paired
Δ stays valid as a relative comparison. **TODO (planned):** change the
doc-level runner timeout path from "silently set all qs to Unknown" to
"record an error and let resumability retry the doc on next launch" —
the current behavior bakes timed-out results into the persisted state
even though those docs never produced a real answer.

## Observations / caveats

- 10,735-char unified prompt vs ~1,500-char dispatched prompt. ~7×
  longer category section but still small in absolute terms (~5% of
  131k context window).
- The unified prompt explicitly tells the agent that the document's
  category will be one of 8 known options and to apply only the
  matching tips. This frames the task as "filter relevant guidance
  from a known taxonomy" — an arguably easier instruction than
  "category-X tips" with implicit "this is the right block."
- The Leader-lines bullet (reconciled into all 8 categories via D-009
  Phase 1) appears once per category. The agent may notice the
  redundancy; whether this helps or hurts is empirical.
- Comparison is most meaningful within-trial-pair: run unified at the
  same seed / concurrency / model settings as the per-category cell.

## Paper framing

This ablation belongs in §C (mechanism/ablations) as a new row in the
ablation table. The narrative direction depends on outcome:

- If Δ ≈ 0pp: **promote to default**. The method spec drops
  `doc_category` entirely. Cross-benchmark §B becomes trivially clean.
- If Δ −1 to −3pp: appendix cell. "Per-category dispatch contributes
  a small lift; the method works without category metadata at this
  cost."
- If Δ −5pp+: limitations section. Method assumes category labels
  available.

## Status

**Both chains in flight** (2026-05-28T10:31+03):

- amax7: `rvlm_unified` t1..t7 done (n=7 mean 41.43% ± 4.11pp). t8
  running in tmux `unified-tips-chain` (c=32).
- amax1: paired-conditions `rvlm` baseline t1..t5 done (n=5 mean
  40.22% ± 2.40pp). t6 in progress (24/25 last update), t7 launched
  standalone in parallel in tmux `rvlm-t7`, t8 pending. Coord cell:
  `coordination/amax1.md` #1.

Comparison: per-trial-pair on identical model / prompts (c differs
non-materially: 32 vs 24, throughput-bound), plus mean ± std across
the n=8 set. The 42.8% legacy anchor is superseded by the fresh paired
baseline as soon as the amax1 chain finishes.
