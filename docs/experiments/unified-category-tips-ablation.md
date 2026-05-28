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

| Trial | run_id | Score | Correct/total | Wall | Notes |
|---|---|---|---|---|---|
| t1 | `rvlm-unified-val-t1` | **45.0%** | 36/80 | ~52min | c=32; tmux `unified-tips-t1`; finished 2026-05-28T01:59 |
| t2 | `rvlm-unified-val-t2` | — | — | — | in tmux `unified-tips-chain` |
| t3..t8 | `rvlm-unified-val-t{3..8}` | — | — | — | queued in same chain |

### `rvlm` paired baseline (amax1, c=24)

Matched-conditions baseline for the per-trial comparison: same model
(Qwen 3.5 27B), same prompts, `lm.enable_thinking=false`, val=80Q.
Concurrency is c=24 on amax1 vs c=32 for the unified chain on amax7 —
GPU is throughput-bound so total tokens/sec is unchanged; per-question
accuracy doesn't depend on c. Launched 2026-05-27T23:11Z via
`scripts/run_rvlm_paired_baseline.sh`; t7 launched standalone in
parallel with tail of t6 (interceptor armed against the chain's
duplicate-t7 launch). Last update 2026-05-28T08:02Z.

| Trial | run_id | Score | Correct/total | Notes |
|---|---|---|---|---|
| t1 | `rvlm-val-t1` | 40.0% | 32/80 | |
| t2 | `rvlm-val-t2` | 36.2% | 29/80 | low outlier |
| t3 | `rvlm-val-t3` | 41.2% | 33/80 | last doc ~2h on a hard tile question |
| t4 | `rvlm-val-t4` | 42.5% | 34/80 | high so far |
| t5 | `rvlm-val-t5` | 41.2% | 33/80 | |
| t6 | `rvlm-val-t6` | in progress | 24/25 docs | hard tile question on last doc, ~2h+ |
| t7 | `rvlm-val-t7` | in progress | parallel | standalone in tmux `rvlm-t7` |
| t8 | `rvlm-val-t8` | pending | — | will launch after t6 finishes |
| **n=5 mean** | — | **40.22%** | std ~2.3pp | t1..t5 only |

### Per-trial paired comparison (unified − rvlm)

| Trial | unified | rvlm | Δ |
|---|---|---|---|
| t1 | 45.0% | 40.0% | **+5.0pp** |
| t2..t8 | pending | pending | — |

n=1 paired Δ = +5.0pp, materially above the ±1.5pp noise band — the
direction that would land "promote unified to default" under the
pre-set decision table. n=8 needed to confirm.

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

n=1: **45.0% overall** (unified).

Two comparison anchors, both pointing the same direction:
- vs **fresh paired baseline** (rvlm n=5 mean from amax1 chain so far)
  = **40.22%** → Δ = **+4.78pp**, well outside the ±1.5pp noise band.
- vs legacy per-trial mean from pre-rename `leanest_solo` n=8 = 42.8%
  → Δ = +2.2pp (kept for historical context only; the paired anchor
  supersedes this).

Per the pre-set decision table, +4.78pp is in the "Δ ≈ 0 → promote
unified to default" cell on the strong side — if it holds across the
full n=8 paired chain, the promote decision is robust.

Caveat: still n=1 on the unified side and n=5 on the rvlm side. n=8
paired both sides will close it. The maps=0% category is the obvious
red flag — needs inspection to confirm it's a per-trial noise effect
on a 1-page, counting-heavy category rather than a unified-prompt
failure mode. (Update from amax1 data: the paired rvlm baseline hits
maps 0–10% across t1..t5 too, so the failure mode is RVLM-on-maps,
not unified-prompt-specific.)

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

**Both chains in flight** (2026-05-28):

- amax7: `rvlm_unified` t1 done (45.0%). t2..t8 running in tmux
  `unified-tips-chain` (c=32).
- amax1: paired-conditions `rvlm` baseline t1..t5 done (n=5 mean
  40.22%, std ~2.3pp). t6 in progress (24/25), t7 launched standalone
  in parallel in tmux `rvlm-t7`, t8 pending. Coord cell:
  `coordination/amax1.md` #1.

Comparison: per-trial-pair on identical model / prompts (c differs
non-materially: 32 vs 24, throughput-bound), plus mean ± std across
the n=8 set. The 42.8% legacy anchor is superseded by the fresh paired
baseline as soon as the amax1 chain finishes.
