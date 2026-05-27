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

| Trial | run_id | host | Score | Correct/total | Wall | Notes |
|---|---|---|---|---|---|---|
| t1 | `rvlm-unified-val-t1` | amax7 | **45.0%** | 36/80 | ~52min | c=32; tmux `unified-tips-t1`; finished 2026-05-28T01:59 |

Per-category (n=1):

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

## Summary

n=1: **45.0% overall.** Comparison anchor (rvlm per-category per-trial
mean from legacy `leanest_solo` n=8 = **42.8%**) → Δ = **+2.2pp**,
inside the ±3pp trial-noise band. Per the pre-set decision table this
lands in the "Δ ≈ 0pp → **promote unified to default**" cell.

Caveat: n=1 only. Per D-008 we need n=2 to confirm direction before
acting. The maps=0% category is the obvious red flag — needs
inspection to confirm it's a per-trial noise effect on a 1-page,
counting-heavy category rather than a unified-prompt failure mode.

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

**n=1 done** (2026-05-28; amax7, c=32). User opted to skip D-008's n=2
intermediate step and escalate directly to **n=8**.

- amax7: `rvlm_unified` t2..t8 in tmux `unified-tips-chain` (ETA ~6h).
- amax1: paired-conditions `rvlm` baseline t1..t8 in tmux `rvlm-paired`
  (queued in `coordination/amax1.md` cell #1).

Comparison will be per-trial-pair on identical model / prompts / c, plus
mean ± std across the 8-trial set. The 42.8% legacy per-trial mean (used
for the n=1 sanity check) gets replaced by this fresh paired baseline
once the chain finishes.
