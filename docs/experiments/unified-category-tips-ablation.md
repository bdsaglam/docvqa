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

| Trial | run_id | host | Score | Correct/total | Wall | Sandbox errors |
|---|---|---|---|---|---|---|
| t1 | `rvlm-unified-val-t1` | TBD | TBD | TBD | TBD | TBD |

## Summary

Pending first trial.

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

**Solver built** (2026-05-28). Solver imports OK; sanity-check confirms
`_UNIFIED_TIPS` length 10,735 chars across all 8 categories. **n=1
val cell NOT YET LAUNCHED.** Target host: amax1 (idle while amax7
finishes Phase 2 refactor).
