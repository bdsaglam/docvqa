# rvlm_minimal — generality of the proposed method

## Hypothesis / question

Does the RVLM scaffold's score on DocVQA-2026 depend on
benchmark-tuned prompt engineering (the 8 hand-crafted category tip
blocks for business_report, comics, engineering_drawing, infographics,
maps, science_paper, science_poster, slide), or is the recursive
perception mechanism the load-bearing piece?

Per D-006, the visual-context-budget hypothesis predicts the
mechanism. If `rvlm_minimal` (with the category tips entirely removed,
only generic document-shape guidance in the body) lands within trial
noise of `rvlm_unified`, that's direct evidence for the hypothesis: the
tips are procedural, not foundational.

## Pre-set decision rules (set before launching, locked)

| Δ (minimal − unified) | Reading | Paper action |
|---|---|---|
| ≈ 0pp (within ±1.5pp paired noise) | The hand-crafted tips don't carry the method | **Headline.** rvlm_minimal is the proposed method; the unified/per-category solvers become engineering ablations showing what tip-tuning buys. |
| −1 to −3pp | Tips contribute a small lift; method generalizes at a modest cost | Honest cost line in paper: "the method works without benchmark-specific tips at a ~Xpp cost." Minimal becomes the headline "generalizable variant." |
| −5pp or worse | DocVQA-2026 tips are load-bearing | Honest limitation: per-dataset tip curation is part of the method. Weakens the generality claim materially. |

This is the **strongest threat to the paper's generality claim**. The
unified-tips ablation (Δ ≈ 0pp vs `rvlm`) only showed dispatch wasn't
load-bearing; it left the *content* of the tips intact. This cell
tests the content.

## Implementation

`src/docvqa/solvers/rvlm_minimal_solver.py`. The solver body has:

1. **Tool docs** (brief): `batch_look` + `SUBMIT`, with "what / when /
   how" structure.
2. **Approach**: SURVEY → LOCATE → EXTRACT → VERIFY → SUBMIT.
3. **Document-shape guidance** (4 generic patterns, no benchmark
   category names): high-density single page; many-page document;
   counting/superlatives; VLM disagreement.

Zero DocVQA-2026 strings in the solver body. Dataset-specific content
stays in `profile.answer_formatting_rules` (1.46 kB: the official
answer format rules — "Unknown" sentinel, date format, etc.) and
`profile.question_format_hint_fn` (None for DocVQA-2026; per-question
hint for MMLongBench-Doc).

Prompt sizes (DocVQA-2026 profile):

| Component | rvlm_unified | rvlm_minimal |
|---|---|---|
| Body | 2.6 kB | 2.8 kB |
| Category tips (8 blocks) | 10.7 kB | **0 kB** |
| Profile answer-format rules | 1.5 kB | 1.5 kB |
| **Total agent prompt** | **14.8 kB** | **4.3 kB** |

71% reduction in total agent prompt; 100% reduction in
benchmark-category content.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_minimal \
  data.split=val data.num_samples=null \
  max_concurrency=32 \
  run_id=rvlm-minimal-val-t1
```

Chain (t1..t8) via `bash scripts/run_rvlm_minimal_chain.sh`.

## Comparison anchors (post-D-009, Qwen 3.5 27B, c=32 on amax7)

- `rvlm_unified` n=8: **40.94% ± 4.05pp** (this work)
- `rvlm` (per-category dispatch) n=7 from amax1: **40.71% ± 2.38pp**;
  n=8 final pending t8

## Per-trial table

### `rvlm_minimal` (treatment, amax7)

| Trial | run_id | Score | Correct/total | Notes |
|---|---|---|---|---|
| t1 | `rvlm-minimal-val-t1` | TBD | /80 | |
| t2..t8 | `rvlm-minimal-val-t{2..8}` | TBD | /80 | in tmux `rvlm-minimal-chain` |

### Paired comparison (minimal − unified) — populated as trials land

| Trial | minimal | unified | Δ |
|---|---|---|---|
| t1 | TBD | 45.0% | TBD |
| t2 | TBD | 41.2% | TBD |
| t3 | TBD | 38.8% | TBD |
| t4 | TBD | 35.0% | TBD |
| t5 | TBD | 42.5% | TBD |
| t6 | TBD | 47.5% | TBD |
| t7 | TBD | 40.0% | TBD |
| t8 | TBD | 37.5% | TBD |

## Observations / caveats

- The unified-tips ablation showed dispatch isn't load-bearing
  (Δ ≈ 0pp at n=6 paired). This cell goes one step further and tests
  whether the tip *content* is load-bearing.
- The `_per_question_prefix` plumbing is preserved, so when this
  solver is run on MMLongBench-Doc the per-question format hint is
  still applied — the body is dataset-agnostic, not feature-stripped.
- The 4 document-shape patterns are calibrated against failure modes
  that show up across the literature (high-density pages, long
  documents, superlatives, vision-model disagreement). They're not
  derived from DocVQA-2026 per-category error analysis.

## Status

**In progress.** Solver built 2026-05-28; chain queued on amax7 (vllm
free post-unified-tips). amax1 still finishing rvlm-val-t8 (the
per-category baseline n=8 final).
