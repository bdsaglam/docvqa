# Experiments

Per-cell writeups for paper-relevant experiments under the D-006 framing
(visual-context-budget hypothesis). One file per experiment cell — the
canonical record so future sessions don't have to re-derive numbers
from `output/runs/`.

> **Pre-D-010 experiments** (the original `flat_solo` / `leanest_solo`
> scaffold cells, model-axis cells on pre-scrub prompts, legacy
> cross-benchmark cells) live in
> [`archive/experiments/`](../../archive/experiments/). Their headline
> results are mirrored in `docs/results.md` with the legacy→new name
> mapping.

## File layout

Each experiment file should have these sections:

1. **Hypothesis / question** — what we're testing, in one sentence. Tie
   back to a D-006 prediction (1 model-size, 2 doc-length, 3 active-
   perception mechanism) where applicable.
2. **Setup** — solver, model, profile, max_iterations, concurrency.
3. **Command** — the exact CLI, copy-pasteable. Use post-D-010 solver
   names (`solver=rvlm`, `solver=rvlm_ocr`, etc.).
4. **Per-trial table** — `run_id`, score, correct/total, wall time, any
   contamination signals. One row per trial.
5. **Summary** — mean ± std, n, range. Note any excluded trials.
6. **Comparison** — gap vs the appropriate baseline + standard error.
   Be explicit about what the baseline is and which prediction this
   speaks to.
7. **Observations / caveats** — what was surprising, infra issues,
   links to memory entries.
8. **Status** — `in progress`, `done`, or `done — needs replication`.

## Conventions

- **Run IDs** use post-D-010 solver names: `rvlm-val-tN`,
  `rvlm-ocr-val-tN`, `raw-vlm-multi-<model>-val-tN`, etc. Pre-D-010
  IDs (`flat-solo-*`, `leanest-solo-*`, `no-loop-multi-*`) stay as
  they were per D-010 — only new runs follow the new naming.
- **Trial budget escalation per D-008.** Cells start at n=1. Escalate
  to n=2 if the n=1 direction holds; n=8 only after the paper headline
  framing locks. Document the n at each stage.
- **Cross-benchmark cells** (MP-DocVQA, MMLongBench-Doc, etc.) use the
  DA-by-default solvers (`rvlm`, `rvlm_ocr`, `raw_vlm_multi`) with the
  appropriate `data.dataset` override and `data.use_profile_scoring=true`.
  Per D-009 the profile system carries per-benchmark prompts.
- **Coordination.** Before starting a cell, claim it in
  `coordination/<host>.md` per `coordination/README.md`. Don't
  duplicate work across hosts.
- **One cell per file** unless the cell is a parameter sweep across
  one variable (e.g., turn-budget m={5,10,20,30,40}), in which case
  bundle into one file with one section per cell.

## Index — active

Currently retained in `docs/experiments/` (baselines + in-flight):

| File | Status |
|---|---|
| [no-loop-baseline.md](no-loop-baseline.md) | raw-VLM single-image baseline (`raw_vlm_single`); n=3 tips-on + n=3 tips-off val cells. Anchors prediction-3 "scaffold matters" delta. |
| [no-loop-multi-image.md](no-loop-multi-image.md) | raw-VLM multi-image baseline (`raw_vlm_multi`); n=3 tips-on + n=3 tips-off val cells. Headline baseline for the model-axis lift figure. |
| [official-baseline-qwen27b.md](official-baseline-qwen27b.md) | competition-kit prompt verbatim; val n=3 = 21.67% ± 1.91pp. Test abandoned (Qwen 27B context overflow on long test docs). Context for the paper's baseline-comparison section. |
| [split-calibration-no-loop-multi.md](split-calibration-no-loop-multi.md) | val SC-8 20.0% / test SC-8 11.0% on `raw_vlm_multi`; **9pp split-difficulty floor**. Anchors the val→test gap discussion. |
| [unified-category-tips-ablation.md](unified-category-tips-ablation.md) | `rvlm_unified` ablation: all 8 category tips concatenated. **Currently running** (n=1 val on amax7). Tests whether per-document category metadata is required. |

## Archive

Pre-D-010 scaffold experiments + process narratives + shelved approaches:
[`archive/experiments/`](../../archive/experiments/) — see
[`archive/experiments/README.md`](../../archive/experiments/README.md)
for the index and legacy→new name mapping.

## Template for new cells

Copy this skeleton when adding a new cell (e.g., for `rvlm_ocr` or
`direct_vlm` results):

```markdown
# <Solver/cell name>

## Hypothesis / question

<one-sentence statement; tie to a D-006 prediction>

## Setup

- Solver: `<solver>` (per `docs/solvers/<solver>.md`)
- Model: Qwen 3.5 27B local vllm 8927
- Profile: DocVQA-2026 (default)
- max_iterations: 25
- max_concurrency: 24 (or 32 for c=32 dispatch)

## Command

\```bash
uv run python evals.py \\
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \\
  lm.enable_thinking=false \\
  solver=<solver> \\
  data.split=val data.num_samples=null \\
  max_concurrency=24 \\
  run_id=<solver>-val-t1
\```

## Per-trial table

| Trial | run_id | Score | Correct | Wall | Notes |
|---|---|---|---|---|---|
| t1 | `<solver>-val-t1` | TBD | /80 | TBD | |

## Summary

n=1 (per D-008; escalate to n=2 if direction holds).

## Comparison

Compare against: <baseline cell, e.g., raw_vlm_multi val SC-8 20.0%>.
Δ = TBD.

## Observations / caveats

(empty until trial completes)

## Status

in progress / done / done — needs replication
```
