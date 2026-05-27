# Archived Solver Docs

Solvers shelved by [D-006](../../paper/decisions.md) (research-mode framing
pivot, 2026-05-27). They are out of scope for the paper but the code lives
in the tree and the docs are preserved here for reference.

## What's in here

| File | Solver | Why shelved |
|---|---|---|
| [`routing.md`](routing.md) | `routing` (`docvqa.solvers.routing_solver`) | Per-category dispatch was a competition tactic, not a research finding. The paper reports one solver per cell, no routing. |
| [`flat-batch.md`](flat-batch.md) | `flat_batch` (`docvqa.solvers.flat_batch_solver`) | Batch baseline was ~5-10pp below the solo solvers. D-006 reframing dropped competition-tactic cells; the raw-VLM baseline (`raw_vlm_multi`) is the canonical comparison point. |
| [`lean-solo.md`](lean-solo.md) | `lean_solo` (`docvqa.solvers.lean_solo_solver`) | Strict mid-point between `rvlm` (no OCR) and `rvlm_ocr` (full OCR extension). D-006 isolated the OCR-extension cell on `rvlm_ocr`, which is a clean fork; `lean_solo` is no longer a paper cell. |

These docs use the pre-D-010 engineering names (e.g. `flat_solo`, `leanest_solo`)
deliberately — they describe the state of the codebase at the time the
solvers were active. The rename map is in
[D-010](../../paper/decisions.md#d-010-solver-renames--behavior-based-engineering-names).

## See also

- Current paper-solver docs are one level up in [`docs/solvers/`](../README.md).
- Historical experiment records in `docs/experiments/*.md` also use old names.
