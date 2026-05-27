# Solver Docs

Per-solver documentation for the post-[D-010](../paper/decisions.md#d-010-solver-renames--behavior-based-engineering-names)
codebase. Engineering names below are the names in
`src/docvqa/solvers/`; paper-facing method names are still TBD (see
[D-005](../paper/decisions.md#d-005-position-vs-rvlm-and-madqa)
implication: pick a non-RVLM name to avoid arXiv:2603.24224 collision).

## Paper-cell taxonomy

| Engineering name | Paper role | Tool surface | Doc |
|---|---|---|---|
| **`rvlm`** | Proposed method (M) | `batch_look` only | [rvlm.md](rvlm.md) |
| **`rvlm_ocr`** | +OCR extension | `batch_look` + `search` + `page_texts` | [rvlm-ocr.md](rvlm-ocr.md) |
| **`rvlm_full`** | Kitchen-sink (appendix, role TBD per task #16) | `batch_look` + `look` + `search` + `page_texts` | [rvlm-full.md](rvlm-full.md) |
| **`direct_vlm`** | Alternative-angle method | Multimodal LLM in REPL, `display()` only | [direct-vlm.md](direct-vlm.md) |
| **`raw_vlm_multi`** | Raw-VLM baseline (multi-image) | one forward pass, no scaffold | [baselines.md](baselines.md) |
| **`raw_vlm_single`** | Raw-VLM baseline (single composite) | one forward pass, no scaffold | [baselines.md](baselines.md) |
| **`official_baseline`** | Competition baseline | kit MASTER_PROMPT, verbatim | [baselines.md](baselines.md) |
| **`repl_only`** | Documentation-only ablation (not a paper cell) | REPL + agent loop, no perception | [ablations.md](ablations.md) |
| **`rvlm_unified`** | Category-dispatch ablation | same as `rvlm`, all 8 category tips concatenated | [ablations.md](ablations.md) |

## Architecture quick-look

All paper solvers are **dataset-aware by default**
([D-009](../paper/decisions.md#d-009-refine-d-007--split-semantic-per-profile-from-tool-routing-per-solver)):
they receive a `DatasetProfile` (from
`docvqa.datasets.profile.get_profile(dataset_id)`) that owns the
tool-agnostic semantic content — answer-formatting rules, per-category
semantic tips, per-question format hints, and the scorer. Each solver
file owns:

- its `TASK_INSTRUCTIONS` (the body that documents the tool surface), and
- optionally a per-category `TOOL_HINTS` overlay (tool-routing examples
  for that solver's specific tools — composed on top of the profile's
  semantic tips at runtime).

The split — semantic content per-dataset, tool-routing per-solver — is
the principle the codebase converged on after D-007 / D-009.

## Naming notes

- Engineering solver names are not paper-facing names
  ([D-005](../paper/decisions.md#d-005-position-vs-rvlm-and-madqa),
  [D-010](../paper/decisions.md#d-010-solver-renames--behavior-based-engineering-names)).
- Historical docs in `docs/experiments/*.md` and existing run IDs retain
  the pre-rename names (`leanest-solo-*`, `flat-solo-*`, …).
- The full rename map is in
  [D-010](../paper/decisions.md#d-010-solver-renames--behavior-based-engineering-names).

## See also

- [`docs/paper/README.md`](../paper/README.md) — paper-side taxonomy and
  headline framing
- [`docs/paper/decisions.md`](../paper/decisions.md) — append-only decision
  log
- [`docs/solvers/archive/`](archive/README.md) — shelved-solver docs
