# Archived experiments

Experiments that are NOT part of the paper's evidence chain under the
current D-006 framing. Kept for reproducibility and historical record.

Three reasons to land here:

1. **Process narratives** that won't appear in the paper (e.g., the
   v1/v2 prompt scrub history — D-006 excludes prompt-iteration narrative
   from the paper).
2. **Shelved approaches** that didn't pan out (e.g., the pydantic-ai
   port, the multi-image VLM extension).
3. **Pre-D-010 scaffold experiments** that used the old solver names
   (`flat_solo`, `leanest_solo`) and prompts. Their numerical results
   live in `docs/results.md` (with legacy-name annotations) and in
   submission JSONs; the per-cell writeup here is the historical
   record of how it was run at the time.

If you need the experimental data, it's still in `output/runs/` under
the original run IDs (where present — see `coordination/cleanup-runs.md`
for what was deleted); this folder is the writeup, not the data.

## Index — pre-change cells (invalid numbers, archived 2026-06-01)

These are D-006/D-010-era writeups whose numbers predate the 2026-06-01
code change (prompt scrub to minimized/parity-stripped + whole-agent
`@retry` removed, leaving per-call `num_retries=5` only). **Their numbers
are no longer valid** and are being re-run under current code as the
7-solver matrix (`docs/experiments/{solver}-qwen-3_5-27b.md`). Kept here
for process history only — do not cite as current.

| File | Old-code headline (superseded) | Re-run under |
|---|---|---|
| [react-baseline.md](react-baseline.md) | react n=8 30.47% ± 3.06, Δ=−10.47pp vs rvlm | `react_baseline-qwen-3_5-27b.md` |
| [official-baseline-qwen27b.md](official-baseline-qwen27b.md) | official prompt val n=3 21.67% ± 1.91 | `official_baseline-qwen-3_5-27b.md` |
| [no-loop-multi-image.md](no-loop-multi-image.md) | raw_vlm_multi n=3 tips-on 23.75% | `raw_vlm_multi_baseline-qwen-3_5-27b.md` |
| [no-loop-baseline.md](no-loop-baseline.md) | raw_vlm_single n=3 tips-on 21.25% | (no current single-image cell) |
| [split-calibration-no-loop-multi.md](split-calibration-no-loop-multi.md) | raw_vlm_multi val SC-8 20.0% / test 11.0% (9pp split floor) | pending |
| [unified-category-tips-ablation.md](unified-category-tips-ablation.md) | rvlm_unified n=8 40.94%, Δ=+0.00pp vs rvlm | `rvlm-qwen-3_5-27b.md` (tip-dispatch folded in) |
| [rvlm-minimal-generality.md](rvlm-minimal-generality.md) | rvlm n=8 42.03% ± 2.21 | `rvlm-qwen-3_5-27b.md` |
| [strip-chain-naked-hybrid.md](strip-chain-naked-hybrid.md) | skeletal Δ −1.63 (n.s.) / hybrid Δ −5.31 (sig) | `rvlm_hybrid_ablation-qwen-3_5-27b.md` |
| [direct-vlm-il_n-and-prompt-variance.md](direct-vlm-il_n-and-prompt-variance.md) | direct_vlm il_n sweep ≈35% ± 5pp (parked process log) | `direct_vlm-qwen-3_5-27b.md` |

## Index — process / shelved

| File | Why archived |
|---|---|
| [scrub-audit.md](scrub-audit.md) | v1/v2 prompt-scrub process — won't appear in paper per D-006. Outcome (39.0% test SC-8 for the OCR-free configuration) is captured in `docs/results.md`. |
| [pyai-leanest-solo-da.md](pyai-leanest-solo-da.md) | pydantic-ai-rlm port — single trial underperformed dspy baseline by 8.8pp. Shelved per D-006. |
| [flat-solo-da-multi-image.md](flat-solo-da-multi-image.md) | Multi-image VLM extension — single-trial regression with no clear category lift. Shelved per D-006. |

## Index — pre-D-010 scaffold cells

These document the original `flat_solo` / `leanest_solo` / model-axis /
cross-benchmark cells under their pre-D-010 names. Headline numbers
are mirrored in `docs/results.md` with the legacy→new name mapping.

| File | Headline result |
|---|---|
| [flat-solo-turn-budget-sweep.md](flat-solo-turn-budget-sweep.md) | turn budget {5,10,20,30,40}; m=30 peak 44.69 ± 2.81pp |
| [flat-solo-category-tips-off.md](flat-solo-category-tips-off.md) | tips contribute ~6pp; off → 38.75 ± 3.13pp |
| [flat-solo-vlm-cropping-off.md](flat-solo-vlm-cropping-off.md) | cropping off → 36.88 ± 2.50pp (−7.81pp); active perception matters |
| [flat-solo-search-off.md](flat-solo-search-off.md) | BM25 redundant given `page_texts`; off → 42.50 ± 3.90pp (n.s.) |
| [flat-solo-test-matched-baseline.md](flat-solo-test-matched-baseline.md) | n=8 ICDAR test; SC-8 38.75% |
| [leanest-turn-budget-sweep.md](leanest-turn-budget-sweep.md) | leanest peak m=40 = 43.8% |
| [leanest-ocr-off.md](leanest-ocr-off.md) | OCR off (= leanest) vs flat_solo n=3 comparison |
| [leanest-test-matched-baseline.md](leanest-test-matched-baseline.md) | n=8 ICDAR test; SC-8 36.00% |
| [per-doc-flat-vs-leanest.md](per-doc-flat-vs-leanest.md) | per-doc OCR comparison; long docs benefit, visual docs hurt |
| [efficiency-summary.md](efficiency-summary.md) | cross-cell turns-per-question summary (12 cells) |
| [gemma-4-e4b-baseline-scaffold.md](gemma-4-e4b-baseline-scaffold.md) | model-axis: lift +5.83pp |
| [gemma-4-31b-baseline-scaffold.md](gemma-4-31b-baseline-scaffold.md) | model-axis: lift +25.00pp |
| [qwen-9b-baseline-scaffold.md](qwen-9b-baseline-scaffold.md) | model-axis: lift +6.25pp |
| [mp-docvqa-qwen27b.md](mp-docvqa-qwen27b.md) | cross-benchmark MP-DocVQA; DA pass ≈74% all solvers, +13.68pp on 11-20pp bucket |
| [mmlongbench-doc-qwen27b.md](mmlongbench-doc-qwen27b.md) | cross-benchmark MMLongBench-Doc; +14.81pp judge (rvlm), +16.84pp (rvlm_full kitchen-sink) |
| [rvlm-baseline.md](rvlm-baseline.md) | direct_vlm (was rvlm pre-D-010) smoke; SC-8 chain pending — D-010 naming note in header |

## Pre-D-010 → post-D-010 name mapping

Useful when reading legacy writeups:

| Legacy | Post-D-010 |
|---|---|
| `flat_solo` / `flat_solo_da` | `rvlm_full` (deferred per D-011) |
| `leanest_solo` / `leanest_solo_da` | `rvlm` (proposed method) |
| `no_loop_multi` / `no_loop_multi_da` | `raw_vlm_multi` |
| `no_loop` | `raw_vlm_single` |
| `rvlm` (old) | `direct_vlm` |
| `leanest_ocr` | `rvlm_ocr` |
