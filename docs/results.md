# Experiment Results — DocVQA 2026 (post-D-006)

Paper-relevant results under the D-006 framing
(visual-context-budget hypothesis). All scores are **mean ± std across
trials** (no SC voting in the headline per D-003 — SC-8 numbers are
shown only where they anchor an ICDAR submission).

> **Historical results** from the pre-pivot era live in
> [`archive/docs/results.md`](../archive/docs/results.md). Read for
> context; do not cite directly without checking that the cell is still
> relevant under D-006/D-010. Old solver names there (`flat_solo`,
> `leanest_solo`, etc.) map to new names per D-010.

## Solver taxonomy (engineering names)

| Name | Legacy name (pre-D-010) | Role |
|---|---|---|
| `rvlm` | `leanest_solo` | proposed method — REPL + recursive VLM `batch_look` sub-call |
| `rvlm_ocr` | — (new, no legacy predecessor) | +OCR extension (clean; no `look()` confound) |
| `rvlm_full` | `flat_solo` / `flat_solo_da` | kitchen-sink (rvlm + `look()` + OCR); **deferred per D-011**, appendix-only |
| `rvlm_unified` | — (new ablation) | rvlm with all 8 category tips concatenated |
| `direct_vlm` | `rvlm` (the *old* rvlm; semantic shift per D-010) | single multimodal LLM with `display()` (alt angle) |
| `raw_vlm_multi` | `no_loop_multi` / `no_loop_multi_da` | raw-VLM baseline, multi-image |
| `raw_vlm_single` | `no_loop` | raw-VLM baseline, single-image |
| `repl_only` | — (new ablation) | documentation: REPL with no perception |

See `docs/paper/README.md` for paper-cell taxonomy and full names.
Paper-facing method names TBD per D-005. Legacy names appear in
historical run IDs (`leanest-*`, `flat-solo-*`, `no-loop-*`) and in
`archive/docs/results.md`; D-010 doesn't backfill those.

## Official baselines (ICDAR 2026 — for context)

| Model | Val | Test |
|---|---|---|
| Gemini 3 Pro | 37.50% | **37.50%** |
| GPT-5.2 | — | 35.00% |
| Gemini 3 Flash | 33.75% | 33.75% |
| GPT-5 Mini | — | 22.50% |

## Headline cells — Qwen 3.5 27B (post-D-009 clean prompts)

| Cell | Solver | n | Val per-trial | Val SC-8 | Test SC-8 | Notes |
|---|---|---|---|---|---|---|
| **rvlm post-scrub** | rvlm | 8 | 42.81 ± 4.42pp | **48.8%** | **39.0%** | proposed method, paper headline |
| rvlm_full post-scrub | rvlm_full | 8 | 42.35 ± 3.23pp | 47.5% | 38.0% | **deferred per D-011**; existing data stays as footnote |
| raw_vlm_multi (baseline) | raw_vlm_multi | 8 | 21.07 ± 1.81pp | 20.0% | **11.0%** | split-difficulty anchor |

**Val→test gap:** rvlm 9.8pp, raw_vlm_multi 9.0pp. The 9pp baseline gap
is the **split-difficulty floor**; scrubbed rvlm sits at the floor (no
measurable generalization gap remaining). See
`docs/experiments/split-calibration-no-loop-multi.md`.

## In-flight cells (per coordination/amax7.md, coordination/amax1.md)

| Cell | Solver | Host | Status | Notes |
|---|---|---|---|---|
| `rvlm-unified-val-t1` | rvlm_unified | amax7 | running | task #25 — promotes unified to default if Δ ≈ 0pp |
| `rvlm-ocr-val-t1` | rvlm_ocr | amax7 | queued | task #14 |
| `direct-vlm-val-t1` | direct_vlm | amax7 | queued | task #19 |
| Gemma E4B baseline+rvlm | rvlm | amax1 | queued | task #8 part 1 |
| Qwen 9B baseline+rvlm | rvlm | amax1 | queued | task #8 part 2 |
| Gemma 31B baseline+rvlm | rvlm | amax1 | queued | task #8 part 3 |

## Model-axis (prediction 1)

n=3 cells from 2026-05-09/10 used **pre-scrub** CATEGORY_TIPS — re-runs
in flight on amax1 with clean prompts (task #8). Direction is robust;
absolute magnitudes may shift 2–4pp.

| Model | Tier | Baseline (`raw_vlm_multi`) | Scaffold (legacy `flat_solo`) | Lift |
|---|---|---|---|---|
| Gemma 4 E4B-it | ≤8B | 3.75 ± 0.00pp | 9.58 ± 1.44pp | +5.83pp |
| Qwen 3.5 9B | ≤8B | 15.00 ± 1.25pp | 21.25 ± 2.50pp | +6.25pp |
| Gemma 4 31B-it | 8–35B | 10.42 ± 0.72pp | 35.42 ± 5.20pp | +25.00pp |
| Qwen 3.5 27B | 8–35B | 21.07 ± 1.81pp | 42.81 ± 4.42pp | +21.74pp |

Detail per cell:
- `docs/experiments/gemma-4-e4b-baseline-scaffold.md`
- `docs/experiments/qwen-9b-baseline-scaffold.md`
- `docs/experiments/gemma-4-31b-baseline-scaffold.md`

## Document-length axis (prediction 2)

200Q val samples, Qwen 3.5 27B with DA profiles, n=3 each.

| Benchmark | Avg pages | Solver | Score | Δ vs baseline |
|---|---|---|---|---|
| MP-DocVQA | 1–20 (67% ≤5p) | raw_vlm_multi (DA) | 74.15 ± 0.84% ANLS | — |
| MP-DocVQA | | rvlm (no OCR) | 72.52 ± 2.45% | −1.63 (n.s.) |
| MP-DocVQA 11-20pp bucket | | rvlm_full (legacy +OCR) | — | **+13.68pp** on bucket |
| MMLongBench-Doc | 47 avg | raw_vlm_multi (pages=80) | 46.97 ± 0.51% judge | — |
| MMLongBench-Doc | | rvlm (no OCR) | 61.78 ± 1.17% | **+14.81pp** |
| MMLongBench-Doc | | rvlm_full (legacy +OCR) | 63.81 ± 0.76% | +16.84pp |
| MMLongBench-Doc | | rvlm_ocr (clean) | TBD | task #15 |

The legacy OCR cells used `flat_solo_da` which bundles `look()` with OCR.
The clean `rvlm_ocr` cell (task #15) is queued on amax7 after task #14.

Detail:
- `docs/experiments/mp-docvqa-qwen27b.md`
- `docs/experiments/mmlongbench-doc-qwen27b.md`

## Active-perception mechanism (prediction 3)

Reframed from "REPL-only collapse" to "active-perception" — triangulated
by three independent ablations, all measured.

| Component | Ablation | n | Result | Reading |
|---|---|---|---|---|
| Region selection (cropping) | `flat_solo` cropping-off | 8 | 36.88 ± 2.50% | **−7.81pp** vs 44.69%; active region selection matters |
| Iteration count | turn budget m=5 | 3 | 30.00 ± 0.00% | **−14.69pp** vs m=30; iterative, not one-shot |
| Recursive sub-call structure | `rvlm` vs `raw_vlm_multi` | 8 vs 8 | 42.81% vs 21.07% (per-trial) | **+21.74pp**; recursive agent↔VLM dominates one-shot |

Per-component sanity:
- `flat_solo` no-search: 42.50 ± 3.90% (−2.19pp, n.s. — BM25 redundant given `page_texts`)
- `flat_solo` no-tips: 38.75 ± 3.13% (−5.94pp — tips contribute)
- `repl_only` (VLM-off smoke): 0/5 on 2 docs — predicted full collapse, not a paper cell

Detail:
- `docs/experiments/flat-solo-vlm-cropping-off.md`
- `docs/experiments/flat-solo-turn-budget-sweep.md`
- `docs/experiments/leanest-turn-budget-sweep.md`
- `docs/experiments/flat-solo-search-off.md`
- `docs/experiments/flat-solo-category-tips-off.md`
- `docs/experiments/efficiency-summary.md`

## Conventions for adding rows

When a new cell completes:

1. Update the **In-flight cells** table — move the row to the appropriate
   section (headline, model-axis, doc-length, or mechanism).
2. Cite the per-cell experiment doc under `docs/experiments/`.
3. If the result triggers a paper-framing decision (e.g., unified-tips
   Δ ≈ 0pp → promote to default), add a `decisions.md` D-NNN entry too.
4. Per D-008: n=1 cells annotate as "n=1, n=2 if direction holds." Don't
   table-headline an n=1 number — flag the trial count.
5. Update `coordination/<host>.md` to mark the cell `[✓]` with the
   run_id and the one-line result.
