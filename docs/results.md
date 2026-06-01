# Experiment Results — DocVQA 2026 (post-D-006)

Cross-axis summary under the D-006 framing (visual-context-budget
hypothesis). Scores are **mean ± std across trials** (no SC voting in the
headline per D-003 — SC-8 numbers shown only where they anchor an ICDAR
submission). Per-cell detail lives in `docs/experiments/{solver}-{model}.md`.

> **⚠ All current numbers are from the post-2026-06-01 code** (minimized /
> parity-stripped prompts + per-call `num_retries=5` only; whole-agent
> `@retry` removed). **Pre-change numbers are no longer valid** — the
> prompt scrub and retry-logic change moved them; their writeups are in
> [`archive/experiments/`](../archive/experiments/) and
> [`archive/docs/results.md`](../archive/docs/results.md). Do not cite
> archived numbers as current. The 7-solver Qwen-27B re-run below is
> **in progress** (n=3 target; most cells n=2); numbers lock at n=3.

## Official baselines (ICDAR 2026 — external, for context)

| Model | Val | Test |
|---|---|---|
| Gemini 3 Pro | 37.50% | **37.50%** |
| GPT-5.2 | — | 35.00% |
| Gemini 3 Flash | 33.75% | 33.75% |
| GPT-5 Mini | — | 22.50% |

## Method vs baselines — Qwen 3.5 27B, val (current code, in progress)

7-solver comparison re-run (val 25 docs / 80 Qs, `enable_thinking=false`,
local vllm :8927, n=3 target). All Δ measured vs the `rvlm` reference.

| Solver | Role | Val (current) | n | Status |
|---|---|---|---|---|
| **`rvlm`** | **proposed method** — REPL + recursive VLM `batch_look` (OCR-free) | **~39.4%** (40.00, 38.75) | 2/3 | reference |
| `rvlm_hybrid_ablation` | + direct `display()` channel on top of sub-call | ~39.4% (40.00, 38.75) | 2/3 | Δ ≈ 0 (redundant) |
| `rvlm_ocr_ablation` | + OCR `page_texts` + BM25 `search` | **37.08% ± 2.60** | 3/3 ✓ | Δ ≈ −2 (OCR adds nothing) |
| `direct_vlm` | `display()` pages into own context, no sub-call | 21.25% | 1/3 | sub-call ≈ +18.75pp |
| `raw_vlm_multi_baseline` | raw multi-image, no scaffold | ~20.6% (18.75, 22.50) | 2/3 | scaffold floor |
| `react_baseline` | perception, no REPL | ~23.8% (17.50, 30.00) | 2/3 | high variance |
| `rlm_ocr` | RLM + OCR text, no vision | — | 0/3 | queued |
| `official_baseline` | competition `MASTER_PROMPT`, no scaffold | — | 0/3 | queued (prior kit-faithful 21.67% ± 1.91) |

Detail: `docs/experiments/{solver}-qwen-3_5-27b.md` for each row.

## Active-perception mechanism (prediction 3)

The matrix above triangulates the mechanism — both halves of the scaffold
are load-bearing and the recursive sub-call is the active ingredient:

| Component dropped | Solver | vs `rvlm` (~39.4%) | Reading |
|---|---|---|---|
| Recursive sub-call | `raw_vlm_multi_baseline` (~20.6%) | **≈ +21pp** | recursive agent↔VLM dominates one-shot multi-image |
| The REPL | `react_baseline` (~23.8%) | **≈ +16pp** | code REPL is load-bearing (crop/arith/compose) |
| Sub-call (kept pixels) | `direct_vlm` (21.25%) | **≈ +18pp** | raw pixels in-context ≠ a focused VLM sub-call |

Dropping either half of the scaffold collapses the score: perception
served one-shot instead of via the recursive sub-call (`raw_vlm_multi`,
`direct_vlm`) and perception-without-REPL (`react`) both fall well below
`rvlm`. Adding OCR (`rvlm_ocr`) or a direct image channel (`rvlm_hybrid`)
on top of the OCR-free sub-call buys ≈ 0 → supports the OCR-free
recursive-perception framing.

## Model-size / VLM-quality axis (prediction 1)

Hold the reasoner fixed, swap **only** the VLM tool backend. n=8 per arm,
val, current code. Detail:
`docs/experiments/qwen-9b-rvlm-minimal-vlm-axis.md`.

| Reasoner (LLM) | v1 homog (VLM = LLM) | v2 mixed (VLM = 27B) | Δ (v2 − v1) | n |
|---|---|---|---|---|
| Qwen 3.5 9B | 16.67% ± 3.40 | 24.54% ± 5.30 | **+7.87pp** — Welch t=3.54, 95% CI [+3.4, +12.3], **sig.** | 8 |
| Qwen 3.5 4B | _running (Phase 2b)_ | 21.09% ± 3.16 | _pending v1_ | 8 |
| Qwen3 8B (text-only LLM) | — | _queued (Phase 3, v2 only)_ | — | — |

At 9B, swapping only the VLM 9B→27B with the reasoner fixed lifts ~8pp →
the scaffold is **perception-budget-bound** for a mid/small reasoner
(supports D-006). The baseline-vs-scaffold model-axis sweep (Gemma E4B /
Gemma 31B / Qwen 27B on clean prompts) is queued on amax1 — numbers TBD.

## Document-length axis (prediction 2)

> Prior MP-DocVQA / MMLongBench-Doc cross-benchmark numbers were run on
> pre-change prompts/retry logic and are **invalid** (archived under
> `archive/experiments/mp-docvqa-qwen27b.md`,
> `mmlongbench-doc-qwen27b.md`). The mechanism (lift sign + magnitude
> scale with the benchmark's page-count distribution) is robust, but the
> magnitudes need a current-code re-run before citing. **Pending.**

## Solver taxonomy (engineering names)

| Name | Role |
|---|---|
| `rvlm` | proposed method — REPL + recursive VLM `batch_look` (OCR-free) |
| `rvlm_ocr_ablation` | + OCR `page_texts` + BM25 search (OCR extension) |
| `rvlm_hybrid_ablation` | + direct `display()` image channel on top of the sub-call |
| `direct_vlm` | single multimodal LLM with `display()`, no sub-call (alt angle) |
| `raw_vlm_multi_baseline` | raw multi-image, single VLM call, no REPL |
| `react_baseline` | `dspy.ReAct` + same VLM tools as `rvlm`, no Python REPL |
| `rlm_ocr` | REPL + OCR text + BM25, no vision (text-perception variant) |
| `official_baseline` | competition `MASTER_PROMPT`, multi-image, no scaffold |

Legacy pre-D-010 names (`flat_solo`, `leanest_solo`, `no_loop_multi`,
etc.) appear only in historical run IDs and `archive/`; D-010 doesn't
backfill them.

## Conventions for adding rows

1. When a cell reaches n=3 (or n=8), update its `docs/experiments/`
   file's Summary + Status, then refresh the matrix row here with the
   locked mean ± std.
2. Per D-008: flag the trial count on every number; don't headline an
   n=1 as if locked.
3. If a result triggers a paper-framing decision, add a `decisions.md`
   D-NNN entry.
4. Mark the cell `[✓]` in `coordination/<host>.md` with the run_id and
   one-line result.
