# MP-DocVQA — Qwen 3.5 27B baseline + scaffold (val, 200Q sample)

**Hypothesis:** The scaffold (active page perception via VLM `look()` +
agent loop) should generalize beyond DocVQA-2026 to another multi-page
DocVQA benchmark. MP-DocVQA (Tito et al., Pattern Recognition 2023)
gives a clean ANLS-on-multi-page anchor outside the DocVQA-2026 pool.

**Setup:** Qwen 3.5 27B on vllm 8927 (local A100×4, DP=4, prefix caching,
max_model_len 131072). Both LM and VLM = the same endpoint.
`lm.enable_thinking=false`, `max_concurrency=8`. OCR data at
`data/mp-docvqa/val/ocr/<doc_id>/page_*.md` (202 pages across 39 docs,
generated via `scripts/run_ocr.py` + docling-serve; picture-description
disabled because of a model-path bug in the current docling-serve build).

**Sample:** 39 docs / 205 questions, stratified by page count
proportionally to the val distribution (seed=42). Bucketing:
1pp=12 docs/70Q, 2-5pp=15/66, 6-10pp=5/30, 11-20pp=7/39.
File: `data/mp-docvqa/val/sample_200q_doc_ids.txt`.

**Metric:** ANLS (≥0.9 relaxed-text threshold; strict numeric/date
fall-through). Same `evaluate_prediction` path as DocVQA-2026.

## Commands

```bash
# Baseline (no_loop_multi + tips), 3 trials
for i in 1 2 3; do
  uv run python evals.py \
    data.dataset=lmms-lab/MP-DocVQA data.split=val data.num_samples=null \
    data.doc_ids_file=data/mp-docvqa/val/sample_200q_doc_ids.txt \
    lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
    lm.enable_thinking=false solver=no_loop_multi \
    max_concurrency=8 run_id=no-loop-multi-mpdv-local-t$i
done

# Scaffold (leanest_solo), 3 trials
for i in 1 2 3; do
  uv run python evals.py \
    data.dataset=lmms-lab/MP-DocVQA data.split=val data.num_samples=null \
    data.doc_ids_file=data/mp-docvqa/val/sample_200q_doc_ids.txt \
    lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
    lm.enable_thinking=false solver=leanest_solo \
    max_concurrency=8 run_id=leanest-solo-mpdv-local-t$i
done
```

Chain script: `scripts/run_mp_docvqa_chain.sh`. Log:
`/tmp/docvqa-bench-logs/mpdv-local.log`.

## Per-trial scores

### Baseline — no_loop_multi + tips

| Trial | run_id | Score | Correct |
|---|---|---|---|
| t1 | `no-loop-multi-mpdv-local-t1` | 63.41% | 130/205 |
| t2 | `no-loop-multi-mpdv-local-t2` | 63.90% | 131/205 |
| t3 | `no-loop-multi-mpdv-local-t3` | 63.90% | 131/205 |

**Mean: 63.74% ± 0.28pp | range 63.41–63.90%**

### Scaffold (no OCR) — leanest_solo

| Trial | run_id | Score | Correct |
|---|---|---|---|
| t1 | `leanest-solo-mpdv-local-t1` | 58.05% | 119/205 |
| t2 | `leanest-solo-mpdv-local-t2` | 57.07% | 117/205 |
| t3 | `leanest-solo-mpdv-local-t3` | 61.46% | 126/205 |

**Mean: 58.86% ± 2.31pp | range 57.07–61.46%**

### Scaffold (with OCR) — flat_solo (lean RLM)

| Trial | run_id | Score | Correct |
|---|---|---|---|
| t1 | `flat-solo-mpdv-local-t1` | 63.41% | 130/205 |
| t2 | `flat-solo-mpdv-local-t2` | 63.90% | 131/205 |
| t3 | `flat-solo-mpdv-local-t3` | 61.95% | 127/205 |

**Mean: 63.09% ± 0.97pp | range 61.95–63.90%**

## Summary

| Configuration | Mean | Std | n |
|---|---|---|---|
| no_loop_multi + tips (baseline) | 63.74% | 0.28pp | 3 |
| **leanest_solo (scaffold, no OCR)** | **58.86%** | **2.31pp** | **3** |
| **flat_solo (scaffold + OCR)** | **63.09%** | **0.97pp** | **3** |
| Δ leanest vs baseline | **−4.88pp** (SE 1.34, t=−3.64, **significant regression**) | — | — |
| Δ flat_solo vs baseline | **−0.65pp** (SE 0.59, t=−1.10, n.s.) | — | — |
| Δ flat_solo vs leanest | **+4.23pp** (SE 1.45, t=2.92, significant) | — | — |

**OCR rescues the scaffold on MP-DocVQA.** Without OCR (leanest_solo)
the scaffold regresses 4.88pp; with OCR (flat_solo) the scaffold
matches the raw-VLM baseline (n.s. gap). The OCR channel adds +4.23pp
on top of leanest — exactly the "OCR-as-stability-anchor" channel from
DocVQA-2026, and consistent with the per-doc OCR analysis showing OCR
helps short docs via better tokenized search even when active page
perception isn't useful.

## Per page-bucket breakdown (mean over 3 trials)

| Bucket | n_q | base t1/t2/t3 | scaf t1/t2/t3 | base mean | scaf mean | Δ |
|---|---|---|---|---|---|---|
| 1pp | 70 | 37/37/35 | 31/25/29 | **51.9%** | **40.5%** | **−11.4pp** |
| 2-5pp | 66 | 43/44/45 | 40/42/45 | 66.7% | 64.1% | −2.5pp |
| 6-10pp | 30 | 23/23/24 | 20/21/24 | 77.8% | 72.2% | −5.6pp |
| 11-20pp | 39 | 27/27/27 | 28/29/28 | 69.2% | **72.6%** | **+3.4pp** |

**The scaffold's sign flips with page count.**

- **1pp (70Q, 34% of sample):** scaffold drops 11.4pp. The agent loop
  has no headroom on single-page docs — the raw VLM already sees the
  whole document in one shot. Iterations + tool calls + code-write
  introduce error opportunities (formatting drift, off-target crops,
  hallucinated rationales) that the baseline cleanly avoids.
- **2-5pp and 6-10pp (47% of sample):** scaffold loses 2.5–5.6pp. Same
  failure mode but smaller magnitude.
- **11-20pp (39Q, 19% of sample):** scaffold wins +3.4pp. Active page
  perception finally pays off when a 10-page truncation cap starts
  biting the baseline. This matches the DocVQA-2026 story for the
  long-doc categories (business_report etc.).

The pooled headline (−4.88pp) is mostly a function of MP-DocVQA's
short-doc dominance, not a uniform scaffold weakness.

## Variance

- Baseline std 0.28pp — extremely tight across trials.
- Scaffold std 2.31pp — ~8× higher. t3 (61.5%) is +4.4pp over t2
  (57.1%); the agent path is genuinely stochastic at this length.
- Even scaffold's best trial (t3, 61.5%) does not beat baseline's
  worst (t1, 63.4%). The regression sign is robust to trial noise.

## Wall time

| Variant | Total elapsed / max_concurrency (per trial) |
|---|---|
| baseline (no_loop_multi) | ~24-45 min |
| scaffold (leanest_solo) | ~16-20 min |

Scaffold is actually *faster* than baseline here because the per-doc
parallelism (8 docs in flight) interacts well with the doc-level
question concurrency in `leanest_solo` (batch=8), and short docs
finish fast. The baseline pays more time per question on multi-image
VLM calls. So scaffold is cheaper *and* worse on this benchmark.

## Comparison vs DocVQA-2026 (cross-benchmark generality check)

| Benchmark | Baseline (no_loop_multi) | Scaffold (leanest/flat_solo) | Δ |
|---|---|---|---|
| DocVQA-2026 val | ~37.5% (flat_batch) / ~44.7% (no_loop) | 43.8% (leanest) / 51.2% (flat_solo SC-8) | **positive on long-doc-heavy cats** |
| MP-DocVQA val (200Q) | 63.74% | 58.86% | **−4.88pp** |

**Interpretation.** The §B generality claim does NOT cleanly transfer:
on MP-DocVQA the scaffold is a net regression. The mechanism is
visible in the page-bucket cut — the scaffold's win on 11-20pp docs
(+3.4pp) is real and consistent with DocVQA-2026, but the regression
on 1pp docs (−11.4pp) — which dominate MP-DocVQA's val sample — drags
the headline negative. MP-DocVQA's distribution is much shorter than
DocVQA-2026's (where comics 36-69pp, business_report 105-181pp, etc.
are common).

In §B of the experiment plan we should report:
1. Headline regression (−4.88pp, t=−3.64) on the pooled metric.
2. Page-bucket cut showing scaffold *wins* on 11-20pp docs and *loses*
   on 1pp docs.
3. Conclusion: scaffold lift is *length-dependent*, not benchmark-
   dependent. The §B claim should be rephrased as "scaffold lift
   scales with effective document length, not benchmark identity."

## Observations

- **1pp regression dominates the headline.** 70/205 questions are
  single-page. Baseline gets 51.9% of them, scaffold 40.5%. Removing
  the 1pp bucket (re-pooling at 135 questions), the gap shrinks to
  approximately baseline 70.4% / scaffold 68.2% — about −2pp. So most
  of the "bad" headline is concentrated in one bucket.
- **Scaffold lift on 11-20pp is real.** +3.4pp on 39 questions. Small
  sample, but the sign is consistent across all 3 trials
  (scaffold > baseline on this bucket in every trial).
- **Trial-to-trial variance: scaffold ≫ baseline.** Baseline is
  near-deterministic on MP-DocVQA (σ=0.28pp); the scaffold's agent
  trajectory is genuinely stochastic (σ=2.31pp).
- **No sandbox subprocess errors across all 6 trials.**

## Status

Done. 3 baseline + 3 leanest_solo + 3 flat_solo trials = 9 cells.
Sample, loader, dispatch, chain scripts, and OCR data all committed.

**Headlines:**
- Pooled scaffold (leanest, no OCR) **regresses −4.88pp** — concentrated
  on 1pp docs (−11.4pp); scaffold *wins* +3.4pp on the 11-20pp bucket.
- flat_solo (scaffold + OCR) matches the baseline (n.s.) — OCR rescues
  +4.23pp on top of leanest. Confirms OCR-as-stability-anchor.

## Dataset-aware re-run (2026-05-13/14)

The original chains above all used the DocVQA-2026 default prompt
(`ANSWER_FORMATTING_RULES` + DocVQA-2026 category tips). Those rules
strip commas from numbers, normalize dates to ISO, etc., which silently
mangled MP-DocVQA's literal-span answers. After rebuilding each solver
with a per-dataset profile (`MP_DOCVQA_PROFILE`: preserve document's
own number/currency/date representation, no category tips) and an
ANLS scorer, all three solvers were re-run on the same 200Q sample.

### Per-trial scores (DA prompts)

| Solver | t1 | t2 | t3 | Mean ± Std |
|---|---|---|---|---|
| `no_loop_multi_da` | 73.66 | 75.12 | 73.66 | **74.15% ± 0.84pp** |
| `leanest_solo_da` | 75.12 | 72.20 | 70.24 | **72.52% ± 2.45pp** |
| `flat_solo_da` | 73.66 | 70.73 | 75.12 | **73.17% ± 2.30pp** |

### Old prompt vs DA prompt (mean)

| Solver | Legacy prompt | DA prompt | Δ |
|---|---|---|---|
| no_loop_multi | 63.74% ± 0.28pp | **74.15% ± 0.84pp** | **+10.41pp** |
| leanest_solo | 58.86% ± 2.31pp | **72.52% ± 2.45pp** | **+13.66pp** |
| flat_solo | 63.09% ± 0.97pp | **73.17% ± 2.30pp** | **+10.08pp** |

The DA prompt unlocks ~10–14pp on every solver. The DocVQA-2026
formatting rules were the limiting factor on MP-DocVQA, not the
scaffold's reasoning capability.

### Closed-loop scaffold vs baseline (DA prompts)

| Comparison | Δ | Welch t | Significance |
|---|---|---|---|
| `leanest_solo_da` vs `no_loop_multi_da` | −1.63pp | t≈−1.1 | n.s. |
| `flat_solo_da` vs `no_loop_multi_da` | −0.98pp | t≈−0.7 | n.s. |
| `flat_solo_da` vs `leanest_solo_da` (OCR effect) | +0.65pp | t≈0.3 | n.s. |

**On MP-DocVQA, neither the scaffold nor OCR helps once the baseline
has a fair prompt.** The "scaffold regression" and "OCR rescues
scaffold" stories from the legacy run were both prompt-mismatch
artifacts, equally available to the baseline.

This is consistent with the mechanism described in
[[feedback_scaffold_lift_scales_with_doc_length]]: MP-DocVQA's short
docs (67% ≤5pp) don't exercise the scaffold's value props (page
routing, active perception, iterative reasoning). When the baseline
already sees the whole doc in one shot, the scaffold's iteration
overhead can't pay for itself.

### Variance

- Baseline std 0.84pp — near-deterministic.
- Scaffold stds 2.30pp / 2.45pp — ~3× higher. Three flat_solo_da
  trials span 70.7-75.1%; two leanest_solo_da trials underperformed
  every baseline trial. Even when expected accuracy matches the
  baseline, individual trials spread further.
