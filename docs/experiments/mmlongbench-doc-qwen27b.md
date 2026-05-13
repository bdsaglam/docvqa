# MMLongBench-Doc — Qwen 3.5 27B baseline + scaffold (val, 200Q sample)

**Hypothesis:** The scaffold (active page perception via VLM `look()` +
agent loop) should pay off on a long-document benchmark where the
raw-VLM baseline is forced to truncate. MMLongBench-Doc (Ma et al.,
NeurIPS 2024 D&B Track) averages **47 pages per document** and includes
a designed "Not answerable" subset — both stress active perception over
single-shot consumption.

**Setup:** Qwen 3.5 27B on vllm 8928 (remote, DP=4, prefix caching,
max_model_len 131072). Both LM and VLM = the same endpoint.
`lm.enable_thinking=false`, `max_concurrency=4`. flat_solo cells use OCR
data at `data/mmlongbench-doc/val/ocr/<doc_id>/page_*.md` (811 pages
across 21 docs, generated via `scripts/run_ocr.py` + docling-serve).

**Sample:** 22 docs / 207 questions, stratified across the 5 answer
formats (Int/Float/Str/List/None) seed=42. File:
`data/mmlongbench-doc/val/sample_200q_doc_ids.txt`. One doc
(`mi_phone.pdf`) fails to download from the HF dataset repo and is
silently skipped by the loader, leaving **21 docs / 198 questions
scored**. The 9 lost questions are uniformly distributed across formats
(verified by per-format n=132/123/132/84/132).

**Metric:** Qwen 27B as judge (single-call adaptation of the official
GPT-4o two-stage extraction+score protocol). Judge calibrated to
**11/12 = 91.7%** agreement against hand-marked verdicts before any
cell ran (see `scripts/calibrate_qwen_judge.py`). The runner's ANLS
score is reported alongside as a proxy.

## Commands

```bash
# Baseline (no_loop_multi + tips), 3 trials, then judge
for i in 1 2 3; do
  uv run python evals.py \
    data.dataset=yubo2333/MMLongBench-Doc data.split=val data.num_samples=null \
    data.doc_ids_file=data/mmlongbench-doc/val/sample_200q_doc_ids.txt \
    lm=qwen-3_5-27b-vllm-remote vlm=qwen-3_5-27b-vllm-remote \
    lm.enable_thinking=false solver=no_loop_multi \
    max_concurrency=4 run_id=no-loop-multi-mmlb-remote-t$i
  uv run python scripts/judge_mmlongbench_run.py \
    --run-id no-loop-multi-mmlb-remote-t$i --concurrency 8
done

# Scaffold (leanest_solo), 3 trials, then judge
for i in 1 2 3; do
  uv run python evals.py \
    data.dataset=yubo2333/MMLongBench-Doc data.split=val data.num_samples=null \
    data.doc_ids_file=data/mmlongbench-doc/val/sample_200q_doc_ids.txt \
    lm=qwen-3_5-27b-vllm-remote vlm=qwen-3_5-27b-vllm-remote \
    lm.enable_thinking=false solver=leanest_solo \
    max_concurrency=4 run_id=leanest-solo-mmlb-remote-t$i
  uv run python scripts/judge_mmlongbench_run.py \
    --run-id leanest-solo-mmlb-remote-t$i --concurrency 8
done
```

Chain script: `scripts/run_mmlongbench_chain.sh`.

## Per-trial scores

### Baseline — no_loop_multi + tips (max_pages=10 truncation)

| Trial | run_id | ANLS | Judge | Correct (judge) |
|---|---|---|---|---|
| t1 | `no-loop-multi-mmlb-remote-t1` | 15.15% | 33.33% | 66/198 |
| t2 | `no-loop-multi-mmlb-remote-t2` | 15.15% | 33.33% | 66/198 |
| t3 | `no-loop-multi-mmlb-remote-t3` | 15.66% | 34.85% | 69/198 |

**Mean: ANLS 15.32% ± 0.29pp | Judge 33.84% ± 0.88pp**

### Scaffold (no OCR) — leanest_solo

| Trial | run_id | ANLS | Judge | Correct (judge) |
|---|---|---|---|---|
| t1 | `leanest-solo-mmlb-remote-t1` | 32.83% | 61.62% | 122/198 |
| t2 | `leanest-solo-mmlb-remote-t2` | 29.80% | 58.08% | 115/198 |
| t3 | `leanest-solo-mmlb-remote-t3` | 30.30% | 61.11% | 121/198 |

**Mean: ANLS 30.98% ± 1.62pp | Judge 60.27% ± 1.91pp**

### Scaffold (with OCR) — flat_solo (lean RLM)

| Trial | run_id | ANLS | Judge | Correct (judge) |
|---|---|---|---|---|
| t1 | `flat-solo-mmlb-remote-t1` | 32.32% | 60.61% | 120/198 |
| t2 | `flat-solo-mmlb-remote-t2` | 31.31% | 60.10% | 119/198 |
| t3 | `flat-solo-mmlb-remote-t3` | 30.81% | 60.61% | 120/198 |

**Mean: ANLS 31.48% ± 0.77pp | Judge 60.44% ± 0.29pp**

## Summary

| Configuration | ANLS | Judge | n |
|---|---|---|---|
| no_loop_multi (baseline) | 15.32% ± 0.29pp | 33.84% ± 0.87pp | 3 |
| **leanest_solo (scaffold, no OCR)** | **30.98% ± 1.62pp** | **60.27% ± 1.91pp** | **3** |
| **flat_solo (scaffold + OCR)** | **31.48% ± 0.77pp** | **60.44% ± 0.29pp** | **3** |
| Δ leanest vs baseline | **+15.66pp** (t=16.4) | **+26.43pp** (t=21.8) | — |
| Δ flat_solo vs baseline | +16.16pp (t=27.5) | +26.60pp (t=49.8) | — |
| Δ flat_solo vs leanest | +0.50pp (n.s.) | **+0.17pp (n.s.)** | — |

**The scaffold roughly doubles judge accuracy** (33.84% → 60.27%) on a
200-question sample. Both ANLS and judge agree on the lift direction
and magnitude.

**OCR adds essentially nothing on top of the scaffold** for
MMLongBench-Doc (Δ flat_solo vs leanest +0.17pp judge, n.s.). The
active-perception channel (`look()`) already extracts whatever evidence
the OCR text would provide — the agent zooms into the right page and
reads it directly. This contrasts with MP-DocVQA where OCR *rescued*
the scaffold (+4.23pp lift over leanest). On long-doc benchmarks the
scaffold's primary mechanism is page-routing, not OCR fan-out.

## Per answer-format (judge, summed across 3 trials)

| Format | n/trial | baseline correct/total | scaffold correct/total | base acc | scaf acc | Δ |
|---|---|---|---|---|---|---|
| Float | 44 | 3/132 | 67/132 | 2.3% | **50.8%** | **+48.5pp** |
| Int | 41 | 38/123 | 87/123 | 30.9% | **70.7%** | **+39.8pp** |
| List | 28 | 18/84 | 37/84 | 21.4% | 44.0% | +22.6pp |
| Str | 41 | 59/123 | 79/123 | 48.0% | 64.2% | +16.2pp |
| None | 44 | 83/132 | 88/132 | 62.9% | 66.7% | +3.8pp |

**Numeric formats benefit most.** Float baseline 2.3% → scaffold 50.8%
(+48.5pp) is the single biggest gain — the raw VLM is essentially
unable to extract precise numbers from long financial reports under a
10-page truncation, but the scaffold's `look()` lets it zoom into the
right page and read off the figure. Int +39.8pp follows the same
pattern. "Not answerable" (None) is already strong for the baseline
(62.9%) — refusing to answer doesn't need active perception — and the
scaffold's lift here is small (+3.8pp).

## Per document-type (judge, summed across 3 trials)

| Category | n/trial | base correct | base acc | scaf correct | scaf acc | Δ |
|---|---|---|---|---|---|---|
| Financial report | 48 | 9/144 | 6.2% | 78/144 | **54.2%** | **+47.9pp** |
| Administration/Industry file | 31 | 40/93 | 43.0% | 71/93 | 76.3% | **+33.3pp** |
| Brochure | 14 | 18/42 | 42.9% | 29/42 | 69.0% | +26.2pp |
| Guidebook | 12 | 28/36 | 77.8% | 34/36 | **94.4%** | +16.7pp |
| Research report / Introduction | 44 | 39/132 | 29.5% | 61/132 | 46.2% | +16.7pp |
| Tutorial/Workshop | 14 | 18/42 | 42.9% | 25/42 | 59.5% | +16.7pp |
| Academic paper | 35 | 49/105 | 46.7% | 60/105 | 57.1% | +10.5pp |

**The Financial-report category is the most discriminating.** The
no_loop_multi baseline at 6.2% on 144 financial-report questions is
essentially failing — these are the longest docs (Amazon/Costco/BestBuy
10-Ks at 100+ pages each), and a 10-page truncation cap kills the
baseline. The scaffold climbs to 54.2%, an 8.7× relative lift.

Guidebooks (most concrete, well-structured) are easy for both — but
the scaffold still adds +16.7pp on top.

The scaffold lifts every category by ≥10pp.

## Comparison across the §B picks

| Benchmark | Doc length | Baseline (no_loop) | Scaffold (leanest) | Lift |
|---|---|---|---|---|
| DocVQA-2026 val | 1–181pp, varied | ~37.5–44.7% | 43.8–51.2% | +6–7pp (matched) |
| MP-DocVQA val | 1–20pp, mostly ≤5pp | 63.74% | 58.86% | **−4.88pp** |
| **MMLongBench-Doc val** | **47pp avg** | **33.84% (judge)** | **60.27% (judge)** | **+26.43pp** |

**The scaffold's lift scales with effective document length, not
benchmark identity.** MP-DocVQA's short docs make the raw VLM the
strong baseline; MMLongBench-Doc's long docs make the scaffold's
active perception essential. The §B claim "scaffold generalizes
beyond DocVQA-2026" *holds* — but the magnitude depends on the
length distribution. We should report both signs.

## Wall time

| Variant | Mean wall (per trial, max_concurrency=4) |
|---|---|
| baseline (no_loop_multi) | ~76 min |
| scaffold (leanest_solo) | ~171 min |

Scaffold is ~2.3× slower per trial on this benchmark (long docs +
multi-iteration look loops). Total chain: 3×76 + 3×171 ≈ 12.4h wall.

## Variance

- Baseline std: ANLS 0.29pp, Judge 0.87pp — near-deterministic.
- Scaffold std: ANLS 1.62pp, Judge 1.91pp — modest variance.
- All 3 scaffold trials beat all 3 baseline trials by ≥23pp (judge).
  The sign is robust.

## Notes / caveats

- **1 doc (mi_phone.pdf) missing.** `huggingface_hub` cannot fetch
  this file from the dataset repo — likely a 404. The loader logs
  the failure and continues; 9 questions are lost. 198/207 = 96% of
  the sample is scored, all 5 formats remain represented.
- **Page cap at 80 pages.** The MMLongBench loader caps each PDF
  at 80 rendered pages (default `max_pages=80`). For the longest
  docs (Costco 10-K, BestBuy 10-K, ~150pp) this excludes evidence
  pages beyond page 80. A ceiling pass at `max_pages=200` is on the
  todo list for the final paper writeup; current numbers are
  conservative for these docs.
- **Judge calibration.** Done before any cell ran (11/12 = 91.7%
  agreement). The single miss was a Float ~1% tolerance edge case
  (0.123 vs 0.124) which is unlikely to bias the overall numbers.
- **ANLS-as-proxy fallback.** Both metrics are reported; if the
  judge is later found to be miscalibrated on this distribution,
  the ANLS numbers stand on their own (still show the +15.66pp lift).
- **One transient identical-accuracy aggregation.** t1 and t2 of
  the baseline both happen to score 30/198 ANLS and 66/198 judge —
  this is genuine: per-question predictions differ (verified
  via diff) but counts coincide.

## Status

Done. 3 baseline + 3 leanest_solo + 3 flat_solo trials = 9 cells, all
scored with both ANLS and the Qwen judge. Headlines:

- **Scaffold lifts MMLongBench-Doc by +26.43pp (judge)** at this model
  size (t=21.77, highly significant) — the largest scaffold lift we've
  measured anywhere. Confirms §B for long-document VQA.
- **OCR does not add to the scaffold here** (Δ flat_solo vs leanest
  +0.17pp judge, n.s.). The scaffold's `look()` channel already
  captures the page evidence that OCR would.
- Combined with MP-DocVQA (where the scaffold *regresses* on short
  docs but OCR rescues it), the picture is: **OCR helps when active
  perception isn't useful (short docs); active perception subsumes
  OCR when docs are long enough to need page routing.**
