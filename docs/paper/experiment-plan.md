# Experiment Plan — DocVQA Paper (post-D-006 pivot)

> **2026-05-27 restructure.** This file was significantly reframed per
> D-006 (visual-context-budget hypothesis). Prior versions framed the
> work as a "competition scaffold." See `docs/paper/decisions.md` D-006
> through D-008 for the decision chain. Section §A (model-size axis)
> and §B (document-length axis) keep their empirical tables but the
> narrative around them is now hypothesis-shaped. Section §C
> reorganized around the proposed method (`leanest_solo`, OCR-free) as
> the headline configuration. Some sections still carry legacy framing
> language pending cleanup.

## Headline claim (D-006)

**Mid-sized open VLMs are bottlenecked by visual context budget on
document VQA, not by reasoning capacity.** Recursive perception
(RLM with VLM sub-call) gives the main agent the ability to ration its
high-resolution VLM calls per question, lifting accuracy to
frontier-level without scaling model size or training data.

Three falsifiable predictions support the hypothesis:

1. **Model-size axis (§A).** Lift scales with model code-writing
   capability — the recursive sub-call is invoked via code, so models
   that write better code extract more.
2. **Document-length axis (§B).** Lift scales with effective document
   length — longer docs exhaust the raw VLM's visual budget; the
   recursive sub-call sidesteps it.
3. **Mechanism axis (§C).** Removing the recursive VLM sub-call (but
   keeping the REPL + agent loop) collapses the lift back to baseline.

The proposed method is the **OCR-free** variant (current engineering
name: `leanest_solo`; paper-facing name TBD). OCR/search is reported as
an **extension** (currently a new fork of leanest must be built — the
legacy `flat_solo` confounds OCR with a `look()` ergonomic wrapper and
is not the clean extension cell).

Out of scope for the paper (per D-006):

- SC voting framing (already D-003)
- `FLAT_SOLO_TOOL_HINTS` / "v2" overlay history
- Multi-image extension (`flat_solo_da_mi`) — shelved
- GEPA-based prompt optimization — shelved
- pyai port — shelved
- Per-category tips presented as a method contribution (kept as an
  ablation row only)
- Engineering solver names (`leanest_solo`, `flat_solo`, `rvlm`) — paper
  uses different names selected later

## Variance discipline + escalation policy (D-008)

- **Trial-budget escalation:** new cells start at n=1, escalate to n=2
  if the n=1 direction holds, escalate to n=8 only after the paper
  headline framing locks. Avoids burning compute through pivots.
- **Variance reporting:** ≥3 trials per headline number, mean ± std.
  No single-trial headlines (Gemini Pro/Flash single-trial numbers are
  reported with the credits-exhausted caveat per D-006).
- **No SC voting in headline numbers** (D-003) — SC-8 lives in
  appendix-only contexts as a competition tactic.
- Cells already at n=8 with clean prompts stay at n=8 (Qwen 27B
  leanest scrubbed, no_loop_multi split-calibration, MMLongBench-Doc
  DA, MP-DocVQA DA). These are paper-grade.

## Cross-solver prompt parity (D-007)

All solvers reported in the paper pass the same prompt-parity audit
standard. Each solver owns its category tips inline; only
`ANSWER_FORMATTING_RULES` is shared via `src/docvqa/prompts.py`.

Refactor in flight: `CATEGORY_TIPS` / `BASELINE_CATEGORY_TIPS` /
`FLAT_SOLO_TOOL_HINTS` move from `prompts.py` into each paper solver's
own file. RVLM_CATEGORY_TIPS reconciled with canonical content
(missing TEXT TRUNCATION / COUNTING-OBJECTS / CITED PAPER FINDINGS
bullets propagated in; Leader-lines bullet propagated to all solvers).
DA profiles (per-benchmark) already follow the principle and pass
audit (2026-05-27).

## A. Model-size axis (prediction 1)

> **Prediction.** Lift scales with model code-writing capability.
> Mechanism: the recursive sub-call is invoked via code, so models that
> write better tool-call code extract more from the scaffold.
>
> **Caveat (2026-05-27, D-007).** The n=3 cells in the table below
> were run 2026-05-09/10 with **pre-scrub `CATEGORY_TIPS`**. The lift
> direction is robust (same prompts for baseline and scaffold), but the
> absolute numbers may shift 2–4pp under post-scrub prompts. Re-runs
> at n=1 → n=2 escalation are scheduled (task #8) before paper
> headlines lock; existing numbers carry a "pre-scrub prompts" footnote
> in the interim.

We span the size spectrum across two open model families.

| Model | Tier | Baseline (val, n=3) | Scaffold (val, n=3) | Lift | Status |
|---|---|---|---|---|---|
| **Gemma 4 E4B-it** (~4B effective, Google) | ≤8B | **3.75% ± 0.00pp** | **9.58% ± 1.44pp** | **+5.83pp** (t=7.0) | **DONE** — see `docs/experiments/gemma-4-e4b-baseline-scaffold.md` |
| **Qwen 3.5 9B** (open small) | ≤8B | **15.00% ± 1.25pp** | **21.25% ± 2.50pp** | **+6.25pp** (t=3.88) | **DONE** — see `docs/experiments/qwen-9b-baseline-scaffold.md` |
| **Gemma 4 31B-it** (Google mid) | 8–35B | **10.42% ± 0.72pp** | **35.42% ± 5.20pp** | **+25.00pp** (t=8.25) | **DONE** — see `docs/experiments/gemma-4-31b-baseline-scaffold.md` (vllm triage required) |
| **Qwen 3.5 27B** (open mid) | 8–35B | 23.75% ± 2.17pp | 44.69% ± 2.81pp (n=8) | +20.94pp | done (existing) — see `docs/experiments/no-loop-multi-image.md` + `flat-solo-turn-budget-sweep.md` |
| **Gemini 3 Pro** (closed frontier) | >35B | 37.5% test (official) | 1 trial test (59.4%) | — | **frozen — no more runs (out of credits)** |
| **Gemini 3 Flash** (closed small) | ≤8B (proprietary) | 33.75% test (official) | mentioned ~50%, not in results.md | — | **frozen — no more runs (out of credits)** |
| **GPT-5.x or Claude** (closed alt-frontier) | >35B | TBD | TBD | TBD | both ≥3 — load-bearing primary frontier point |

Critical: **matched within-model** (same model, baseline vs scaffold) is
the headline lift figure. The cross-model comparison is secondary.

**Headline pattern across the open-model axis (2026-05-09 / 10):**
the scaffold lifts every open model class. Lift scales sublinearly
with model capacity — small models (≤9B) get ~6pp, mid-tier (27–31B)
get ~21–25pp. The mechanism is the agent loop's reliance on the
LM writing tool-call code; smaller models write worse code, so they
extract less from the scaffold (per memory entry
`feedback_scaffold_lift_scales_with_model_size`). Gemma-vs-Qwen
within tier: Gemma 4 baseline is weaker than Qwen 3.5 (raw VLM gap),
but the scaffold absorbs a meaningful fraction of that family gap
at the mid tier (gap −13pp baseline → −9pp scaffold).

Notes:
- Both Gemma sizes ran (E4B for the ≤8B tier, 31B for the 8–35B
  tier). The original "pick one" guidance was relaxed because both
  fit on the available GPUs and add a non-Qwen family per tier.
- **Gemini results are frozen** (out of API credits). The 59.4% Pro
  test and the ~50% Flash number are single-trial; we cannot
  replicate them. Two ways to handle in paper: (a) report with a
  **"single trial; replication blocked by compute budget"** caveat
  and rely on multi-trial open-model lift as the headline; or
  (b) drop them from the headline lift figure, keep them only as
  context. **Recommend (a)** — the lift sign is consistent with
  open-model multi-trial results, and dropping them weakens the
  frontier story. The variance-discipline rule in §"Variance
  discipline" needs an explicit exception for these two cells.
- The alt-frontier model (GPT-5 / Claude) becomes the **primary**
  frontier datapoint, not a sanity-check secondary one. Originally
  defensive; now load-bearing. Plan ≥3 trials baseline + ≥3 trials
  scaffold.

## B. Document-length axis (prediction 2)

> **Prediction.** Lift scales with effective document length. Mechanism:
> longer docs exhaust the raw VLM's visual context budget; the
> recursive sub-call lets the agent ration perception across pages.
> Falsifiable: a benchmark with mostly short docs should show small or
> zero lift; a long-doc benchmark should show large lift.
>
> **Caveat (2026-05-27, D-006).** The OCR-extension cells below were
> run with the legacy `flat_solo` solver, which confounds OCR with a
> single-image `look()` ergonomic wrapper. The clean OCR-extension
> story requires a new fork of `leanest_solo` that adds `search` +
> `page_texts` only, without `look`. Build + run is scheduled (tasks
> #13–15). Until then, OCR-extension cells carry a "confounded with
> `look()`" footnote.

**Picks:** MP-DocVQA (Tito et al., PR 2023) + MMLongBench-Doc
(NeurIPS 2024). DocVQA-2026 reuses InfographicVQA/SlideVQA/DocVQA-original
data, so those are ruled out as redundant. MADQA still on the list but
shelved until RVLM verification finishes.

### Qwen 27B results — 200Q val sample per benchmark, n=3 trials each

We report two passes per benchmark:

- **Legacy**: original chains with DocVQA-2026's prompt
  (`ANSWER_FORMATTING_RULES` + DocVQA-2026 category tips), and the
  default `no_loop_multi max_pages=10`. Numbers below in italics.
- **DA (closed-loop)**: per-dataset profile
  (:mod:`docvqa.datasets.profile`) drives answer-formatting, tips,
  per-question format hint, and scorer; baseline on MMLongBench-Doc
  uses `max_pages=80` to match the loader's render cap.

The DA pass is the comparison that isolates *scaffold capability*
from *prompt mismatch + truncation*.

#### MP-DocVQA val (ANLS)

| Solver | Legacy | DA |
|---|---|---|
| no_loop_multi | *63.74% ± 0.28pp* | **74.15% ± 0.84pp** |
| leanest_solo (no OCR) | *58.86% ± 2.31pp (−4.88pp)* | 72.52% ± 2.45pp (−1.63pp, n.s.) |
| flat_solo (+OCR) | *63.09% ± 0.97pp (−0.65pp, n.s.)* | 73.17% ± 2.30pp (−0.98pp, n.s.) |

**Legacy → DA shifts the baseline by +10.41pp.** The DocVQA-2026
rules (strip commas, ISO dates, etc.) silently mangled MP-DocVQA's
literal-span answers. After the prompt fix, **all three solvers
converge to ~73-74% in aggregate; neither the scaffold nor OCR adds
anything significant pooled**.

**But the per-page-bucket cut tells a different story** (n=3 trials,
mean accuracy):

| Bucket | n | baseline | leanest_da | flat_da | Δ flat |
|---|---|---|---|---|---|
| 1pp | 70 | 63.3% | 56.7% | 59.5% | −3.81pp |
| 2-5pp | 66 | 84.8% | 83.8% | 80.8% | −4.04pp |
| 6-10pp | 30 | 86.7% | 77.8% | 80.0% | −6.67pp |
| 11-20pp | 39 | **65.8%** | 77.8% | 79.5% | **+13.68pp** |

The aggregate ~0 lift is a bucket-mix artifact. The scaffold *wins*
**+13.68pp on the 11-20pp bucket** — exactly where the baseline's
`max_pages=10` truncation starts biting (baseline accuracy drops from
~85% on shorter buckets to 65.8% on 11-20pp). The mechanism survives
the prompt fix; it just doesn't show in the pooled metric on a
benchmark with 67% docs ≤5pp.

#### MMLongBench-Doc val (judge)

| Solver | Legacy | DA |
|---|---|---|
| no_loop_multi | *33.84% ± 0.87pp* (pages=10) | **46.97% ± 0.51pp** (pages=80) |
| leanest_solo (no OCR) | *60.27% ± 1.91pp (+26.43pp)* | 61.78% ± 1.17pp (**+14.81pp**, t=18.8) |
| flat_solo (+OCR) | *60.44% ± 0.29pp (+26.60pp)* | 63.81% ± 0.76pp (**+16.84pp**, t=22.7) |

**Baseline gains +13.13pp from a fair eval** (max_pages=10→80 = +5pp;
DocVQA-2026 prompt → MMLongBench-Doc profile = +8pp). Scaffold lift
shrinks from the legacy +26pp to a fair **+16.84pp** — still the
largest scaffold lift we've measured, still highly significant, but
about half the legacy headline was an evaluation artifact. OCR's
incremental contribution over the scaffold is +2.03pp (consistent
sign across all 3 trials).

### Revised key finding

**Scaffold lift is real and scales with effective document length** —
but the legacy numbers significantly *overstated* its magnitude on
both benchmarks by holding the baseline to an unfair prompt:

- MP-DocVQA aggregate ANLS: scaffold delta = 0 after fair eval. **But
  per-page-bucket, the scaffold still wins +13.68pp on the 11-20pp
  bucket** where the baseline's max_pages=10 truncation bites — the
  mechanism is preserved; it just gets diluted in the aggregate by the
  67% of questions on docs ≤5pp.
- MMLongBench-Doc (47pp avg): scaffold delta = +16.84pp judge, with
  ~+2pp incremental from OCR on top. **Per-category, the lift is
  dominated by Financial reports (baseline 0% → flat 62.5%, +62.5pp).
  Excluding that category the aggregate lift is ~0.** Same
  mechanism: scaffold helps when the raw VLM can't see the evidence.

Practically: for any new benchmark we report on in the paper, we
should run the baseline with a dataset-aware profile + a fair page
budget before claiming a scaffold lift, or the comparison risks
double-counting the prompt-fit + truncation gains as scaffold
capability.

Full per-trial tables, per-format/per-category breakdowns, and
variance analysis:
- `docs/experiments/mp-docvqa-qwen27b.md`
- `docs/experiments/mmlongbench-doc-qwen27b.md`

Full per-trial / per-format / per-category breakdowns:
- `docs/experiments/mp-docvqa-qwen27b.md`
- `docs/experiments/mmlongbench-doc-qwen27b.md`

### Key finding: scaffold lift scales with effective doc length

The two benchmarks land on *opposite signs* and the page-bucket cut on
MP-DocVQA explains why:

| Bucket (MP-DocVQA) | Baseline | Scaffold | Δ |
|---|---|---|---|
| 1pp (70Q, 34% of sample) | 51.9% | 40.5% | **−11.4pp** |
| 2-5pp (66Q) | 66.7% | 64.1% | −2.5pp |
| 6-10pp (30Q) | 77.8% | 72.2% | −5.6pp |
| 11-20pp (39Q) | 69.2% | 72.6% | **+3.4pp** |

On MMLongBench-Doc (47pp avg), the same mechanism gives a +26.43pp
judge lift; on MP-DocVQA (≤20pp, mostly ≤5pp), the cost of the agent
loop on short docs dominates and the pooled headline is negative.
Per-category on MMLongBench-Doc: Financial-report (longest docs)
**+47.9pp**; Guidebook (most concrete) +16.7pp; even Academic paper
+10.5pp.

### Per answer-format on MMLongBench-Doc (judge)

| Format | Baseline | Scaffold | Δ |
|---|---|---|---|
| Float | 2.3% | **50.8%** | **+48.5pp** |
| Int | 30.9% | 70.7% | +39.8pp |
| List | 21.4% | 44.0% | +22.6pp |
| Str | 48.0% | 64.2% | +16.2pp |
| None ("Not answerable") | 62.9% | 66.7% | +3.8pp |

Numeric formats benefit most: the raw VLM cannot extract precise
numbers from a truncated 10-page slice of a 100-page financial report,
but the scaffold's `look()` zooms into the right page. "Not answerable"
already strong for the baseline (refusal needs no perception).

### Frontier model on these benchmarks

**Not yet run.** Plan: re-use Gemini 3 Pro / Flash on the same 200Q
samples once §A frontier-model results are in. Expected wall ~2-3h
per benchmark per solver. Hold off until OCR pipeline is extended to
the new datasets so flat_solo is also runnable.

### Caveats

- **MMLongBench: 1 of 22 sample docs (`mi_phone.pdf`) fails to download
  from HF**; 198/207 questions scored. All 5 formats remain represented.
- **MMLongBench: page cap at 80**; the longest 10-Ks (~150pp) lose
  evidence pages beyond 80. Numbers are conservative on those docs.
  Sensitivity check at `max_pages=200` is on the todo list.
- **Qwen judge calibration** done before any cell ran: 11/12 = 91.7%
  agreement on hand-marked triples (single miss on Float ~1% tolerance
  edge case). Below 70% would have blocked cell runs; above threshold
  by 20+ pp.

### Method baselines from the lit review (per D-005)

In addition to the official ICDAR baselines and our matched
within-model baselines, run / report against:

- **MADQA constrained-agent baseline** (Borchmann et al., 2026,
  needs verification) — engages with their "focused RLMs work" thesis
  directly. Run on MADQA if we pick it as a secondary benchmark.
- **ARIAL** (arXiv:2511.18192) — closest agentic-DocVQA competitor on
  the original DocVQA benchmark; report their published number for
  context if we evaluate on DocVQA original.
- **Frontier models with our scaffold** vs **frontier models direct**
  — already in the model-axis table.

We do **not** need to re-implement every prior method. Reporting
published numbers alongside ours on the same benchmark is sufficient
for context.

## C. Active-perception mechanism (prediction 3) + supporting ablations

> **Prediction.** The lift comes specifically from *active, iterative*
> VLM sub-calls — the agent choosing what region to inspect, at what
> resolution, across multiple turns — not from giving more compute to
> a single VLM call. Three independent ablations triangulate this,
> already measured.
>
> **Triangulation evidence:**
>
> | Component of active perception | Ablation | Result | Reading |
> |---|---|---|---|
> | Region selection (cropping) | flat_solo cropping-off | **−7.81pp** | choosing what to perceive matters |
> | Iteration count | turn budget m=5 vs m=30 | **−15pp** at m=5 | the mechanism is iterative, not one-shot |
> | Recursive sub-call structure | leanest 48.8% vs no_loop_multi 20.0% | **+28.8pp** | recursive agent↔VLM dominates one-shot |
>
> Reframed (2026-05-27) from an earlier "REPL-only" framing that
> predicted full collapse if `batch_look` was removed. The REPL-only
> solver was built and smoke-tested (0/5 on 2 docs — agent SUBMITs
> "Unknown" immediately because it has no perception). The result is
> obvious — "no perception → no answer" — and tests a strawman. The
> three ablations above are the sharper test of active perception as
> the mechanism. REPL-only stays as `src/docvqa/solvers/repl_only_solver.py`
> for documentation; it is not a paper cell.

| Ablation | Description | Status |
|---|---|---|
| **VLM sub-call on/off** | Fork of leanest with `batch_look` removed. The load-bearing test. | **NOT DONE** — tasks #7/#9 |
| **No-loop baseline** | Direct VLM Q&A — single forward pass, no REPL, no tools, no agent. Most modern models are thinking models, so CoT is implicit; this is the "raw model" point. | **DONE** — Qwen 3.5 27B val, n=3 with matched tips: **21.25% ± 1.25pp** (composite); **23.75% ± 2.17pp** (multi-image native-res, max_pages=10). vs scaffold 44.69%: composite gap −23.4pp (19.1 SE), multi gap −20.9pp (13.1 SE). See `docs/experiments/no-loop-baseline.md` and `no-loop-multi-image.md`. |
| **OCR on/off** | Full method vs no-OCR variant (existing Leanest solver, `solver=leanest_solo`). Removes the symbolic exploration channel; main agent must rely on VLM sub-call alone. | "REPL without symbolic context access" — partial analogue | **DONE** — Qwen 3.5 27B val n=3: **40.00% ± 0.00pp** (vs flat_solo 44.69%: OCR + cropping add +4.7pp; vs no_loop_multi+tips 23.75%: agent-loop + VLM-tool channel adds +16.3pp). See `docs/experiments/leanest-ocr-off.md`. Caveat: leanest→flat_solo delta also includes cropping; not a pure OCR ablation. |
| **VLM sub-call on/off** | Full method vs OCR-only (no VLM tool). Removes the recursive sub-call; agent must reason from OCR text alone. | "REPL without sub-calling" — direct analogue | not done |
| **Search tool on/off** | Full method vs no-`search()` tool: BM25 retrieval removed but `page_texts` still in scope so the agent can scan OCR manually. Isolates "BM25 retrieval" from "raw OCR text in scope". Cleaner OCR-channel ablation than leanest→flat_solo (which also bundles tool structure). | not in RLM paper | **DONE** — n=8: **42.50% ± 3.90pp** vs 44.69% baseline; gap **−2.19pp (t=1.29) — NOT significant**. BM25 is largely redundant given `page_texts` in scope; agent compensates via `re.search()` regex over OCR text. See `docs/experiments/flat-solo-search-off.md`. |
| **VLM cropping on/off** | Full method (VLM accepts arbitrary PIL Image — pages, crops, regions) vs page-only (VLM accepts only a page index, no cropping/zoom). Isolates the "active perception" contribution from the broader VLM-on/off comparison. | not in RLM paper | **DONE** — n=8: **36.88% ± 2.50pp** vs 44.69% baseline; gap **−7.81pp (5.88 SE)** |
| **Turn budget** | Vary max turns. When does extra budget stop helping? | RLM-style inference scaling curve | **DONE** — 8 trials × {10,20,30,40}; peak m=30 = 44.69% (see below) |
| **Category tips on/off** | Remove per-category prompt tips. Tests whether handcrafted hints carry meaningful weight or are decoration. | not in RLM paper | **DONE** — n=8 clean (t1 excluded, sandbox-error contam): **38.75% ± 3.13pp** vs 44.69% baseline; gap **−5.94pp (3.99 SE)** |

### OCR's role: two opposing forces (see `per-doc-flat-vs-leanest.md`)

The aggregate OCR effect (~1pp on the mean, n.s.) hides two opposing
mechanisms. From per-doc comparison of flat_solo m=30 (full, OCR ON)
vs leanest m=40 (no OCR), n=8 trials each per cell:

- **Long docs (≥20 pages, n=12):** flat_solo +3.3pp over leanest.
  OCR + BM25 retrieval makes "find the relevant page" cheap when
  there are dozens of pages to consider. Cleanest win: `business_report_3`
  (89 pages, 2 questions): flat_solo 100% vs leanest 87.5%.
- **Short docs (<20 pages, n=13):** delta near zero (+1.4pp).
- **Visually-rich docs:** OCR can *hurt*. `engineering_drawing_3`
  (6 pages, leader-line diagrams): leanest beats flat_solo by
  **+29.2pp** (70.8% vs 41.7%). OCR text from diagrams is noisy
  and misdirects the agent away from visual reasoning.

Per-doc winner counts: flat 13/25, leanest 5/25, tie 7/25.

**Paper framing.** OCR is a *long-document navigation tool* and a
*visual-perception distractor*; on a balanced 25-doc benchmark these
roughly cancel. The "OCR-as-stability-anchor" framing (std halves
when OCR is present) is the more robust paper claim. The per-doc
breakdown suggests OCR's value should rise on long-document
benchmarks (MP-DocVQA, MMLongBench-Doc) and fall on
diagram/chart-heavy ones (ChartQA, infographic-only sets) —
falsifiable predictions for the benchmark-generality §B.

### Turn-budget sweep results

Flat Solo lean / Qwen 3.5 27B / val 80q, n=8 per cell:

| max_iterations | mean | std | range |
|---|---|---|---|
| **5** | **30.00%** | **0.00pp** | 30.00 (n=3) |
| 10 | 41.41% | 3.16pp | 36.2–45.0 |
| 20 | 40.94% | 3.32pp | 36.2–46.2 |
| **30** | **44.69%** | **2.81pp** | 40.0–48.8 |
| 40 | 40.78% | 3.89pp | 35.0–47.5 |

m=30 is the peak; both shorter and longer budgets are ~3–4pp lower.
Interpretation: the curve is non-monotonic, with longer-than-needed
budgets actually hurting (likely unproductive trajectories accumulate
errors / waste context). m=30 also has the lowest variance, supporting
the same story. **m=5 added 2026-05-08, n=3: 30.00% ± 0.00pp** —
~11pp below m=10, confirming that the lower end of the curve drops off
sharply (agent doesn't have enough turns to finish its work). The
five-point curve {5, 10, 20, 30, 40} gives the paper figure a clear shape.

Skipped (see `decisions.md`):

- **OCR quality sensitivity** (D-002) — tangential; binary OCR/no-OCR is enough.
- **SC budget curve** (D-003) — SC dropped from paper framing; we report
  mean ± std across independent trials instead.

Future / deferred:

- **Long-document context-fit issue** — for very long documents, raw page
  images may not fit a model's context window. The scaffold sidesteps
  this via retrieval. Worth a paragraph in the paper; full investigation
  deferred unless a reviewer asks.

## D. Error analysis & qualitative material

- **Per-category breakdown** — already in `docs/results.md:88-100` and
  `docs/results.md:163-172`. Reuse.
- **Maps stays at 20%** — even with model upgrade across the 8 trials. Trace examples
  to understand why; potential discussion-section material on scaffold
  limits.
- **Qualitative diagram** — 1 figure showing a full agent trace on a
  representative example (probably from `engineering_drawing` or
  `infographics` where we score well).
- **Failure walkthroughs** — 2–3 traces where the scaffold fails, for
  honest failure section.

## Server split (parallel execution)

Two GPU hosts available. Splitting work across hosts (rather than
piling it on one) avoids sandbox-subprocess OOM contention. Concrete
incident: when a 4-GPU vllm spun up alongside a running eval on the
same box, the eval's sandbox subprocesses started dying — 2064
"Subprocess is not running" errors in `flat-solo-no-tips-3_5-27b-val-t1`,
forcing the agent to abstain ("Unknown") on 55/80 questions and
producing a contaminated 18.75% score (re-run t2 cleanly: 38.8%).
Host-level isolation is the cheapest fix; topic-coherent groups make
each host's queue easy to reason about.

Both hosts serve Qwen 3.5 27B (Host A on local port 8927, 3-GPU; Host
B on its own local port 8928, 4-GPU — accessed from Host A via
tunnel). The bottleneck is **sandbox subprocess load on the
evals.py-running host**, not vllm GPU. Each `evals.py` lane spawns
many concurrent agent sandboxes for code execution — that's where
the OOM happened. Splitting which **machine runs evals.py** is the
real lever; the vllm endpoints are not the contention point.

### Host A (this machine, runs evals.py for these)
- **Done (May 8, 2026):**
  - `tips-off` (vllm 8927, c=24) — category tips ablation: n=8 clean, gap −5.94pp (5.88 SE)
  - `crop-off` (vllm 8928, c=32) — D-004 cropping ablation: n=8, gap −7.81pp (5.88 SE)
- **Policy:** port 8928 (remote 4-GPU vllm) is **off-limits** to Host A
  going forward — reserved for Host B's own workload. Host A future
  ablations use **only the local 8927 lane (single-lane,
  c=24 ceiling)**.
- Currently idle. New work would land on 8927 only.

### Host B (other server, runs evals.py for these)
Host B has its own Qwen 27B vllm + sandbox capacity, so it can carry
**Qwen 27B Group B0** in addition to the B1/B2/B3 groups that don't
need local 27B at all. Pick groups in this priority order; each
group is self-contained.

- **Group B0 — Qwen 27B ablations / test (highest priority on B):**
  - ~~**No-loop baseline** (`solver=no_loop`), val, ≥3 trials — risk-rank #2; kills "scaffold matters" claim if raw model already wins~~ — **DONE 2026-05-08**: 17.08% ± 2.60pp (n=3) vs scaffold 44.7%; +27.6pp lift, claim cleared.
  - ~~**OCR on/off via Leanest** (`solver=leanest_solo`), val, ≥3 trials~~ — **DONE 2026-05-08**: 40.00% ± 0.00pp (n=3). vs flat_solo 44.7% → OCR + tips + cropping add +4.7pp. vs no_loop 17.08% → VLM-tool/agent-loop channel adds +22.9pp.
  - **Qwen 27B test runs** with the best val config (flat_solo lean m=30, full tools) — ≥3 trials on test; locks the matched-baseline test number that anchors the headline lift figure
  - **m=5 turn budget point**, val, ≥3 trials — defensive; 4-point curve already shapes the figure
- **Group B1 — Closed/API models (no local GPU; can overlap B0):**
  - Alt-frontier (Claude or GPT-5) baseline + scaffold, val + test, ≥3 trials each
  - (Gemini Pro / Flash: **dropped**; out of credits, existing single-trial numbers used as-is per §A note)
- **Group B2 — Small open models (separate vllm instance):** **DONE 2026-05-09/10**
  - ~~Qwen 3.5 9B baseline + scaffold, val~~ — n=3+3, lift +6.25pp (`qwen-9b-baseline-scaffold.md`)
  - ~~Gemma 4 E4B-it baseline + scaffold, val~~ — n=3+3, lift +5.83pp (`gemma-4-e4b-baseline-scaffold.md`)
  - ~~Gemma 4 31B-it baseline + scaffold, val~~ — n=3+3, lift +25.00pp (`gemma-4-31b-baseline-scaffold.md`); vllm TP=4 + `--enforce-eager` required for scaffold stability
- **Group B3 — Second benchmark (after lit review picks):**
  - Qwen 27B baseline + scaffold on chosen benchmark, ≥3 trials
  - One frontier model (alt-frontier: Claude or GPT-5 — Gemini dropped, see §A) baseline + scaffold on chosen benchmark, ≥3 trials

**Recommended start order on Host B:** B0 first (clears the highest
risk-rank items + locks the matched-baseline test number), B1 in
parallel since it needs no GPU, then B2, then B3 (depends on lit
review). B0 and B1 can run side-by-side — B1 is API-bound and won't
contend with sandbox CPU/RAM.

## Risk-ranked execution order (post-D-006)

Ordered by which experiment most likely **falsifies the paper's three
predictions**. Each step is gated by D-008 escalation (n=1 first, n=2 if
direction holds, n=8 only after the paper headline locks).

### Critical path

1. **Solver-architecture merge** — merge `*_solo` ← `*_da` solver
   pairs into DA-by-default; move tool-agnostic content into
   DocVQA-2026 profile per D-009. Refactor-first prevents technical
   debt across all subsequent runs. *Tasks #20, #21, #22.*
2. **Build new M+OCR solver** (DA-by-default) = fork of `leanest_solo`
   + `search` + `page_texts`, no `look`. Required for the clean
   OCR-extension cell (current `flat_solo` data is confounded with
   `look()`). *Task #13.*
3. **Run M+OCR on DocVQA-2026 val** (prediction 2 mechanism check
   on the moderate-doc benchmark). Predicted ≈ leanest (OCR is neutral
   on moderate-length docs). *Task #14. Host A 8927.*
4. **Re-run model-axis cells with synchronized prompts** (Gemma E4B,
   Qwen 9B, Gemma 31B) — current numbers used pre-scrub prompts. Lift
   sign should be preserved; absolute magnitudes may shift 2–4pp.
   *Task #8. Host B.*
5. **Re-do MMLongBench-Doc + MP-DocVQA OCR-extension cells with M+OCR**
   (prediction 2 long-doc evidence, clean version). *Task #15.*
6. **Restart RVLM chain** with reconciled prompts (alternative-angle
   datapoint). *Task #19. Other host.*

**Dropped from critical path 2026-05-27:** the REPL-only ablation
(prediction 3 strawman). Reframing to active-perception mechanism is
already supported by existing ablations (cropping, turn budget,
leanest-vs-no_loop_multi).

### Done — keep

- **No-loop baseline (raw VLM)** — done, n=8 SC-8 split-calibration:
  val 20.0% / test 11.0%. Anchors the split-difficulty floor.
- **OCR-free vs OCR-extension legacy comparison** — done on `flat_solo`
  (confounded). Result: post-scrub OCR-free wins on test by 1pp.
  Re-do with clean M+OCR (item #5 above) for the paper number.
- **Model-size axis trends** — done at n=3 pre-scrub. Direction is
  robust; re-runs in item #4 lock absolute numbers.
- **MMLongBench-Doc + MP-DocVQA at n=3 DA** — done. Doc-length
  prediction supported; OCR-extension cells need re-running per item #5.
- **Turn-budget sweep, cropping ablation, search ablation, tips
  ablation** — all done on `flat_solo`. Transfer to leanest since the
  ablated components are present in both. Cropping −7.81pp, search
  n.s., tips −5.94pp.

### Out — D-006

- Gemini 3 Pro/Flash multi-trial replication — credits exhausted;
  single-trial numbers carry caveat per D-006.
- Alt-frontier (GPT-5 / Claude) — deferred; closed-frontier story
  rests on the Gemini single-trial caveat + the open-model
  multi-trial axis.
- SC-8 voting curve, multi-image extension, GEPA, pyai port — out of
  scope per D-006.

### Falsification gates (when to stop and reframe)

- If item #1 (VLM-sub-call-off) does **not** collapse to baseline:
  prediction 3 fails; the scaffold's load-bearing mechanism is
  something other than the recursive sub-call. Major reframe required.
- If item #4 model-axis re-runs reverse the lift sign on any cell:
  prediction 1's monotonicity weakens. Re-examine.
- If item #5 long-doc OCR-extension shows zero lift under M+OCR:
  prediction 2 weakens; the OCR contribution may have been a
  `look()`-wrapper artifact.

## Required figures (post-D-006)

1. **Headline figure: model-size axis (prediction 1)** — for each model
   (E4B / 9B / 27B / 31B): bar chart of {raw VLM baseline, proposed
   method (M)}. Lift labels per pair.
2. **Doc-length axis (prediction 2)** — scaffold lift vs effective
   document length across benchmarks (DocVQA-2026, MMLongBench-Doc,
   MP-DocVQA per-bucket). Either scatter or grouped bar.
3. **Mechanism table (prediction 3)** — Qwen 27B on val:
   {raw VLM, REPL-only (VLM-sub-call-off), M (full), M+OCR}.
   The VLM-sub-call-off row is the load-bearing test.
4. **Ablation table** — M with each component ablated: cropping off,
   tips off, search off (for M+OCR variant), turn budget sweep.
   On Qwen 27B val.
5. **Qualitative diagram** — agent trace walkthrough on one
   representative success and one failure.
6. **Per-category breakdown** — appendix table.

Out of the figure list (per D-006):
- The competition-tactic ablations (FLAT_SOLO_TOOL_HINTS, multi-image,
  GEPA) — out of scope.
- SC voting curves — out of scope.
- Alt-frontier datapoint — deferred.

## Open questions for the user

- Paper name for the proposed method (placeholder: M).
- Target venue + deadline (TBD; top candidates EMNLP / ACL-Findings /
  TMLR per `paper/README.md`).
- Whether to also report the legacy `flat_solo` (M + OCR + `look()`)
  as an appendix "kitchen-sink" cell (task #16).
- Whether to include the split-calibration paragraph in the
  evaluation section (would pre-empt reviewer questions about
  val→test asymmetry).
- Whether the model-axis re-runs (task #8) escalate to n=8 once n=1/n=2
  confirms the direction, or whether n=2 is sufficient with the
  pre-scrub n=3 used as the third data point.

Resolved (kept for provenance):
- ~~Which Gemma size?~~ — both ran (E4B for ≤8B, 31B for 8–35B), 2026-05-09.
- ~~Method name picking~~ — deferred until experiments lock (D-006).
- ~~Alt-frontier model~~ — deferred per D-006.
- ~~Multi-trial Gemini Pro replication~~ — dropped, credits exhausted.

## Cross-references

- `docs/results.md` — current experiment results
- `docs/experiment-history.md` — historical context
- `docs/paper/lit-review.md` — to be produced; informs benchmark + baseline picks
- `docs/paper/related-works.md` — paper index with connection notes (RLM
  ablation parallels documented under the foundational entry)
- `docs/paper/README.md` — project overview, status tracker
- `CLAUDE.md` — project root, infrastructure notes
