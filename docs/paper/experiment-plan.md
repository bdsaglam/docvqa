# Experiment Plan — DocVQA Scaffold Paper

## Headline claims to support

1. **RLM applied to DocVQA lifts every model class.** Applying the RLM
   paradigm — REPL + symbolic OCR access + VLM as recursive sub-call —
   delivers double-digit accuracy gains on document VQA across small
   (≤8B), mid (8–35B), and frontier (>35B, closed) models.
2. **Generality across benchmarks.** Gains hold beyond ICDAR 2026 DocVQA.
3. **Component contributions.** The REPL/symbolic-access channel (OCR
   retrieval), the VLM sub-call, and turn budget each contribute
   meaningfully — or surprisingly don't, which is also a paper-worthy
   finding. Mirrors the ablation structure of the RLM paper. (SC dropped
   from method per `decisions.md` D-003 — we report mean ± std across
   independent trials instead.)

## Variance discipline (applies everywhere)

- **≥3 trials per headline number.** Report mean ± std.
- **No SC voting in headline numbers** (per `decisions.md` D-003) — we
  report the per-trial mean and its variance, not the voted score.
- Single-trial numbers in `docs/results.md` (e.g., 59.4% Pro+Flash test,
  35B variant) **must be replicated** before being headlined.
- Use the existing 8-trial setup at `docs/results.md:144` as the template.
- For Qwen 3.6 27B test: compute the 8-trial test mean ± std from the
  individual per-trial test scores (currently summarized only as SC-8
  43.75% in `docs/results.md:161`).

## A. Model generality (claim 1)

We span the size spectrum, hitting all three competition tiers plus a
small-open and an alt-frontier point.

| Model | Tier | Baseline status | Scaffold status | Trials needed |
|---|---|---|---|---|
| **Qwen 3.5 9B** (open small) | ≤8B | not run | not run | both ≥3 |
| **Gemma 4 (size TBD, open)** | ≤8B or 8–35B | not run | not run | both ≥3 |
| **Qwen 3.6 27B** (open mid) | 8–35B | not run | done (8 trials) | baseline ≥3 |
| **Gemini 3 Pro** (closed frontier) | >35B | 37.5% test (official) | 1 trial test (59.4%) | scaffold ≥3 |
| **Gemini 3 Flash** (closed small) | ≤8B (proprietary) | 33.75% test (official) | mentioned ~50%, not in results.md | both ≥3 |
| **GPT-5.x or Claude** (closed alt-frontier) | >35B | TBD | TBD | both ≥3 |

Critical: **matched within-model** (same model, baseline vs scaffold) is
the headline lift figure. The cross-model comparison is secondary.

Notes:
- Pick **one** Gemma size based on what's practical to host. Don't run
  every Gemma variant.
- The alt-frontier model (GPT-5 / Claude) shows the lift isn't a
  Gemini-specific quirk. Cost-permitting; if budget tight, defer.

## B. Benchmark generality (claim 2)

Pick 2 benchmarks. Final picks deferred until RVLM/MADQA verification
completes. Candidates:

- **DocVQA (original 2020)** — near-distribution, established baselines.
- **MP-DocVQA** — multi-page; tests scaffold's ability on long docs.
- **MMLongBench-Doc** (NeurIPS 2024) — long-context, GPT-4o reportedly ~44.9% F1.
- **MADQA** (Borchmann et al., arXiv:2603.12180) — multimodal agentic doc
  QA. **Strong candidate** because their constrained-agent method is also
  on our baseline list (D-005), so MADQA enables a direct head-to-head.
- **InfographicVQA** — adjacent, infographic-heavy.
- **SlideVQA** — adjacent, slide-heavy.
- **ChartQA** — further; chart reasoning, harder for our OCR-centric tools.

For each picked benchmark:

- Baseline + scaffold on **Qwen 27B** (cheap, open).
- Baseline + scaffold on **at least one frontier model**.
- ≥3 trials each.

Two benchmarks is enough for "generalizes." More than that is benchmark
suite territory — skip unless lit review surfaces a specific comparison
reviewers will demand.

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

## C. Ablations (claim 3)

| Ablation | Description | RLM-paper parallel | Status |
|---|---|---|---|
| **No-loop baseline** | Direct VLM Q&A — single forward pass, no REPL, no tools, no agent. Most modern models are thinking models, so CoT is implicit; this is the "raw model" point. | "no-REPL baseline" in RLM | impl ready (`solver=no_loop`); 0/8 |
| **OCR on/off** | Full method vs no-OCR variant (existing Leanest solver, `solver=leanest_solo`). Removes the symbolic exploration channel; main agent must rely on VLM sub-call alone. | "REPL without symbolic context access" — partial analogue | partial — Leanest val runs exist (43.8%) |
| **VLM sub-call on/off** | Full method vs OCR-only (no VLM tool). Removes the recursive sub-call; agent must reason from OCR text alone. | "REPL without sub-calling" — direct analogue | not done |
| **VLM cropping on/off** | Full method (VLM accepts arbitrary PIL Image — pages, crops, regions) vs page-only (VLM accepts only a page index, no cropping/zoom). Isolates the "active perception" contribution from the broader VLM-on/off comparison. | not in RLM paper | impl ready (`solver.vlm_cropping=false`); 0/8 |
| **Turn budget** | Vary max turns. When does extra budget stop helping? | RLM-style inference scaling curve | **DONE** — 8 trials × {10,20,30,40}; peak m=30 = 44.69% (see below) |
| **Category tips on/off** | Remove per-category prompt tips. Tests whether handcrafted hints carry meaningful weight or are decoration. | not in RLM paper | in progress (1/8 launched) |

### Turn-budget sweep results

Flat Solo lean / Qwen 3.5 27B / val 80q, n=8 per cell:

| max_iterations | mean | std | range |
|---|---|---|---|
| 10 | 41.41% | 3.16pp | 36.2–45.0 |
| 20 | 40.94% | 3.32pp | 36.2–46.2 |
| **30** | **44.69%** | **2.81pp** | 40.0–48.8 |
| 40 | 40.78% | 3.89pp | 35.0–47.5 |

m=30 is the peak; both shorter and longer budgets are ~3–4pp lower.
Interpretation: the curve is non-monotonic, with longer-than-needed
budgets actually hurting (likely unproductive trajectories accumulate
errors / waste context). m=30 also has the lowest variance, supporting
the same story. m=5 is queued as a lower-end check but the existing
shape already suggests it will be worse than m=10. For the paper figure
("turn budget curve"), the headline shape is clear from these four
points.

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

## Risk-ranked execution order

Run experiments in the order most likely to **falsify the paper first**.

1. **Qwen 27B baseline (no scaffold) on val + test** — locks the
   matched-baseline figure. Cheap. If lift is small, the paper's spine
   is at risk.
2. **No-loop ablation on Qwen 27B** — kills "scaffold matters" claim if
   the raw model already does most of the work.
3. **Gemini 3 Pro test replication (≥3 trials)** — validates the headline
   59.4% number. Single-trial means we don't actually know what the
   real number is.
4. **Second benchmark — Qwen 27B baseline + scaffold** — kills
   "generality" claim if it fails. Run after lit review picks the
   benchmark.
5. **Small-model runs (Qwen 3.5 9B, Gemma)** — extends model axis,
   relatively cheap.
6. **OCR on/off ablation** — clean comparison vs Leanest solver.
7. **VLM cropping on/off ablation** — page-only variant vs full
   arbitrary-image variant. Isolates the active-perception contribution.
8. **Second benchmark — frontier model with scaffold** — strengthens
   generality on harder model class.
9. **Turn budget curve** — feeds the efficiency story. **Done** (Qwen 3.5
   27B val): peak at m=30 (44.69% ± 2.81pp); m=10/20/40 each ~3–4pp
   lower. See section C results table.
10. **Category tips ablation** — defensive.
11. **Alt-frontier model (GPT-5 / Claude)** — generalization beyond
    Gemini. Cost-permitting.

If steps 1–3 fail or weaken substantially, **stop and reframe** before
spending compute on later steps.

## Required figures (preliminary — refine after lit review)

1. **Headline bar chart** — model × {baseline, scaffold}, showing lift
   across the size spectrum.
2. **Benchmark generality grid** — benchmark × {baseline, scaffold} on
   key models.
3. **Per-category breakdown** — already have data.
4. **Turn budget curve** — accuracy vs max turns.
5. **Qualitative diagram** — agent trace walkthrough on one example.
6. **Ablation table** — full vs no-OCR vs no-VLM vs no-cropping vs no-tips vs no-loop, on Qwen 27B.

## Open questions for the user

- Method name? (placeholder: "the scaffold")
- Target venue + deadline?
- Compute budget — how aggressively to span the model axis (drop
  alt-frontier or Gemma if tight)?
- Which 2 benchmarks (defer to lit review)?
- Which Gemma size?

## Cross-references

- `docs/results.md` — current experiment results
- `docs/experiment-history.md` — historical context
- `docs/paper/lit-review.md` — to be produced; informs benchmark + baseline picks
- `docs/paper/related-works.md` — paper index with connection notes (RLM
  ablation parallels documented under the foundational entry)
- `docs/paper/README.md` — project overview, status tracker
- `CLAUDE.md` — project root, infrastructure notes
