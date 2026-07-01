# Experiments

Per-experiment writeups under the D-006 framing (visual-context-budget
hypothesis). One file per solver×model pairing — the canonical record so
future sessions don't re-derive numbers from `output/runs/`.

> **All numbers here are from current code** (post 2026-06-01 cleanup:
> minimized/parity-stripped prompts + per-call `num_retries=5` only, with
> the whole-agent `@retry` removed). Earlier writeups used pre-change
> prompts and retry logic and their numbers are **no longer valid** —
> they were moved to [`archive/experiments/`](../../archive/experiments/)
> and must not be cited as current.

## High-level results

### Method vs baselines — Qwen 3.5 27B, val (25 docs / 80 Qs)

The 8-solver comparison (`lm.enable_thinking=false`, local vllm :8927),
**n=8** (val 25 docs / 80 Qs). Numbers are the **canonical re-run** (retained
per-trial artifacts → pass@k/SC@k); Δ vs the `rvlm` reference. The append-only/
MDP twin is **`codeact_chat`** (corrected; within ~2pp of `rvlm`); the old dspy
`codeact` budget sweep is deprecated/archived. Full detail + headlines in
[`../results.md`](../results.md).

> **Per-cell vs synthesis.** This table and `results.md` use the re-run batch.
> The individual ablation per-cell docs below report each cell's *original-batch*
> measurement (with per-trial detail), and their "Δ vs `rvlm` 39.38" are clean
> **within-original-batch** comparisons — both numbers re-rolled together in the
> re-run, so the re-run cross-cell deltas (here) differ but the rankings hold.

| Solver | Role | Val (n=8) | Δ vs `rvlm` | Status |
|---|---|---|---|---|
| **`rvlm`** | **proposed method** — REPL + recursive VLM `batch_look` (OCR-free) | **41.88% ± 5.79** | — | ✓ reference |
| **`codeact_chat`** | corrected append-only/MDP twin of `rvlm` (true chat MDP, no dspy; FT target) | **39.53% ± 2.83** | −2.35pp | ✓ done — within ~2pp; old dspy `codeact` deprecated |
| `rvlm_subagent_ablation` | sub-call generalized to `batch_subagent` | 36.72% ± 2.75 | −5.16pp | ✓ done — within combined std (general delegation unused) |
| `rvlm_ocr_ablation` | + OCR `page_texts` + BM25 `search` (OCR extension) | 36.56% ± 2.89 | −5.32pp | ✓ done — **OCR adds nothing** over OCR-free |
| `rvlm_nocrop_ablation` | `batch_look` by page index, no crop/zoom | 35.78% ± 2.31 | −6.10pp | ✓ done — crop is category-specific |
| `react_baseline` | perception tools, **no REPL** (`dspy.ReAct`) | 27.19% ± 3.19 | −14.69pp | ✓ done — REPL load-bearing |
| `direct_vlm` | `display()` pages into agent's own context, **no** VLM sub-call | 22.34% ± 2.79† | −17.04pp† | ✓ done — sub-call load-bearing |
| `raw_vlm_multi_baseline` | raw multi-image, single VLM call, **no scaffold** | 20.94% ± 1.60 | −20.94pp | ✓ done — scaffold-lift floor |
| `official_baseline` | competition `MASTER_PROMPT`, multi-image, no scaffold | 18.91% ± 1.94 | −22.97pp | ✓ done — external anchor (kit-faithful: 21.67% ± 1.91) |
| `rlm_ocr` | RLM + OCR text perception, **no vision** | 14.69% ± 2.19 | −27.19pp | ✓ done — **OCR-free headline control** (matrix floor) |

† `direct_vlm` (and `rvlm_rationale` 39.22, parity) were not in the re-run; their
numbers are original-batch (Δ vs original `rvlm` 39.38). `rvlm_hybrid` is the
accepted context-ceiling failure (35.47 upper bound, not tabled) — see `results.md`.

**Reading:** `rvlm` (~42%) lifts ~+21pp over the no-scaffold raw-VLM floor
(~21%). The two halves of the scaffold each matter — drop the REPL
(`react` ~27%) or serve perception one-shot instead of via the recursive
sub-call (`raw_vlm_multi` ~21%, `direct_vlm` 22%) and it collapses, but
neither alone recovers `rvlm`. Adding OCR (`rvlm_ocr`) or generalizing the
sub-call (`rvlm_subagent`) on top of the OCR-free recursive sub-call buys ~0
(within combined std) — supporting the OCR-free recursive-perception framing.

### Model-size / VLM-quality axis (prediction 1)

Hold the reasoner fixed, swap **only** the VLM tool backend — does a
bigger VLM lift the headline (perception-bound) or not (reasoner-bound)?
n=8 per arm, val. RLM-only summary below; the full harness×model matrix
(RLM / ReAct / CodeAct × sizes, incl. the v2↔v3 reasoning-vs-perception
factorial and Gemma) is in
[`harness-axis-summary.md`](harness-axis-summary.md), with per-model raw
numbers in the by-model files (see *Active files*).

| Reasoner (LLM) | v1 homog (VLM = LLM) | v2 mixed (VLM = 27B) | Δ (v2 − v1) | n |
|---|---|---|---|---|
| Qwen 3.5 9B | 18.91% ± 3.81 | 25.31% ± 4.16 | **+6.41pp** — Welch t=3.21, 95% CI [+2.1, +10.7], **sig.** | 8 |
| Qwen 3.5 4B | 14.22% ± 3.83 | 21.09% ± 3.16 | **+6.88pp** — Welch t=3.91, 95% CI [+3.1, +10.7], **sig.** | 8 |
| Qwen3 8B (older gen) | — (n/a) | 11.73% ± 2.96 | — (off the size curve) | 8 |

At both 9B and 4B, swapping only the VLM →27B lifts ~6–7pp (9B +6.41,
4B +6.88) → the scaffold is **perception-budget-bound** for mid/small
reasoners (supports D-006); the lift holds across reasoner size, the
signature of a perception (not orchestration) bottleneck.

**Older-generation reasoner point:** a Qwen3-8B reasoner on the *same*
27B VLM scores only 11.73% — below 4B v2 and half of 9B v2. Modality is
not a confound (in v2 the reasoner delegates all perception to the VLM
and never sees pixels, so text-only vs multimodal is irrelevant); the
only variable vs 9B/4B is **generation** (Qwen3 vs Qwen3.5). The older
8B is just a weaker orchestrator (thrashed ~18 iters/question; bug ruled
out). Clean reasoner-*quality* signal — kept off the Qwen3.5 9B↔4B size
curve. A clean 8B size point would need Qwen3.5-8B.

## Active files

Two organizing axes (see *Conventions*): the **27B ablation matrix** is
**by solver** (one solver per file, single model); the **cross-model
sweeps** are **by model** (one model per file, all harnesses), with the
cross-cutting narratives in a single synthesis doc.

**27B ablation matrix — `{solver}-qwen-3_5-27b.md`** (Qwen 3.5 27B homog):

| File | What |
|---|---|
| [rvlm-qwen-3_5-27b.md](rvlm-qwen-3_5-27b.md) | proposed method (reference) |
| [rvlm_ocr_ablation-qwen-3_5-27b.md](rvlm_ocr_ablation-qwen-3_5-27b.md) | OCR extension ablation |
| [rvlm_hybrid_ablation-qwen-3_5-27b.md](rvlm_hybrid_ablation-qwen-3_5-27b.md) | hybrid display+sub-call ablation |
| [rvlm_nocrop_ablation-qwen-3_5-27b.md](rvlm_nocrop_ablation-qwen-3_5-27b.md) | no crop/zoom (whole-page reads) ablation |
| [rvlm_subagent_ablation-qwen-3_5-27b.md](rvlm_subagent_ablation-qwen-3_5-27b.md) | generalized `batch_subagent` ablation |
| [rvlm_subagent_full-qwen-3_5-27b.md](rvlm_subagent_full-qwen-3_5-27b.md) | sub-call = full agent (negative, 10× cost) |
| [rvlm_rationale-qwen-3_5-27b.md](rvlm_rationale-qwen-3_5-27b.md) | VLM answer + `[note:]` channel ablation |
| [direct_vlm-qwen-3_5-27b.md](direct_vlm-qwen-3_5-27b.md) | see-it-yourself (no sub-call) |
| [raw_vlm_multi_baseline-qwen-3_5-27b.md](raw_vlm_multi_baseline-qwen-3_5-27b.md) | raw multi-image baseline |
| [react_baseline-qwen-3_5-27b.md](react_baseline-qwen-3_5-27b.md) | no-REPL ablation |
| [rlm_ocr-qwen-3_5-27b.md](rlm_ocr-qwen-3_5-27b.md) | RLM + OCR (text-only) control |
| [official_baseline-qwen-3_5-27b.md](official_baseline-qwen-3_5-27b.md) | competition-prompt anchor |
| [codeact-chat-qwen-3_5-27b.md](codeact-chat-qwen-3_5-27b.md) | corrected append-only/MDP twin of `rvlm` (chat MDP; 27B-homog + model axis; FT target) |

**Cross-model sweeps — by model** (all harnesses RLM / ReAct / CodeAct per file):

| File | Model | Covers |
|---|---|---|
| [qwen-3_5-4b.md](qwen-3_5-4b.md) | Qwen 3.5 4B reasoner | v1 homog + v2 mixed (VLM=27B), 6 cells |
| [qwen-3_5-9b.md](qwen-3_5-9b.md) | Qwen 3.5 9B reasoner | v1 homog + v2 mixed, 6 cells |
| [qwen3-8b.md](qwen3-8b.md) | Qwen3 8B (older gen, text-only) | v2 mixed only, 3 cells |
| [gemma-4-e4b.md](gemma-4-e4b.md) | Gemma-4 E4B homog | 3 harnesses + 2 baselines (n=8) — harness-lift |
| [gemma-4-31b.md](gemma-4-31b.md) | Gemma-4 31B homog | rvlm + react + codeact (n=5) + 2 baselines — harness-lift |
| [harness-axis-summary.md](harness-axis-summary.md) | — synthesis — | cross-size tables, rank-flip, v2↔v3 mechanism, cross-family read |

**Cross-benchmark / document-length axis (prediction 2)** — main solvers, Qwen
3.5 27B homog, dataset-aware profile + `use_profile_scoring=true` + raised page
budget; stratified-random subsets (`scripts/stratified_sample.py`, seed 0):

| File | Benchmark | Covers |
|---|---|---|
| [dataset-axis-mp-docvqa.md](dataset-axis-mp-docvqa.md) | MP-DocVQA (≤20pg, moderate) | rvlm / codeact_chat / official / raw_vlm_multi, ANLS |
| [dataset-axis-mmlongbench.md](dataset-axis-mmlongbench.md) | MMLongBench-Doc (~47pg, long) | same 4 solvers, Qwen-judge scored |

**Find numbers by model** (where a given model's RLM headline lives):

| Model | RLM (`rvlm`) headline | File |
|---|---|---|
| Qwen 3.5 27B | 41.88 ± 5.79 (re-run) | `rvlm-qwen-3_5-27b.md` |
| Qwen 3.5 9B | 18.91 (homog) / 25.31 (VLM=27B) | `qwen-3_5-9b.md` |
| Qwen 3.5 4B | 14.22 (homog) / 21.09 (VLM=27B) | `qwen-3_5-4b.md` |
| Qwen3 8B (older gen) | 11.73 (VLM=27B) | `qwen3-8b.md` |
| Gemma-4 31B | 33.04 ± 4.56 (homog) | `gemma-4-31b.md` |
| Gemma-4 E4B | 7.34 ± 3.30 (homog) | `gemma-4-e4b.md` |

The cross-axis summary lives in [`docs/results.md`](../results.md).

## File layout

Each experiment file has these sections:

1. **Hypothesis / question** — what's tested, one sentence; tie to a D-006
   prediction (1 model-size, 2 doc-length, 3 active-perception) where
   applicable.
2. **Setup** — solver, model, profile, max_iterations, concurrency.
3. **Command** — exact CLI, copy-pasteable. Post-D-010 solver names.
4. **Per-trial table** — `run_id`, score, correct/total, wall, contamination signals.
5. **Summary** — mean ± std, n, range. Note excluded trials.
6. **Comparison** — Δ vs the appropriate baseline + standard error; which prediction it speaks to.
7. **Observations / caveats** — surprises, infra issues, memory links.
8. **Status** — `in progress`, `done`, or `done — needs replication`.

## Conventions

- **File naming — two axes:**
  - **27B ablation matrix → by solver:** `{solver}-qwen-3_5-27b.md`, one file
    per solver, accumulating all trials. `{solver}` = canonical post-D-010 name
    incl. any `_ablation`/`_baseline` suffix; `{model}` = config slug minus the
    infra suffix (`qwen-3_5-27b-vllm-local` → `qwen-3_5-27b`). Use this whenever
    the cell is one solver on the headline 27B model.
  - **Cross-model sweeps → by model:** `{model}.md` (e.g. `qwen-3_5-4b.md`,
    `gemma-4-31b.md`), one file per model holding **all** harness/perception
    cells for that model. Use this for the model-size / VLM-quality / model-family
    sweeps — they are indexed by model, not solver, so a reader can find e.g.
    all Qwen 3.5 4B numbers in one place.
  - **Cross-cutting narratives** that span models (rank-flip with scale, the
    v2↔v3 reasoning-vs-perception factorial, cross-family reads) go in the
    single synthesis doc `harness-axis-summary.md`, which references the
    by-model files for raw numbers rather than duplicating them.
- **Run IDs** use post-D-010 solver names. Historical IDs
  (`flat-solo-*`, `leanest-*`, `no-loop-*`) stay as-is per D-010.
- **Trial budget per D-008:** cells start n=1 → n=2 if direction holds →
  n=8 only after the paper headline locks. Document the n at each stage.
- **Cross-benchmark cells** (MP-DocVQA, MMLongBench-Doc) use DA-by-default
  solvers with the `data.dataset` override + `data.use_profile_scoring=true`.
- **Coordination:** claim a cell in `coordination/<host>.md` before
  starting; don't duplicate across hosts.

## Archive

Pre-change writeups (old prompts + old whole-agent retry — numbers no
longer valid) and pre-D-010 scaffold cells live in
[`archive/experiments/`](../../archive/experiments/). Read for process
history only; do not cite as current.

## Template for new cells

```markdown
# <solver> — <model> (<split>)

## Hypothesis / question
<one sentence; tie to a D-006 prediction>

## Setup
- Solver: `<solver>`
- Model: Qwen 3.5 27B local vllm 8927, `enable_thinking=false`
- Profile: DocVQA-2026 (default)
- max_concurrency: 24

## Command
\```bash
uv run python evals.py \\
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \\
  lm.enable_thinking=false solver=<solver> \\
  data.split=val data.num_samples=null \\
  max_concurrency=24 run_id=<solver>-val-t1
\```

## Per-trial table
| Trial | run_id | Score | Correct | Wall | Notes |
|---|---|---|---|---|---|
| t1 | `<solver>-val-t1` | TBD | /80 | TBD | |

## Summary
n=1 (per D-008; escalate to n=2 if direction holds).

## Comparison
Compare against: <baseline cell>. Δ = TBD.

## Observations / caveats

## Status
in progress / done / done — needs replication
```
