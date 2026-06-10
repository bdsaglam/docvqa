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

The 8-solver comparison re-run (`lm.enable_thinking=false`, local vllm
:8927), **complete at n=8** (val 25 docs / 80 Qs). Mean ± std over 8
trials; Δ vs the `rvlm` reference. A 9th solver (`codeact`, append-only/MDP
twin) is also done (3-budget sweep). Full detail + headlines in
[`../results.md`](../results.md).

| Solver | Role | Val (n=8) | Δ vs `rvlm` | Status |
|---|---|---|---|---|
| **`rvlm`** | **proposed method** — REPL + recursive VLM `batch_look` (OCR-free) | **39.38% ± 1.49** | — | ✓ reference |
| `rvlm_ocr_ablation` | + OCR `page_texts` + BM25 `search` (OCR extension) | 37.81% ± 3.12 | −1.56pp | ✓ done — **OCR adds nothing** over OCR-free |
| `codeact` | append-only/MDP twin of `rvlm` (no compaction; FT target) | 36.74% ± 4.29 (pooled, n=23) | −2.64pp | ✓ done — compaction ~free; budget flat |
| `rvlm_hybrid_ablation` | + direct `display()` image channel on top of the sub-call | 35.47% ± 4.48 | −3.91pp | ✓ done — direct channel mildly harmful (3× variance) |
| `react_baseline` | perception tools, **no REPL** (`dspy.ReAct`) | 25.16% ± 4.60 | −14.22pp | ✓ done — REPL load-bearing |
| `direct_vlm` | `display()` pages into agent's own context, **no** VLM sub-call | 22.34% ± 2.79 | −17.03pp | ✓ done — sub-call load-bearing |
| `raw_vlm_multi_baseline` | raw multi-image, single VLM call, **no scaffold** | 20.47% ± 1.63 | −18.91pp | ✓ done — scaffold-lift floor |
| `official_baseline` | competition `MASTER_PROMPT`, multi-image, no scaffold | 17.81% ± 1.86 | −21.56pp | ✓ done — external anchor (kit-faithful: 21.67% ± 1.91) |
| `rlm_ocr` | RLM + OCR text perception, **no vision** | 13.91% ± 1.56 | −25.47pp | ✓ done — **OCR-free headline control** (matrix floor) |

**Reading:** `rvlm` (~39%) lifts ~+21pp over the no-scaffold raw-VLM floor
(~20%). The two halves of the scaffold each matter — drop the REPL
(`react` ~24%) or serve perception one-shot instead of via the recursive
sub-call (`raw_vlm_multi` ~20%, `direct_vlm` 21%) and it collapses, but
neither alone recovers `rvlm`. Adding OCR (`rvlm_ocr`) or a direct
image channel (`rvlm_hybrid`) on top of the OCR-free recursive sub-call
buys ~0 — supporting the OCR-free recursive-perception framing.

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
| Qwen 3.5 9B | 16.67% ± 3.40 | 24.54% ± 5.30 | **+7.87pp** — Welch t=3.54, 95% CI [+3.4, +12.3], **sig.** | 8 |
| Qwen 3.5 4B | 12.49% ± 3.74 | 21.09% ± 3.16 | **+8.60pp** — Welch t=4.96, 95% CI [+5.20, +11.99], **sig.** | 8 |
| Qwen3 8B (older gen) | — (n/a) | 11.73% ± 2.96 | — (off the size curve) | 8 |

At both 9B and 4B, swapping only the VLM →27B lifts ~8pp (9B +7.87,
4B +8.60) → the scaffold is **perception-budget-bound** for mid/small
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
| [codeact-qwen-3_5-27b.md](codeact-qwen-3_5-27b.md) | append-only/MDP twin of `rvlm` (budget sweep, FT target) |

**Cross-model sweeps — by model** (all harnesses RLM / ReAct / CodeAct per file):

| File | Model | Covers |
|---|---|---|
| [qwen-3_5-4b.md](qwen-3_5-4b.md) | Qwen 3.5 4B reasoner | v1 homog + v2 mixed (VLM=27B), 6 cells |
| [qwen-3_5-9b.md](qwen-3_5-9b.md) | Qwen 3.5 9B reasoner | v1 homog + v2 mixed, 6 cells |
| [qwen3-8b.md](qwen3-8b.md) | Qwen3 8B (older gen, text-only) | v2 mixed only, 3 cells |
| [gemma-4-e4b.md](gemma-4-e4b.md) | Gemma-4 E4B homog | 3 harnesses + 2 baselines (n=8) — harness-lift |
| [gemma-4-31b.md](gemma-4-31b.md) | Gemma-4 31B homog | rvlm + react + codeact (n=5) + 2 baselines — harness-lift |
| [harness-axis-summary.md](harness-axis-summary.md) | — synthesis — | cross-size tables, rank-flip, v2↔v3 mechanism, cross-family read |

**Find numbers by model** (where a given model's RLM headline lives):

| Model | RLM (`rvlm`) headline | File |
|---|---|---|
| Qwen 3.5 27B | 39.38 ± 1.49 | `rvlm-qwen-3_5-27b.md` |
| Qwen 3.5 9B | 16.67 (homog) / 24.54 (VLM=27B) | `qwen-3_5-9b.md` |
| Qwen 3.5 4B | 12.49 (homog) / 21.09 (VLM=27B) | `qwen-3_5-4b.md` |
| Qwen3 8B (older gen) | 11.73 (VLM=27B) | `qwen3-8b.md` |
| Gemma-4 31B | 32.50 ± 4.48 (homog) | `gemma-4-31b.md` |
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
