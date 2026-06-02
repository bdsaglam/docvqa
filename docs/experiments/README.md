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

The 7-solver comparison re-run (n=3 target, `lm.enable_thinking=false`,
local vllm :8927). Most cells are mid-run — numbers below are the current
per-trial means with explicit trial counts; lock at n=3.

| Solver | Role | Val (current) | n | Status |
|---|---|---|---|---|
| **`rvlm`** | **proposed method** — REPL + recursive VLM `batch_look` sub-call (OCR-free) | **~39.4%** (40.00, 38.75) | 2/3 | in progress (reference; all Δ measured vs this) |
| `rvlm_hybrid_ablation` | + direct `display()` image channel on top of the sub-call | ~39.4% (40.00, 38.75) | 2/3 | in progress — **Δ ≈ 0** vs `rvlm` (direct image access redundant) |
| `rvlm_ocr_ablation` | + OCR `page_texts` + BM25 `search` (OCR extension) | **37.08% ± 2.60** | 3/3 ✓ | done — **OCR adds nothing** over OCR-free, trends ~−2pp |
| `direct_vlm` | "see it yourself": `display()` pages into the agent's own context, **no** VLM sub-call | 21.25% | 1/3 | in progress — Δ ≈ **+18.75pp** for the sub-call |
| `raw_vlm_multi_baseline` | raw multi-image, single VLM call, **no scaffold** | ~20.6% (18.75, 22.50) | 2/3 | in progress — scaffold-lift floor (~+21pp) |
| `react_baseline` | perception tools, **no REPL** (`dspy.ReAct`) | ~23.8% (17.50, 30.00) | 2/3 | in progress (high variance) — REPL load-bearing |
| `rlm_ocr` | RLM + OCR text perception, **no vision** | — | 0/3 | queued — the OCR-free claim's text-modality control |
| `official_baseline` | competition `MASTER_PROMPT`, multi-image, no scaffold | — | 0/3 | queued — external anchor (prior kit-faithful: 21.67% ± 1.91, n=3) |

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
n=8 per arm, val. Detail: [`qwen-9b-rvlm-minimal-vlm-axis.md`](qwen-9b-rvlm-minimal-vlm-axis.md).

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

| File | What |
|---|---|
| [rvlm-qwen-3_5-27b.md](rvlm-qwen-3_5-27b.md) | proposed method (reference) |
| [rvlm_ocr_ablation-qwen-3_5-27b.md](rvlm_ocr_ablation-qwen-3_5-27b.md) | OCR extension ablation |
| [rvlm_hybrid_ablation-qwen-3_5-27b.md](rvlm_hybrid_ablation-qwen-3_5-27b.md) | hybrid display+sub-call ablation |
| [direct_vlm-qwen-3_5-27b.md](direct_vlm-qwen-3_5-27b.md) | see-it-yourself (no sub-call) |
| [raw_vlm_multi_baseline-qwen-3_5-27b.md](raw_vlm_multi_baseline-qwen-3_5-27b.md) | raw multi-image baseline |
| [react_baseline-qwen-3_5-27b.md](react_baseline-qwen-3_5-27b.md) | no-REPL ablation |
| [rlm_ocr-qwen-3_5-27b.md](rlm_ocr-qwen-3_5-27b.md) | RLM + OCR (text-only) control |
| [official_baseline-qwen-3_5-27b.md](official_baseline-qwen-3_5-27b.md) | competition-prompt anchor |
| [qwen-9b-rvlm-minimal-vlm-axis.md](qwen-9b-rvlm-minimal-vlm-axis.md) | VLM-quality axis (9B + 4B), n=8 |

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

- **File name:** `{solver}-{model}.md` — one file per solver×model pairing,
  accumulating all trials. `{solver}` = canonical post-D-010 name incl.
  any `_ablation`/`_baseline` suffix; `{model}` = config slug minus the
  infra suffix (`qwen-3_5-27b-vllm-local` → `qwen-3_5-27b`).
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
