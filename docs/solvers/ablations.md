# Ablations

Solvers that exist to test a specific mechanism claim, not to win a
benchmark. Per
[D-006](../paper/decisions.md#d-006-reframe-paper-around-visual-context-budget-hypothesis),
two ablation solvers live here.

| Solver | Source | Tests | Paper status |
|---|---|---|---|
| [`repl_only`](#repl_only) | `repl_only_solver.py` | Is the recursive VLM sub-call the load-bearing mechanism? | Documentation only — not a paper cell |
| [`rvlm_unified`](#rvlm_unified) | `rvlm_unified_solver.py` | Does the scaffold need per-document category metadata? | Paper-strengthening cell if Δ ≈ 0pp |

---

## `repl_only`

Fork of [`rvlm`](rvlm.md) with the recursive VLM sub-call (`batch_look`)
removed. The agent keeps the REPL + Python execution + `SUBMIT` but
has **no image perception** (and no OCR text channel — this is a *pure*
no-perception ablation, not an "OCR-only" one).

- **Source:** `src/docvqa/solvers/repl_only_solver.py`
- **Hydra config:** `configs/solver/repl_only.yaml`
- **Paper role:** Not a paper cell. Documentation-only.

### Tool surface

| Tool | Notes |
|---|---|
| Python REPL | Standard library, `SUBMIT(answer=...)`. |

No `batch_look`, no `look`, no `search`, no `page_texts`. The agent
cannot see or read the document.

### What it tests

Per D-006 prediction 3: the lift comes from *active, iterative* VLM
sub-calls. If you strip the VLM sub-call but keep the REPL and agent
loop, the lift should collapse to the raw-VLM baseline floor.

### Smoke result (2026-05-27, Qwen 3.5 27B)

On a 2-doc smoke (5 questions), the agent SUBMITs `"Unknown"` in 1
iteration per question — `0/5`. The result is mechanistically obvious
("no perception → no answer") and tests a strawman rather than a sharp
prediction. Per D-006, the reframing uses three existing ablations
(cropping-off −7.81pp, m=5 turn budget −15pp vs m=30, `rvlm` 48.8% vs
`raw_vlm_multi` 20.0%) as the mechanism evidence instead. `repl_only`
stays in the tree as documentation.

### Prompt composition (per D-009)

- Body explicitly tells the agent it has no perception tools and that
  the correct default answer is `"Unknown"`.
- `answer_formatting_rules` and `category_tips` still come from the
  dataset profile so the only thing this solver changes vs `rvlm` is
  the tool surface (the inherited "crop the legend" semantic content is
  useless without perception — that's the point).

### Command

```bash
uv run python evals.py lm=qwen-3_5-27b-vllm-local \
  solver=repl_only solver.dataset=${data.dataset} \
  data.split=val data.num_samples=2 \
  max_concurrency=4 run_id=repl-only-smoke
```

---

## `rvlm_unified`

Identical to [`rvlm`](rvlm.md) except it ignores `document.doc_category`
and injects the **concatenation of all 8 DocVQA-2026 category tip
blocks** as the per-question prompt overlay, instead of dispatching per
document.

- **Source:** `src/docvqa/solvers/rvlm_unified_solver.py`
- **Hydra config:** `configs/solver/rvlm_unified.yaml`
- **Paper role:** Category-dispatch ablation. Defends against the
  reviewer objection "this only works because you hand it the category
  as metadata."

### Tool surface

Same as [`rvlm`](rvlm.md) — `batch_look` only.

### What it tests

Whether the scaffold requires per-document category metadata or whether
a unified prompt suffices. Expected outcomes (from
`docs/experiments/unified-category-tips-ablation.md`):

| Δ vs `rvlm` | Interpretation |
|---|---|
| ≈ 0pp (within noise) | Per-category dispatch is procedural convenience, not load-bearing. **Strongest paper outcome** — promote unified to default. |
| −1pp to −3pp | Small prompt-length cost; still paper-strengthening ("our method does not require category labels"). |
| ≤ −5pp | Category dispatch IS load-bearing; honest caveat in paper. |

### Prompt composition (per D-009)

Pulls `_DOCVQA_2026_CATEGORY_TIPS` from
`docvqa.datasets.profile` and concatenates all 8 blocks at import time.
Per D-009, the *content* lives in the profile (single source of truth);
this solver merely presents it without dispatching. The body and
`answer_formatting_rules` are identical to `rvlm`.

### Command

```bash
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=rvlm_unified solver.dataset=${data.dataset} \
  data.split=val data.num_samples=null \
  max_concurrency=16 run_id=rvlm-unified-val
```

See [`docs/experiments/unified-category-tips-ablation.md`](../experiments/unified-category-tips-ablation.md)
for the full hypothesis and experimental design.

## See also

- [`rvlm`](rvlm.md) — the reference cell both ablations fork from.
- [`docs/paper/decisions.md` — D-006](../paper/decisions.md#d-006-reframe-paper-around-visual-context-budget-hypothesis)
  — the three existing mechanism-evidence ablations.
