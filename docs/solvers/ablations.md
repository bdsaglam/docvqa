# Ablations

Solvers that exist to test a specific mechanism claim, not to win a
benchmark. Per
[D-006](../paper/decisions.md#d-006-reframe-paper-around-visual-context-budget-hypothesis),
one ablation solver lives here.

| Solver | Source | Tests | Paper status |
|---|---|---|---|
| [`rvlm_unified`](#rvlm_unified) | `rvlm_unified_solver.py` | Does the scaffold need per-document category metadata? | Paper-strengthening cell if Δ ≈ 0pp |

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
