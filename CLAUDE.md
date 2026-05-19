# DocVQA 2026

ICDAR 2026 DocVQA competition. RLM agents with active document perception.

## Best Results

| Config | Val | Test |
|--------|-----|------|
| Flat Solo SC-8 (lean, no-think, Qwen 3.6 27B / Qwen 3.6 27B) | **51.2%** | **43.75%** |
| Flat Solo SC-8 (lean, no-think, Qwen 3.5 27B / Qwen 3.5 27B) | 51.2% | 39.0% |
| Flat Solo single run (lean, no-think, Qwen 3.5) | 48.8% | 35.6% |
| Flat Batch (Pro+Flash) | 55.0% | — |
| Gemini 3 Pro baseline | 37.5% | 37.5% |

## Best Config Per Solver

| Solver | Config Override | Best Val |
|--------|----------------|----------|
| **Flat Solo** | `solver=flat_solo solver.rlm_type=lean` | **48.8%** |
| Leanest Solo | `solver=leanest_solo` | 43.8% |
| Lean Solo | `solver=lean_solo` | 42.5% |
| Flat Batch | `solver=flat_batch` | 37.5% |
| Ensemble | `solver=ensemble_lean_solo` | — |

**IMPORTANT**: `flat_solo` yaml default is `rlm_type=code` (~40%). ALWAYS override to `lean`.

## Infrastructure

- **LLM**: `vertex_ai/gemini-3-pro-preview` (best) or `qwen-3_5-27b` (local)
- **VLM**: Qwen/Qwen3.5-27B at localhost:8927 (3x A100 GPUs)
- **OCR data**: `data/docvqa-2026/{split}/ocr/{doc_id}/page_*.md` (new dataset layout: `data/{dataset-slug}/{split}/...`)
- **BM25 indexes**: auto-built per doc during eval

## Key Commands

```bash
# Best single-run solver
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=flat_solo solver.rlm_type=lean \
  data.split=val data.num_samples=null max_concurrency=16 run_id=flat-solo-val

# Ensemble (5x lean solo)
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=ensemble_lean_solo data.split=val data.num_samples=null \
  max_concurrency=15 run_id=ens-val

# Generate report
python scripts/report.py --all --min-questions 80 --recent 7
```

## Key Findings

1. **Solo >> Batch**: ~10pp gap — one question at a time is much better
2. **Lean RLM > Code RLM** for solo: lean+nothink = 46.2%, code+think = 40.0%
3. **Thinking hurts lean**: 38.8% (think) vs 41.6% mean (nothink)
4. **High variance**: ~3-4% std across trials — always run 3+ trials
5. **Per-category tips** in `src/docvqa/prompts.py` help precision-heavy categories

## Cross-benchmark methodology rule (critical)

When reporting baseline-vs-scaffold on **any benchmark other than
DocVQA-2026**, the baseline must use a **dataset-aware profile** and
a **fair page budget**, or the scaffold lift double-counts prompt fit
+ truncation as scaffold capability.

Concretely:

- Use `*_da` solver variants (`solver=no_loop_multi_da`,
  `solver=flat_solo_da`, `solver=leanest_solo_da`). They pull from
  `docvqa.datasets.profile.get_profile(dataset)` for prompt, tips,
  per-question hint, and scorer.
- Pass `data.use_profile_scoring=true` so the runner uses the
  profile's `score_fn` (e.g. Qwen judge for MMLongBench) instead of
  ANLS.
- On long-doc benchmarks, override `solver.max_pages=80` (or the
  loader's `DEFAULT_MAX_PAGES`) so the raw-VLM baseline can see the
  evidence pages. The default `max_pages=10` is fine for short-doc
  benchmarks.

**Empirical evidence (2026-05-14, Qwen 3.5 27B, 200Q val samples):**

| Benchmark | Legacy lift | Fair lift (DA + pages) | Δ from baseline crippling |
|---|---|---|---|
| MP-DocVQA (ANLS) | −4.88pp (leanest "regresses") | **~0pp** | +5pp |
| MMLongBench-Doc (judge) | +26.43pp (leanest) | **+14.81pp** | +11.6pp |
| MMLongBench-Doc (judge) | +26.60pp (flat_solo) | **+16.84pp** | +9.8pp |

About half the MMLongBench legacy headline came from baseline
crippling (+5pp from max_pages=10→80, +8pp from DocVQA-2026 prompt →
MMLongBench profile). The MP-DocVQA legacy "regression" was 100%
prompt mismatch.

See `docs/experiments/mp-docvqa-qwen27b.md` and
`docs/experiments/mmlongbench-doc-qwen27b.md` for the full closed-loop
numbers and `src/docvqa/datasets/profile.py` for the registered
profiles.

## Project Structure

| File | Purpose |
|------|---------|
| `evals.py` | Hydra entry point |
| `src/docvqa/solvers/` | Solver implementations |
| `src/docvqa/solvers/*_da_solver.py` | Dataset-aware variants (profile-driven) |
| `src/docvqa/datasets/profile.py` | `DatasetProfile` + `get_profile(dataset_id)` |
| `src/docvqa/prompts.py` | DocVQA-2026 answer-formatting rules + per-category tips |
| `src/docvqa/data.py` | Dataset loading, OCR integration |
| `src/docvqa/runner.py` | Eval runner (concurrent, resumable; accepts profile `score_fn`) |
| `src/docvqa/metrics.py` | ANLS evaluation |
| `scripts/report.py` | Results report from run IDs |

## GCP Credits

- EDU Credit: ~41K/43.8K remaining (94%) — expires March 2027
- Gen App Builder: 41.5K — expires October 2026
