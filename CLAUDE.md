# DocVQA 2026

ICDAR 2026 DocVQA competition. RLM agents with active document perception.

## Best Results

| Config | Val | Test |
|--------|-----|------|
| Flat Solo SC-8 (lean, no-think, Qwen 3.6 27B / Qwen 3.6 27B) | **51.2%** | **43.75%** |
| Flat Solo SC-8 (lean, no-think, Qwen 3.5 27B / Qwen 3.5 27B) | 51.2% | 41.0% |
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
- **OCR data**: `data/{split}/ocr/{doc_id}/page_*.md`
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

## Project Structure

| File | Purpose |
|------|---------|
| `evals.py` | Hydra entry point |
| `src/docvqa/solvers/` | Solver implementations |
| `src/docvqa/prompts.py` | Answer formatting rules + per-category tips |
| `src/docvqa/data.py` | Dataset loading, OCR integration |
| `src/docvqa/runner.py` | Eval runner (concurrent, resumable) |
| `src/docvqa/metrics.py` | ANLS evaluation |
| `scripts/report.py` | Results report from run IDs |

## GCP Credits

- EDU Credit: ~41K/43.8K remaining (94%) — expires March 2027
- Gen App Builder: 41.5K — expires October 2026
