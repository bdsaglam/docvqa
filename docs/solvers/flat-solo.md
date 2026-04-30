# Flat Solo Solver

Highest single-run score. Full tool suite per question with direct VLM access.

## Architecture

Each question gets its own RLM session with the complete tool suite. No
sub-agents, no hierarchy -- just the agent, its tools, and the full iteration
budget.

- **Tools**:
  - `look(image, query)` -- single VLM call on any PIL Image
  - `batch_look([(image, query)])` -- parallel VLM calls
  - `search(query, k)` -- BM25 search over OCR text
- **Inputs**: `page_texts` (OCR text per page) and `pages` (PIL Images)
- Agent can search OCR text first, then visually inspect relevant pages
- Full iteration budget dedicated to each question independently

## Configuration

| Parameter    | Value                | Note                          |
|-------------|----------------------|-------------------------------|
| `solver`    | `flat_solo`          |                               |
| `rlm_type`  | `lean`               | **MUST override** -- yaml default is `code` |
| `max_iterations` | 30              |                               |
| `lm`        | `qwen-3_5-27b-vllm-local` |                               |
| `vlm`       | `qwen-3_5-27b-vllm-local` |                               |
| `enable_thinking` | `false`        |                               |

**IMPORTANT**: The yaml default is `rlm_type=code` which gives ~40%. Always
override to `lean` for the best results (~46%).

## Results

| Split | Score       | Notes                        |
|-------|-------------|------------------------------|
| val   | 46.2% (m25) | lean, no-think -- highest single run ever |

## Command

```bash
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=flat_solo solver.rlm_type=lean data.split=val data.num_samples=null \
  max_concurrency=16 run_id=flat-solo-val
```

## Strengths

- Highest single-run accuracy of any solver
- Search tool enables targeted lookups in text-heavy documents
- Full budget per question -- no competition for iterations
- OCR + visual cross-referencing reduces hallucination

## Weaknesses

- Per-question RLM session overhead (process spawn, PIL loading)
- Slower than leanest_solo on documents where search is not helpful
- BM25 index build time on first run per document
