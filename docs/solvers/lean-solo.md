# Lean Solo Solver

Like leanest solo but adds search capability and OCR page text.

## Architecture

Single question per session. Adds `search()` and `page_texts` on top of the
leanest solo visual pipeline. Agent can locate relevant content via BM25 search
before visually inspecting pages.

- **Tools**:
  - `look(image, query)` -- single VLM call on any PIL Image
  - `batch_look([(image, query)])` -- parallel VLM calls
  - `search(query, k)` -- BM25 search over OCR text
- **Inputs**: `page_texts` (OCR text per page) -- agent can search before
  looking at images
- `look()` for sequential reads, `batch_look()` for parallel reads
- Full iteration budget dedicated to each question

## Configuration

| Parameter    | Value                |
|-------------|----------------------|
| `solver`    | `lean_solo`          |
| `rlm_type`  | `lean`               |
| `max_iterations` | 25              |
| `lm`        | `qwen-3_5-27b-vllm-local` |
| `vlm`       | `qwen-3_5-27b-vllm-local` |
| `enable_thinking` | `false`        |

## Results

| Split | Score       | Notes                        |
|-------|-------------|------------------------------|
| val   | 42.5%       | lean, no-think               |

## Command

```bash
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=lean_solo data.split=val data.num_samples=null \
  max_concurrency=16 run_id=lean-solo-val
```

## Strengths

- Search tool helps on text-heavy documents (papers, reports)
- OCR text gives agent a roadmap before committing VLM calls
- More efficient than flat_solo when search alone locates the answer

## Weaknesses

- Lower peak score than flat_solo (42.5% vs 46.2%)
- BM25 search adds index build overhead
- OCR artifacts can mislead the agent if not cross-checked visually
- Slightly higher token usage than leanest_solo due to page_texts
