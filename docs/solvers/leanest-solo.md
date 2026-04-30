# Leanest Solo Solver

Best overall solver. Pure visual perception with minimal tooling.

## Architecture

One tool only: `batch_look`. No page text, no search. The agent must find
everything by looking at the document images directly.

- **Tools**: `batch_look([(image, query)])` -- parallel VLM calls accepting any
  PIL Image (full pages, crops, regions)
- **No `page_texts` input** -- no OCR text provided to the agent
- **No `search()` tool** -- no BM25 index lookup
- Agent discovers all information through visual inspection via batch_look

## Configuration

| Parameter    | Value                |
|-------------|----------------------|
| `solver`    | `leanest_solo`       |
| `rlm_type`  | `lean`               |
| `max_iterations` | 25              |
| `lm`        | `qwen-3_5-27b-vllm-local` |
| `vlm`       | `qwen-3_5-27b-vllm-local` |
| `enable_thinking` | `false`        |

## Results

| Split | Score       | Notes                       |
|-------|-------------|------------------------------|
| val   | 43.8% (m25) | lean, no-think               |
| test  | 38%         | 8-vote majority ensemble     |

## Command

```bash
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=leanest_solo data.split=val data.num_samples=null \
  max_concurrency=16 run_id=leanest-solo-val
```

## Strengths

- Minimal prompt complexity -- agent focuses on visual extraction
- No OCR dependency -- immune to bad OCR artifacts
- Lower token usage than solvers with search/page_texts
- Fast iteration -- single tool, no context switching

## Weaknesses

- Cannot leverage OCR text for keyword-based lookup
- Large documents require many batch_look calls to scan thoroughly
- No text search means agent must visually locate every answer
