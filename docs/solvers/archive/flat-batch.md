# Flat Batch Solver

Batch baseline. All questions processed in a single RLM session.

## Architecture

ALL questions are provided at once as a JSON list. The agent must explore the
document, plan its approach, solve each question sequentially, then submit all
answers together. Direct VLM access with no sub-agents.

- **Tools**:
  - `look(image, query)` -- single VLM call
  - `batch_look([(image, query)])` -- parallel VLM calls
  - `search(query, k)` -- BM25 search
- **Inputs**: `page_texts` and `pages` -- same as flat_solo
- Iteration budget scales with question count:
  `base_iterations + iterations_per_question * num_questions`
- Agent prompt enforces: EXPLORE -> PLAN -> SOLVE SEQUENTIALLY -> SUBMIT ALL

## Configuration

| Parameter              | Value     |
|-----------------------|-----------|
| `solver`              | `flat_batch` |
| `rlm_type`            | `lean`    |
| `max_iterations`      | 40        |
| `base_iterations`     | 6         |
| `iterations_per_question` | 4     |
| `lm`                  | `qwen-3_5-27b-vllm-local` |
| `vlm`                 | `qwen-3_5-27b-vllm-local` |
| `enable_thinking`     | `false`   |

## Results

| Split | Score       | Notes                        |
|-------|-------------|------------------------------|
| val   | 37.5%       | lean, no-think               |

## Command

```bash
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=flat_batch data.split=val data.num_samples=null \
  max_concurrency=16 run_id=flat-batch-val
```

## Strengths

- Single session -- no per-question overhead
- Fastest overall wall time for a full document
- Shared exploration -- agent can reuse VLM observations across questions

## Weaknesses

- ~5-10pp lower than solo solvers due to question interference
- Agent struggles to give equal attention to all questions
- "Solve sequentially" is hard to enforce -- agent tends to juggle questions
- Iteration budget shared across all questions; early questions consume
  disproportionate budget
- Later questions may run out of iterations before being answered properly
