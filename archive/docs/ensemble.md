# Ensemble Solver

Voting approach. Runs N copies of a solver and takes majority vote per question.

## Architecture

Wraps any solver that implements `solve_document()`. Runs N independent copies
of the base solver on the same document, collects all predictions, then selects
the most common answer per question. Ties are broken by first occurrence.

- **Wraps**: any solver (e.g., `lean_solo`, `flat_solo`, `leanest_solo`)
- **Voting**: majority vote per question; ties broken by first occurrence
- **Parallelism**: configurable `max_parallel` controls concurrent solver
  instances
- Default configuration: 5x `lean_solo` with `max_parallel=5`

## Configuration

| Parameter       | Value                |
|----------------|----------------------|
| `solver`       | `ensemble_lean_solo` |
| `num_solvers`  | 5                    |
| `max_parallel` | 5                    |
| `lm`           | `qwen27b`            |
| `vlm`          | `qwen27b`            |
| `enable_thinking` | `false`           |

Each underlying solver inherits its own config from the base solver definition.

## Results

Accuracy improves with more voters, with diminishing returns after ~5.

| Voters | Notes                                |
|--------|--------------------------------------|
| 1      | Single run, no voting benefit        |
| 3      | Moderate improvement                 |
| 5      | Good accuracy/variance tradeoff      |
| 8      | Used for test submission (38%)       |

## Command

```bash
uv run python evals.py lm=qwen27b vlm=qwen27b lm.enable_thinking=false \
  solver=ensemble_lean_solo data.split=val data.num_samples=null \
  max_concurrency=15 run_id=ens-val
```

## Strengths

- Reduces variance from stochastic Qwen LLM outputs (~3-5% std per trial)
- Majority vote filters out random errors from individual runs
- No code changes needed -- wraps existing solvers
- Scales compute for accuracy: more voters = more stable predictions

## Weaknesses

- N times the compute cost of a single run
- Diminishing returns after ~5 voters
- Does not fix systematic errors (if all voters get a question wrong,
  voting cannot help)
- Requires `max_concurrency` high enough to run voters in parallel for
  reasonable wall time
