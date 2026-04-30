# Routing Solver

Routes documents to different solver implementations based on category.

## How It Works

```
Document (category=engineering_drawing)
        |
        v
   RoutingSolver._pick_solver()
        |
        +-- category in visual_categories? -> flat_solo
        +-- category in category_overrides? -> per-category solver
        +-- otherwise -> flat_batch (default)
        |
        v
   chosen_solver.solve_document(document)
```

## Configuration

```yaml
# configs/solver/routing.yaml
_target_: docvqa.solvers.routing_solver.create_routing_solver
default_type: flat_batch
default_config:
  iterations_per_question: 4
  base_iterations: 6
  vlm: ${vlm}
  rlm_type: lean
  page_factor: 1.5
  max_iterations: 40
visual_type: flat_solo
visual_config:
  max_iterations: 30
  vlm: ${vlm}
  rlm_type: lean
  page_factor: 1.5
  question_concurrency: 4
visual_categories:
  - engineering_drawing
  - business_report
  - comics
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `default_type` | Solver for unmapped categories (e.g. `flat_batch`) |
| `default_config` | Config dict passed to the default solver factory |
| `visual_type` | Solver for visual-heavy categories (e.g. `flat_solo`) |
| `visual_config` | Config dict for the visual solver |
| `visual_categories` | Categories to route to the visual solver |
| `category_overrides` | Per-category overrides: `{category: {type, config}}` |

## Usage

```bash
# Default routing: visual categories → flat_solo, rest → flat_batch
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=routing data.split=val data.num_samples=null \
  max_concurrency=16 run_id=routing-val

# Custom per-category overrides
uv run python evals.py solver=routing \
  'solver.category_overrides={maps: {type: leanest_solo, config: {max_iterations: 25, vlm: ${vlm}, rlm_type: lean}}}' \
  data.split=val data.num_samples=null run_id=routing-custom
```

## Supported Solver Types

| Type | Description |
|------|-------------|
| `flat_solo` | Best solo solver (46.2% val) |
| `leanest_solo` | Minimal tool solo (43.8% val) |
| `lean_solo` | Lean solo with search (42.5% val) |
| `flat_batch` | Batch baseline (37.5% val) |
