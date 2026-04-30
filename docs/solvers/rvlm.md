# RVLM Solver

Multimodal agent that sees images directly via `display()`. No VLM tool calls needed.

## How It Works

Unlike other solvers that use `look()` / `batch_look()` to send images to a separate VLM, the RVLM agent calls `display(image)` to show a PIL Image inline in the conversation. The multimodal LLM (e.g., Gemini Pro) sees it natively in the next iteration.

```
Question + Document
        |
        v
  RVLM Agent (multimodal LLM)
        |
        |-- display(pages[0])      # show full page
        |-- display(crop)          # show cropped region
        |-- search(query)          # BM25 text search
        |-- SUBMIT(answer)
```

**Key difference**: The agent itself is multimodal — it sees images directly, not through a VLM intermediary. This eliminates the VLM call overhead but requires a multimodal-capable LLM.

## Configuration

```yaml
# configs/solver/rvlm.yaml
_target_: docvqa.solvers.rvlm_solver.create_rvlm_program
max_iterations: 20
images_for_last_n: 3        # include images from last N iterations in context
max_image_pixels: 8000000   # downsample large images
use_category_tips: true
question_concurrency: 4
```

## Usage

```bash
# With Gemini Pro (multimodal)
uv run python evals.py lm=gemini-3_1-pro-vertex solver=rvlm \
  data.split=val data.num_samples=null \
  max_concurrency=4 run_id=rvlm-val
```

**Note**: Requires a multimodal LLM. Does not work with text-only models like Qwen.
