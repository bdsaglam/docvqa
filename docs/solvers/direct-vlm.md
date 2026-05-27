# `direct_vlm` — Single-Multimodal-Model Alt Angle

A single multimodal LLM in a REPL with no recursive sub-call. The agent
calls `display(image)` to push a PIL Image inline into its own
conversation history; the multimodal model sees it natively in the next
iteration. No `look()`, no `batch_look()`, no VLM delegation —
perception is **direct** via the LLM's own multimodal channel.

Per
[D-010](../paper/decisions.md#d-010-solver-renames--behavior-based-engineering-names):
"direct" pairs lexically with "recursive" in `rvlm`. This solver is the
alt-angle method cell — same REPL affordance as `rvlm`, but the
recursive VLM sub-call is replaced by direct multimodal perception by
the main model.

- **Source:** `src/docvqa/solvers/direct_vlm_solver.py`
- **Hydra config:** `configs/solver/direct_vlm.yaml`
- **Paper role:** Alternative-angle method (requires multimodal LLM)
- **Engineering name only** per
  [D-005](../paper/decisions.md#d-005-position-vs-rvlm-and-madqa) /
  [D-010](../paper/decisions.md#d-010-solver-renames--behavior-based-engineering-names).

## Tool surface

| Tool | Signature | Notes |
|---|---|---|
| `display` | `display(image: PIL.Image)` | Push a PIL Image into the conversation; the multimodal LLM sees it in the next step. No VLM sub-call. |
| `print` | `print(...)` | Standard Python print — values flow back as text. |

Plus the REPL (Python + `SUBMIT(answer=...)`) and the pre-loaded
`pages: list[PIL.Image]`.

## When to use

- Multimodal main LLM available (Gemini 3 Pro, etc.). Does **not** work
  with text-only models — there's no recursive VLM sub-call to delegate
  perception to.
- The alt-angle paper cell: tests whether the lift mechanism is the
  *agent-loop + REPL* (this solver retains both) versus specifically
  the *recursive VLM sub-call* (which `rvlm` has and this does not).

## Architecture

```
question + doc_info
        |
        v
   RVLM agent (multimodal LLM)  ←─ same model class on every iteration
        |  REPL sandbox: pages[], display(), print(), SUBMIT()
        |
        +-- display(pages[i])           # full page pushed into context
        +-- display(pages[i].crop(...)) # crop pushed into context
        +-- ...iterate; recent N images retained in window...
        |
        v
   SUBMIT(answer="...")
```

The `RVLM` class (`docvqa.rlm.RVLM`) keeps images from the last
`images_for_last_n` iterations in the multimodal conversation (default
1) and downsamples to `max_image_pixels` to control context size.

## Prompt composition (per D-009)

Per [D-009](../paper/decisions.md#d-009-refine-d-007--split-semantic-per-profile-from-tool-routing-per-solver):

| Layer | Owner | Content |
|---|---|---|
| Tool surface body (`_TASK_BODY`) | solver (`direct_vlm_solver.py`) | Documents `display`, the REPL, the approach. |
| `answer_formatting_rules` | dataset profile | Substituted into the body. |
| Per-category semantic tips | dataset profile (`category_tips_fn`) | Tool-agnostic. |
| Per-category `TOOL_HINTS` overlay | solver (`direct_vlm_solver.TOOL_HINTS`) | `display()`-specific tool-routing examples for all 8 DocVQA-2026 categories. Composed on top of profile tips via `_get_category_tips`. |

## Configuration

| Hydra key | Default | Notes |
|---|---|---|
| `solver` | `direct_vlm` | Hydra config choice |
| `max_iterations` | `20` |  |
| `images_for_last_n` | `1` | Number of recent iterations whose images are kept in context |
| `max_image_pixels` | `1_000_000` | Downsample threshold |
| `use_category_tips` | `true` | Toggle the per-category overlay |
| `question_concurrency` | `4` |  |
| `dataset` / `profile_name` | (auto) | DocVQA-2026 default |

`vlm` config is accepted but unused — `direct_vlm` does not delegate to a
separate VLM. The main `lm` must be multimodal.

## Command (Gemini 3 Pro, DocVQA-2026 val)

```bash
uv run python evals.py lm=gemini-3_1-pro-vertex solver=direct_vlm \
  solver.dataset=${data.dataset} \
  data.split=val data.num_samples=null \
  max_concurrency=4 run_id=direct-vlm-val
```

## Strengths

- No VLM call overhead — perception is in-band, one model, one
  inference engine.
- Conceptually simpler than RLM-with-recursive-sub-call — useful as a
  contrast cell in the paper's mechanism story.

## Weaknesses

- Requires a multimodal main LLM. Cuts open-model coverage to the
  multimodal subset.
- Context grows with images — the `images_for_last_n` /
  `max_image_pixels` knobs are necessary to control cost.
- Active visual perception is constrained by the multimodal model's
  own resolution / context — no separate high-resolution VLM call.

## See also

- [`rvlm`](rvlm.md) — proposed method (recursive VLM sub-call).
- [`rvlm_full`](rvlm-full.md) — kitchen-sink (different tool surface).
- [`raw_vlm_multi` / `raw_vlm_single`](baselines.md) — raw-VLM baselines
  without any REPL or agent loop.
