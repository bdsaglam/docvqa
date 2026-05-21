# Flat Solo DA — Multi-Image VLM Solver

**Status:** Approved (design)
**Date:** 2026-05-21
**Author:** brainstorming session (bdsaglam + Claude)
**Target solver name:** `flat_solo_da_mi` (multi-image)

## Motivation

The existing `flat_solo_da_solver.py` exposes a VLM that takes exactly one
image per call. That forces the agent into sequential lookups when a question
naturally calls for cross-image reasoning — e.g. "find the shape in this
cropped patch on the larger page," "are these two charts using the same
legend," or feeding the VLM a sequence of patches to inspect together. Each
of those would currently be a chain of single-image `look()` calls with the
agent reconciling the results in Python.

We want a sibling solver that lets the VLM see multiple images in a single
call and answer one query about them together. The intent is general — we
do not want to pin the prompt to a fixed list of "supported uses."

## Scope

In scope:

- New solver module `src/docvqa/solvers/flat_solo_da_mi_solver.py`.
- New Hydra config `configs/solver/flat_solo_da_mi.yaml`.
- Cropping mode AND page-only mode, mirroring `flat_solo_da_solver`'s
  ablation coverage (`vlm_cropping=True|False`).

Out of scope (intentional):

- New ablation knobs (e.g. max images per call).
- Automated tests beyond a smoke run on a few val docs after implementation.
- Prompt examples that enumerate uses — tool docstrings stay generic.

## Tool API

Tool signatures are **overloaded**: the existing single-image / single-index
form continues to work; the new list form is additive. The agent can pass a
single value or a list and the sandbox helpers normalize.

**Cropping mode** (`vlm_cropping=True`):

```python
look(image_or_images, query)               # PIL.Image OR list[PIL.Image]
batch_look([(image_or_images, query), ...]) # each request: same overload
```

**Page-only mode** (`vlm_cropping=False`):

```python
look(page_idx_or_list, query)              # int OR list[int]
batch_look([(page_idx_or_list, query), ...]) # each request: same overload
```

Singletons are accepted purely for backward compatibility with prompt
examples and existing agent muscle memory — internally everything is a list.

## Implementation outline

### New / forked pieces

- **VLM signature.** `dspy.Predict` over
  `(images: list[dspy.Image], query: str) -> answer: str`. Always a list
  internally; the singleton path wraps `[image]` before calling.
- **`_look_impl(image_paths: list[str], query) -> str`.** Loads N images,
  passes them as a `list[dspy.Image]` to `vlm_predict`. Logfire span records
  the count and a truncated query.
- **`_batch_look_impl`.** Same `ThreadPoolExecutor` pattern as today, but
  each request carries `paths: list[str]` instead of a single `path`.
  Concurrency limit derives from the VLM provider the same way it does now
  (vertex → 2, else → 8).
- **Sandbox builders.**
  - `_build_sandbox_code_mi(page_dir, num_pages, use_search)` — defines
    `look(image_or_images, query)` and `batch_look(requests)` that normalize
    to lists, write each image to a temp PNG, and forward paths to the tool
    bridge.
  - `_build_sandbox_code_page_only_mi(page_dir, num_pages, use_search)` —
    defines `look(page_idx_or_list, query)` and `batch_look(requests)` that
    normalize ints to `[int]`, validate range, and forward the precomputed
    page paths.
- **Prompt bodies.** Clones of `_CROPPING_BODY` and `_PAGE_ONLY_BODY` from
  `flat_solo_da_solver`, with the `look` and `batch_look` lines rewritten
  to document the overloaded list-or-singleton signature. Neutral wording —
  no closed list of uses.
- **`FlatSoloDAMIProgram`.** Same shape as `FlatSoloDAProgram`; only its
  `vlm_predict` signature and the sandbox / tool factories it calls differ.
- **`create_flat_solo_da_mi_program(...)` factory.** Same parameters as
  `create_flat_solo_da_program`; no new knobs.

### Reused as-is (imported, not copied)

From `docvqa.solvers.flat_solo_solver`: `RunContext`, `_format_page_texts`,
`_build_signature`, `_strip_search_tool`.

From `docvqa.solvers.flat_solo_da_solver`: nothing imported — but the new
solver mirrors its `_build_task_instructions(profile, vlm_cropping)` helper
and its `_per_question_prefix(q)` method exactly.

### Hydra config

`configs/solver/flat_solo_da_mi.yaml`: near-copy of
`configs/solver/flat_solo_da.yaml` with `_target_` pointing to
`docvqa.solvers.flat_solo_da_mi_solver.create_flat_solo_da_mi_program` and
the same defaults. Used via `solver=flat_solo_da_mi` on the CLI.

## File layout

```
src/docvqa/solvers/flat_solo_da_mi_solver.py   # new
configs/solver/flat_solo_da_mi.yaml            # new
```

No existing files are modified.

## Verification

After implementation:

1. Smoke run: a handful of val docs through the new solver in both cropping
   and page-only modes. Confirm the agent successfully issues at least one
   multi-image `look()` call when the question warrants it (inspect a
   trajectory or two). Numbers don't need to beat baseline at this stage —
   we just need the path to work end-to-end.
2. Confirm Hydra instantiation via the new config: `uv run python evals.py
   solver=flat_solo_da_mi data.num_samples=2 …`.

We are **not** running a full eval as part of this design — that's a
follow-up experiment, not part of building the solver.

## Risks / open items

- **VLM provider support.** DSPy accepts `list[dspy.Image]` as a field type
  (verified during brainstorming). Per-provider behavior of multi-image
  payloads (Qwen 3.5 27B local vLLM vs Vertex Gemini) is assumed-OK but
  will be confirmed during the smoke run.
- **Image count blow-up.** A single `look()` with too many large images
  could exhaust context. We are deliberately not adding a guard rail yet;
  the smoke run will tell us whether the agent over-uses it.
