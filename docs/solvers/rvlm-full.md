# `rvlm_full` — Kitchen-Sink Variant

`rvlm` + a single-image `look()` ergonomic wrapper + `search` + OCR
`page_texts`. The full tool suite per question.

Per [D-006](../paper/decisions.md#d-006-reframe-paper-around-visual-context-budget-hypothesis):
this solver bundles OCR **and** the `look()` ergonomic wrapper, so it is
**not** the clean OCR extension — that's [`rvlm_ocr`](rvlm-ocr.md).
`rvlm_full` is reported (if at all) as a kitchen-sink appendix cell;
its paper role is TBD per task #16.

- **Source:** `src/docvqa/solvers/rvlm_full_solver.py`
- **Hydra config:** `configs/solver/rvlm_full.yaml`
- **Paper role:** Kitchen-sink (appendix, role TBD)
- **Engineering name only** per
  [D-010](../paper/decisions.md#d-010-solver-renames--behavior-based-engineering-names).

## Tool surface

| Tool | Signature | Notes |
|---|---|---|
| `look` | `look(image: PIL.Image, query: str) -> str` | Single-image VLM call. Ergonomic wrapper over a `batch_look` of one. Sole reason this solver differs from `rvlm_ocr`. |
| `batch_look` | `batch_look(requests: list[tuple[PIL.Image, str]]) -> list[str]` | Parallel VLM calls. |
| `search` | `search(query: str, k: int = 5) -> list[dict]` | BM25 over per-page OCR markdown. Returns `[{page, score, text}]`. |
| `page_texts` | input | OCR-extracted text per page (list of strings, 0-indexed). Available as a signature input field. |

Plus the REPL (Python + `SUBMIT(answer=...)`) and the pre-loaded
`pages: list[PIL.Image]`.

The D-004 cropping ablation is implemented here: setting
`vlm_cropping=False` swaps the sandbox for a page-only variant where
`look` / `batch_look` accept only `page_idx: int`, not arbitrary PIL
Images. The `use_search=False` ablation strips the `search` tool and
its prompt mention.

## When to use

- Legacy comparison cell — historical numbers on DocVQA-2026 used this
  solver under its old name.
- D-004 cropping ablation (`vlm_cropping=False`) — page-only restriction
  test.
- Use-search-on/off ablation.

For the clean OCR extension, use [`rvlm_ocr`](rvlm-ocr.md). For the
no-OCR proposed method, use [`rvlm`](rvlm.md).

## Architecture

Same per-question RLM session as `rvlm`, plus `page_texts` as a signature
input, BM25 index built per document, and `look()` / `search()`
registered as additional tool proxies. The `look()` wrapper is what
distinguishes this solver from `rvlm_ocr` — it's a confound for any
"OCR adds X" attribution because lift could come from the ergonomic
single-image call instead of the OCR channel.

## Prompt composition (per D-009)

Per [D-009](../paper/decisions.md#d-009-refine-d-007--split-semantic-per-profile-from-tool-routing-per-solver):

| Layer | Owner | Content |
|---|---|---|
| Tool surface body | solver (`rvlm_full_solver.py`) — `_CROPPING_BODY` (default) or `_PAGE_ONLY_BODY` (D-004 ablation) | Documents `look`, `batch_look`, `search`, `page_texts`. |
| `answer_formatting_rules` | dataset profile | Substituted into the body. |
| Per-category semantic tips | dataset profile (`category_tips_fn`) | Tool-agnostic per-category guidance. |
| Per-category `TOOL_HINTS` overlay | solver (`rvlm_full_solver.TOOL_HINTS`) | Tool-routing examples for `science_paper`, `slide`, `infographics` — the categories where BM25 + `page_texts` is the dominant strategy. Appended to profile tips at the call site via `_get_category_tips`. |

The solver also exports `TASK_INSTRUCTIONS` (default DocVQA-2026 +
cropping) as a back-compat seed for the shelved `flat_solo_gepa`
optimizer.

## Configuration

| Hydra key | Default | Notes |
|---|---|---|
| `solver` | `rvlm_full` | Hydra config choice |
| `rlm_type` | `lean` | **Must override** — yaml default historically was `code` (~40%); always run with `lean` (`solver.rlm_type=lean`) |
| `max_iterations` | `20` (+ page bonus, capped at +10) |  |
| `vlm_cropping` | `true` | `false` activates the D-004 page-only ablation |
| `use_search` | `true` | `false` strips the `search` tool (and its prompt line) |
| `question_concurrency` | `4` (factory) / `1` (program default) | Questions per doc solved concurrently |
| `page_factor` | `1.5` | Multiplier on the page bonus |
| `dataset` / `profile_name` | (auto) | DocVQA-2026 default |

## Historical results

| Split | Solver | Score | Notes |
|---|---|---|---|
| DocVQA-2026 val | `rvlm_full` (lean, no-think, Qwen 3.5 27B) | 47.5% | n=8 SC-8, scrubbed prompts |
| DocVQA-2026 test | `rvlm_full` (lean, no-think, Qwen 3.5 27B) | 38.0% | n=8 SC-8 |

Per
[D-006](../paper/decisions.md#d-006-reframe-paper-around-visual-context-budget-hypothesis):
`rvlm` (no OCR) edges this on both val (48.8%) and test (39.0%) — OCR's
value concentrates on long-doc benchmarks (MMLongBench-Doc +2pp on top
of `rvlm`).

## Command (DocVQA-2026 val)

```bash
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=rvlm_full solver.rlm_type=lean solver.dataset=${data.dataset} \
  data.split=val data.num_samples=null \
  max_concurrency=16 run_id=rvlm-full-val
```

## Strengths

- Largest tool surface — useful when you want the agent to have every
  available channel.
- The D-004 cropping and search-on/off ablations live here.

## Weaknesses

- Confounded for attributing OCR lift — the `look()` wrapper is an
  ergonomic delta on top of `batch_look`, so any lift over `rvlm` could
  be either OCR or ergonomics. Use [`rvlm_ocr`](rvlm-ocr.md) instead
  for the clean OCR-extension cell.
- Higher token usage than `rvlm` (BM25 search + OCR `page_texts` in
  context).
- BM25 index build on first run per document.

## See also

- [`rvlm`](rvlm.md) — proposed method (`batch_look` only).
- [`rvlm_ocr`](rvlm-ocr.md) — clean OCR extension (no `look()`
  wrapper).
- [`direct_vlm`](direct-vlm.md) — single-multimodal-model alt angle.
