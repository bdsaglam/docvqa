# `rvlm_ocr` — Proposed Method + OCR Extension

Clean OCR extension of [`rvlm`](rvlm.md): adds `search` (BM25 over OCR
text) and `page_texts` to the tool surface, **without** the
single-image `look()` ergonomic wrapper that confounds attribution in
[`rvlm_full`](rvlm-full.md). Per
[D-006](../paper/decisions.md#d-006-reframe-paper-around-visual-context-budget-hypothesis):
this is the dedicated M+OCR cell — the paper reports it as an
extension, not as a core contribution.

- **Source:** `src/docvqa/solvers/rvlm_ocr_solver.py`
- **Hydra config:** `configs/solver/rvlm_ocr.yaml`
- **Paper role:** OCR extension of the proposed method
- **Engineering name only** per
  [D-010](../paper/decisions.md#d-010-solver-renames--behavior-based-engineering-names).

## Tool surface

| Tool | Signature | Notes |
|---|---|---|
| `batch_look` | `batch_look(requests: list[tuple[PIL.Image, str]]) -> list[str]` | Parallel VLM calls — same as `rvlm`. Single-call idiom: `batch_look([(img, q)])[0]`. |
| `search` | `search(query: str, k: int = 5) -> list[dict]` | BM25 over per-page OCR markdown. Returns `[{page, score, text}]`. |
| `page_texts` | input | OCR-extracted text per page (list of strings, 0-indexed). Available as a signature input field. |

Plus the REPL (Python + `SUBMIT(answer=...)`) and the pre-loaded
`pages: list[PIL.Image]`.

**No `look()` is registered.** The agent never sees a single-image
wrapper in its REPL; for one query it uses
`batch_look([(image, query)])[0]`. This is the property that makes
`rvlm_ocr` the clean OCR-extension cell — any lift over `rvlm` is
attributable to the OCR channel, not to the `look()` ergonomic delta.

## When to use

- Headline OCR-extension cell on long-document benchmarks where the
  retrieval channel pays off: MMLongBench-Doc (+2pp on top of `rvlm`),
  MP-DocVQA 11-20pp bucket.
- Any cell where you want to attribute lift to OCR specifically without
  the `rvlm_full` confound.

For the no-OCR proposed method, use [`rvlm`](rvlm.md). For the
kitchen-sink with `look()` + OCR, see [`rvlm_full`](rvlm-full.md).

## Architecture

Per-question RLM session. Pages saved to a tempdir; BM25 index built
once per document from the OCR markdown (`docvqa.search.get_or_build_index`).
The agent's REPL has `pages: list[PIL.Image]` pre-loaded plus
`batch_look()` and `search()` as proxies. `page_texts` is in scope as a
signature input field on the LLM call.

## Prompt composition (per D-009)

Per [D-009](../paper/decisions.md#d-009-refine-d-007--split-semantic-per-profile-from-tool-routing-per-solver):

| Layer | Owner | Content |
|---|---|---|
| Tool surface body (`_TASK_BODY`) | solver (`rvlm_ocr_solver.py`) | Documents `batch_look`, `search`, `page_texts`, the REPL, the approach. |
| `answer_formatting_rules` | dataset profile | Substituted into the body. |
| Per-category semantic tips | dataset profile (`category_tips_fn`) | Tool-agnostic. |
| Per-category `TOOL_HINTS` overlay | solver (`rvlm_ocr_solver.TOOL_HINTS`) | OCR-routing examples for `science_paper`, `slide`, `infographics` — the categories where BM25 + `page_texts` is the dominant strategy. Composed on top of profile tips via `_get_category_tips`. |

The `TOOL_HINTS` overlay mirrors the historical `FLAT_SOLO_TOOL_HINTS`
from `rvlm_full`, but the routing verbs reference `batch_look()` (this
solver's surface) rather than `look()`.

## Configuration

| Hydra key | Default | Notes |
|---|---|---|
| `solver` | `rvlm_ocr` | Hydra config choice |
| `rlm_type` | `lean` | `lean` / `code` / `thinking` |
| `max_iterations` | `20` (+ page bonus, capped at +10) |  |
| `page_factor` | `1.5` | Multiplier on the page bonus |
| `question_concurrency` | `4` (factory) / `1` (program default) | Questions per doc solved concurrently |
| `batch_concurrency` | `8` |  |
| `dataset` / `profile_name` | (auto) | DocVQA-2026 default |

## Command (DocVQA-2026 val)

```bash
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=rvlm_ocr solver.dataset=${data.dataset} \
  data.split=val data.num_samples=null \
  max_concurrency=16 run_id=rvlm-ocr-val
```

## Strengths

- Clean attribution for the OCR extension — no `look()` confound vs
  `rvlm`.
- BM25 retrieval pays off on long documents where exhaustive visual
  scanning is wasteful.
- Same per-question RLM session pattern as `rvlm` — fork is minimal.

## Weaknesses

- BM25 index build overhead on first run per document.
- OCR artifacts in `page_texts` can mislead the agent if not
  cross-checked visually (the prompt body explicitly tells the agent to
  verify critical values via `batch_look`).
- Higher token usage than `rvlm`.

## See also

- [`rvlm`](rvlm.md) — proposed method, no OCR.
- [`rvlm_full`](rvlm-full.md) — kitchen-sink with `look()` ergonomic
  wrapper (confounded for OCR attribution).
- [`direct_vlm`](direct-vlm.md) — single-multimodal-model alt angle.
