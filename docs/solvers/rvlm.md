# `rvlm` — Proposed Method

Code-capable LLM in a REPL whose only perception channel is a recursive
VLM sub-call via `batch_look`. Per
[D-006](../paper/decisions.md#d-006-reframe-paper-around-visual-context-budget-hypothesis),
this is the paper's headline solver — the OCR-free instantiation of the
Recursive Language Models paradigm specialized to document VQA.

- **Source:** `src/docvqa/solvers/rvlm_solver.py`
- **Hydra config:** `configs/solver/rvlm.yaml`
- **Paper role:** Proposed method (M)
- **Engineering name only** — paper-facing name TBD per
  [D-005](../paper/decisions.md#d-005-position-vs-rvlm-and-madqa) /
  [D-010](../paper/decisions.md#d-010-solver-renames--behavior-based-engineering-names).

## Tool surface

| Tool | Signature | Notes |
|---|---|---|
| `batch_look` | `batch_look(requests: list[tuple[PIL.Image, str]]) -> list[str]` | Parallel VLM calls. Each request is `(image, query)` where `image` can be a full page (`pages[i]`), a crop (`pages[i].crop((l,t,r,b))`), or any PIL Image. Single-call idiom: `batch_look([(img, q)])[0]`. |

Plus the REPL itself (Python + `SUBMIT(answer=...)`), and the pre-loaded
`pages: list[PIL.Image]`.

No `look()`, no `search()`, no `page_texts` in scope. The agent's only
channel into the document is active VLM perception.

## When to use

- Headline cell on every benchmark in the paper — this is the proposed
  method.
- Any dataset where OCR isn't trustworthy or where the lift hypothesis
  is being tested without confounds from a retrieval channel.

For the OCR-extension cell, use [`rvlm_ocr`](rvlm-ocr.md) (clean fork
with `search` + `page_texts` added, *no* single-image `look()`
ergonomic wrapper). For the kitchen-sink ablation, see
[`rvlm_full`](rvlm-full.md).

## Architecture (one diagram)

```
question + doc_info
        |
        v
   RLM (LeanRLM / CodeRLM / ThinkingRLM)
        |  REPL sandbox: pages[], batch_look(), SUBMIT()
        |
        +-- batch_look([(pages[i], q1), (pages[i].crop(...), q2), ...])
        |     |
        |     v
        |  VLM (recursive sub-call) — one call per request, in parallel
        |
        +-- ...iterate...
        |
        v
   SUBMIT(answer="...")
```

Per [D-001](../paper/decisions.md#d-001-frame-the-paper-as-an-application-of-rlm):
the architecture instantiates RLM (Zhang, Kraska, Khattab, arXiv:2512.24601)
with the recursive sub-call specialized as a VLM.

## Prompt composition (per D-009)

[D-009](../paper/decisions.md#d-009-refine-d-007--split-semantic-per-profile-from-tool-routing-per-solver)
splits prompt ownership two ways:

| Layer | Owner | Content |
|---|---|---|
| Tool surface body (`_TASK_BODY`) | solver (`rvlm_solver.py`) | Documents `batch_look` only, the REPL, the approach. No per-category content. |
| `answer_formatting_rules` | dataset profile | Substituted into the body at the trailing `## OUTPUT FORMAT` block. |
| Per-category semantic tips | dataset profile (`category_tips_fn`) | E.g. "for engineering drawings, verify each label is correctly associated with the part it connects to." Tool-agnostic. |

`rvlm` does **not** declare its own `TOOL_HINTS` overlay — `batch_look`
is the only tool, so there's nothing to route. The composed instructions
are `body + profile.answer_formatting_rules + ("\n" + profile.category_tips_fn(cat) if any)`.

## Configuration

| Hydra key | Default | Notes |
|---|---|---|
| `solver` | `rvlm` | Hydra config choice |
| `rlm_type` | `lean` | `lean` / `code` / `thinking` — `lean` is the headline |
| `max_iterations` | `25` | Plus a square-root page bonus capped at +10 |
| `page_factor` | `1.5` | Multiplier on the page bonus |
| `question_concurrency` | `4` | Questions solved concurrently per doc |
| `batch_concurrency` | `8` | Workers per `batch_look` request (clamped to 2 on Vertex) |
| `dataset` / `profile_name` | (auto) | Pass `solver.dataset=${data.dataset}` to pick up the right profile |

## Headline results

| Split | Solver | Score | Notes |
|---|---|---|---|
| DocVQA-2026 val | `rvlm` (lean, no-think, Qwen 3.5 27B) | 48.8% | n=8 SC-8, scrubbed prompts |
| DocVQA-2026 test | `rvlm` (lean, no-think, Qwen 3.5 27B) | **39.0%** | n=8 SC-8 |

Per
[D-006](../paper/decisions.md#d-006-reframe-paper-around-visual-context-budget-hypothesis):
this is the proposed method's headline. The paper reports mean ± std
across the 8 individual test trials per
[D-003](../paper/decisions.md#d-003-drop-self-consistency-from-the-papers-method-framing),
not the SC-8 voted score.

## Command (DocVQA-2026 val, n=1)

```bash
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=rvlm solver.dataset=${data.dataset} \
  data.split=val data.num_samples=null \
  max_concurrency=16 run_id=rvlm-val
```

## Strengths

- Minimal tool surface — easier to attribute lift to the recursive
  sub-call mechanism (clean cell for predictions 1 and 3 in D-006).
- No OCR dependency — immune to bad-OCR confounds.
- Lower token usage than `rvlm_ocr` / `rvlm_full`.

## Weaknesses

- No textual retrieval — long documents require many `batch_look` calls
  to scan thoroughly. The OCR-extension cell ([`rvlm_ocr`](rvlm-ocr.md))
  exists precisely for long-doc benchmarks (MMLongBench-Doc, MP-DocVQA
  11-20pp bucket).
- Requires a code-capable main LLM. Per
  [D-006](../paper/decisions.md#d-006-reframe-paper-around-visual-context-budget-hypothesis)
  prediction 1, lift scales with model code-writing capability.

## See also

- [`rvlm_ocr`](rvlm-ocr.md) — clean OCR extension (`batch_look` +
  `search` + `page_texts`).
- [`rvlm_full`](rvlm-full.md) — kitchen-sink (adds `look()` ergonomic
  wrapper on top of `rvlm_ocr`).
- [`direct_vlm`](direct-vlm.md) — single-multimodal-model alt angle.
- [`repl_only`](ablations.md#repl_only) — VLM-sub-call-off ablation.
- [`rvlm_unified`](ablations.md#rvlm_unified) — category-dispatch
  ablation.
