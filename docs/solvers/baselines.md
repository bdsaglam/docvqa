# Baselines

Single-call baselines used as the comparison floor for scaffold lift.
None of these use a REPL, an agent loop, or recursive sub-calls — one
forward pass, then the answer is extracted from the model output.

Per [D-006](../paper/decisions.md#d-006-reframe-paper-around-visual-context-budget-hypothesis):
the raw-VLM baselines are the main scaffold-vs-raw comparison point.
`official_baseline` is included separately because the competition kit
ships its own prompt (`MASTER_PROMPT`) — keeping it unchanged is
methodological honesty when comparing to ICDAR 2026 submissions.

| Solver | Source | Hydra config | Pages → model |
|---|---|---|---|
| [`raw_vlm_multi`](#raw_vlm_multi) | `raw_vlm_multi_solver.py` | `raw_vlm_multi.yaml` | All pages as separate images, interleaved with `[Page i]` labels |
| [`raw_vlm_single`](#raw_vlm_single) | `raw_vlm_single_solver.py` | `raw_vlm_single.yaml` | Pages stacked vertically into a single composite image |
| [`official_baseline`](#official_baseline) | `official_baseline_solver.py` | `official_baseline.yaml` | All pages as separate images (kit-style), kit's MASTER_PROMPT verbatim |

---

## `raw_vlm_multi`

Direct VLM Q&A with native multi-image input — pages are passed as
separate images, interleaved with `[Page i]` labels and the question.
No tools, no scaffold. This is the **canonical raw-VLM baseline** for
the scaffold-vs-raw lift.

- **Paper role:** Raw-VLM baseline (multi-image).
- **Engineering name only** per
  [D-010](../paper/decisions.md#d-010-solver-renames--behavior-based-engineering-names).
- Dataset-aware per
  [D-009](../paper/decisions.md#d-009-refine-d-007--split-semantic-per-profile-from-tool-routing-per-solver):
  the dataset profile supplies `answer_formatting_rules`,
  `baseline_category_tips`, and `question_format_hint`.

### Tool surface

None. One VLM forward pass.

### Prompt composition (per D-009)

- Body (`_TASK_BODY` in `raw_vlm_multi_solver.py`): documents the input
  format (pages as a labelled sequence).
- `answer_formatting_rules`: from the profile.
- Per-category baseline tips: from `profile.baseline_category_tips_fn`
  (smaller surface than the agent-mode tips — the baseline does not
  have a tool surface to route around).
- Per-question format hint: from `profile.question_format_hint_fn`.

### Headline results

| Split | Solver | Score | Notes |
|---|---|---|---|
| DocVQA-2026 val | `raw_vlm_multi` (Qwen 3.5 27B) | 20.0% | n=8 SC-8, split-calibration anchor |
| DocVQA-2026 test | `raw_vlm_multi` (Qwen 3.5 27B) | 11.0% | n=8 SC-8 |

The split-calibration check (val→test gap ~9pp at no-scaffold) shows
the headline gap is dominated by split difficulty, not by prompt
overfitting.

### Command

```bash
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  solver=raw_vlm_multi solver.dataset=${data.dataset} \
  data.split=val data.num_samples=null \
  max_concurrency=16 run_id=raw-vlm-multi-val
```

---

## `raw_vlm_single`

Direct VLM Q&A on a **vertically-stacked composite image** of all
pages. Single forward pass. Defensible single-call baseline for
providers that don't expose native multi-image input.

- **Paper role:** Raw-VLM baseline (single-image variant).
- **Engineering name only** per D-010.
- Dataset-aware per D-009.

### Tool surface

None. One VLM forward pass with a composite image.

### Prompt composition (per D-009)

Same shape as `raw_vlm_multi` — body documents the single-composite
input format; everything else comes from the dataset profile.

### When to use

- Side-by-side with `raw_vlm_multi` to control for the multi-image vs
  composite-image input format.
- Providers with no native multi-image VLM API.

### Command

```bash
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  solver=raw_vlm_single solver.dataset=${data.dataset} \
  data.split=val data.num_samples=null \
  max_concurrency=16 run_id=raw-vlm-single-val
```

---

## `official_baseline`

Literal ICDAR 2026 DocVQA kit baseline. Sends all document pages to the
VLM in one chat-completion request, uses the kit's `MASTER_PROMPT`
verbatim (the competition's mandatory-reasoning protocol + `FINAL
ANSWER:` output format), and extracts the text after `FINAL ANSWER:`.

The prompt is vendored from
`tmp/DocVQA2026/eval_utils.py:get_evaluation_prompt()` — refresh
manually if upstream changes.

- **Paper role:** Competition baseline (comparable to ICDAR 2026
  leaderboard entries).
- **Unchanged by D-009 refactor** — no dataset profile, no category
  tips, no per-question hint. The kit prompt is the contract.
- **Source:** `src/docvqa/solvers/official_baseline_solver.py`.

### Tool surface

None. One VLM forward pass, kit prompt verbatim.

### Notes

- No truncation by default — matches how the kit's Gemini / GPT
  baselines are scored in the README results table.
- `PIL.Image.MAX_IMAGE_PIXELS` is set to `None` to match the kit's
  decompression-bomb policy. The largest test page (maps_5 p0, 246M
  pixels) fits under the other solvers' 500M cap so this is a parity
  setting, not a fix.

### Command

```bash
uv run python evals.py lm=vertex_ai/gemini-3-pro-preview \
  solver=official_baseline \
  data.split=val data.num_samples=null \
  max_concurrency=4 run_id=official-baseline-val
```

## See also

- [`rvlm`](rvlm.md) — proposed method (the scaffold side of the
  scaffold-vs-raw comparison).
- [`rvlm_ocr`](rvlm-ocr.md) — proposed method + OCR extension.
- [`docs/paper/README.md`](../paper/README.md) — headline numbers and
  taxonomy.
