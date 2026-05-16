# Official baseline (MASTER_PROMPT) — Qwen 3.5 27B

**Hypothesis:** evaluate the literal DocVQA 2026 published baseline
(`get_evaluation_prompt()` from `tmp/DocVQA2026/eval_utils.py`) on
Qwen 3.5 27B for a direct comparison to the kit's published numbers
(Gemini 3 Pro 37.5%, GPT-5.2 35.0%, Gemini 3 Flash 33.75%, GPT-5
Mini 22.5%). Anchors "what does the open mid-tier model do under the
same prompt and setup the closed frontier baselines used?"

**Setup:** new solver `official_baseline_solver.py` with the vendored
`MASTER_PROMPT` (verbatim from the kit). Single VLM call per
question, all document pages as multi-image input, no agent loop,
no tools, no category tips. Parses the `FINAL ANSWER:` line out of
the model response. `Image.MAX_IMAGE_PIXELS = None` to mirror the
kit. Qwen 3.5 27B (vLLM local 8927).

`max_pages = null` — kit-faithful: send all pages, accept context
overflows as `Unknown` per the kit README: *"If a sample fails
because the input files are too large, the result counts as a
failure."*

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local \
  vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=official_baseline \
  data.split={val|test} data.num_samples=null \
  max_concurrency=24 \
  run_id=official-baseline-3_5-27b-{val|test}-t${i}
```

## Val results (n=3)

| Trial | Accuracy | Unknown rate | Context-overflow failures (Unknown ← BadRequestError) |
|---|---|---|---|
| t1 | 20.00% | 56.2% (45/80) | 44 |
| t2 | 23.75% | 57.5% (46/80) | 43 |
| t3 | 21.25% | 51.2% (41/80) | 39 |

- **mean = 21.67% ± 1.91pp** (n=3, tight std)
- per-trial = [20.0, 23.8, 21.2]
- range = 20.0-23.8

## Caveat: context-overflow failures are doing most of the work

On Qwen 3.5 27B's 131k context window, ~52-58% of val questions hit
"Input length exceeds max context" errors when all pages of a long
doc (`business_report`, `comics`, `engineering_drawing`, `maps`)
are stacked into the request. Per the kit policy these score as
`Unknown`. The remaining 33-43 questions that DO fit are answered
at roughly the model's true capacity.

If we had truncated to first 10 pages (`max_pages=10`), most
context errors go away but val accuracy stays similar (~18.75%
on t1, 19 errors). So the "raw model capability on the questions
it can read" hovers around 35-50% — but that's not the kit's
official baseline number, it's a Qwen-adapted one.

## Comparison to kit's published baselines

The kit reports baselines on the **test** set (we have val above —
test runs in progress as #25). Side-by-side:

| Model | Test accuracy (kit) | Context fit | Notes |
|---|---|---|---|
| Gemini 3 Pro Preview | 37.5% | ✓ (≥1M ctx) | published |
| GPT-5.2 | 35.0% | mostly ✓ | published |
| Gemini 3 Flash Preview | 33.75% | ✓ | published |
| GPT-5 Mini | 22.5% | partial (50MB input limit per README) | published |
| **Qwen 3.5 27B (this work)** | **TBD** | partial (131k ctx) | val 21.67% ± 1.91pp; test running |

Qwen 27B + official prompt lands in GPT-5 Mini's neighborhood, NOT
the larger-context models'. The val gap to Gemini Pro (≈16pp) and
even to Flash (≈12pp) is real but partly attributable to context
budget rather than capability. The model is being asked to do
something it can't fit in its head.

## Implication for the paper

This experiment makes a useful narrative point: a 27B open-weight
model with the **same prompt** as the kit baselines is bottlenecked
by context, not reasoning. The scaffold (`flat_solo`/`leanest`)
sidesteps that bottleneck by exposing pages individually to the
VLM, which is precisely why scaffolded Qwen 27B (44.69% val,
38.75% test SC-8) blows past raw VLM baselines. *"Active perception
lets a small-context model handle long documents that overflow it
under the kit's single-call multi-image setup."*

## Implementation notes

- Solver: `src/docvqa/solvers/official_baseline_solver.py`
- Config: `configs/solver/official_baseline.yaml`
- MASTER_PROMPT vendored verbatim from `tmp/DocVQA2026/eval_utils.py:get_evaluation_prompt()`
- BadRequestError → `Unknown` with empty trajectory (visible as
  `trajectories[qid] = []` in result.json)

## Status

- Val: **done** (n=3, 21.67% ± 1.91pp)
- Test: in progress (t1 running, target n≥3)
