# repl_only_baseline — Qwen 3.5 27B (val)

## Hypothesis / question

The no-perception floor: a fork of `rvlm` with the VLM sub-call
(`batch_look`) removed and **no** replacement perception — no vision
tools, no OCR, no document-text tools. The agent keeps the Python REPL
but has zero access to document content. Isolates how much of `rvlm`'s
score comes from *perception* vs from the REPL/reasoning shell + answer
priors. Complement of `react_baseline` (which keeps perception but
removes the REPL).

## Setup

- Solver: `repl_only_baseline` (REPL only; no vision, no OCR, no text)
- Model: Qwen 3.5 27B local vllm 8927 (lm; vlm unused), `enable_thinking=false`
- Profile: DocVQA-2026 (default)
- max_concurrency: 24
- Part of the 7-solver comparison re-run (val, n=3), 2026-06-01.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=repl_only_baseline \
  data.split=val data.num_samples=null \
  max_concurrency=24 \
  run_id=repl-only-cmp-val-tN
```

## Per-trial table

| Trial | run_id | Score | Correct | Wall | Notes |
|---|---|---|---|---|---|
| t1 | `repl-only-cmp-val-t1` | 7.50% | 6/80 | ~4 min | no-perception floor; most predictions "Unknown"; fast (no VLM/OCR loop) |

Per-category (t1): business_report 0/10, comics 0/10, engineering_drawing
0/10, infographics 2/10, maps 0/10, science_paper 2/10, science_poster
0/10, slide 2/10.

## Summary

n=1 so far (t2, t3 queued). Expected to stay near the prior/guessing
floor — the agent cannot see the document at all.

## Comparison

This is the perception-floor anchor. `rvlm` t1 (40.00%) − this (7.50%) =
**+32.50pp** — nearly all of `rvlm`'s score is perception, not the
reasoning shell. (Contrast `react_baseline`, which keeps perception but
drops the REPL.) Lock paired Δ at n=3.

## Observations / caveats

- Completes in ~4 min because there is no perception loop — the agent
  quickly concludes it cannot answer and returns "Unknown" for most
  questions. The few correct (infographics/science_paper/slide) are
  answerable from question text / priors alone.
- By-design blindness (verified in the solver: no `look`/`batch_look`,
  no `search`/`page_texts`/OCR) — the low score is the intended control,
  not a misconfiguration.

## Status

in progress (n=1 of 3)
