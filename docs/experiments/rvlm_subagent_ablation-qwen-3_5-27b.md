# rvlm_subagent_ablation — Qwen 3.5 27B (val)

## Hypothesis / question

Fork of `rvlm` that generalizes the recursive sub-call from a narrow
**perception** tool (`batch_look((image, query))`) into a general
**delegation** tool (`batch_subagent((task, image))`). The sub-agent is
still a single multimodal-model call, but the main agent is told it can
delegate *any* well-scoped subtask — visual *or* not (image is optional;
`None` for a text/reasoning subtask). The framing is **balanced**: it does
not bias the agent toward visual or non-visual delegation.

Question: does a general task-decomposition / delegation tool help the
main agent solve more (by breaking the problem into subtasks and
delegating each) than the perception-specific `batch_look` — or does
`rvlm`'s narrow perception sub-call already capture the benefit?

## Setup

- Solver: `rvlm_subagent_ablation` (`batch_subagent`; sub-agent = the VLM, multimodal, image optional)
- Model: Qwen 3.5 27B local vllm 8927 (lm + vlm/sub-agent), `enable_thinking=false`
- Profile: DocVQA-2026 (default); max_iterations 25 (same as `rvlm`)
- max_concurrency: 24 (high-concurrency phase)
- Added 2026-06-04 (user request); n=8.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_subagent_ablation \
  data.split=val data.num_samples=null \
  max_concurrency=24 \
  run_id=rvlm-subagent-cmp-val-tN
```

## Per-trial table

| Trial | run_id | Score | Correct | Wall | Notes |
|---|---|---|---|---|---|
| — | (queued) | — | — | — | n=8 queued (after codeact + nocrop) |

## Summary

n=0 (queued). Mean ± std at n=8; paired Δ vs `rvlm` (39.38% ± 1.49)
tests whether general delegation beats the perception-only sub-call.

## Comparison

vs `rvlm` (perception-only `batch_look`) — same scaffold/model, only the
sub-call's interface + framing change. Also worth checking: how often does
the main agent actually delegate **non-visual** subtasks (image=None), and
does its iteration count (`iter_stats.py`) differ from `rvlm`?

## Status

queued (n=0 of 8)
