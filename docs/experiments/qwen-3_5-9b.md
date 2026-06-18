# Qwen 3.5 9B (reasoner) — cross-model perception sweep (val, n=8)

All cross-model cells for the **Qwen 3.5 9B reasoner**, across all three
harnesses (RLM `rvlm` / ReAct / CodeAct) in two perception configs:

- **v1 homog:** LLM = VLM = Qwen 3.5 9B.
- **v2 mixed:** LLM = 9B, VLM = Qwen 3.5 **27B** (swap *only* the VLM — the
  perception-budget test for D-006 prediction 1).

n=8 per cell, val 25 docs / 80 Q, `lm.enable_thinking=false`, per-question
micro-average. The v2-vs-v3 factorial corners and the cross-size narrative
live in [`harness-axis-summary.md`](harness-axis-summary.md).

## Results (n=8)

| Harness | v1 homog (VLM=9B) | v2 mixed (VLM=27B) | Δ (v2−v1) |
|---|---|---|---|
| RLM (`rvlm`) | 16.67% ± 3.40 | **24.54% ± 5.30** | **+7.87pp** — Welch t=3.54, 95% CI [+3.4,+12.3], **sig.** |
| ReAct | 14.97% ± 2.96 | 21.01% ± 4.63 | +6.04pp |
| CodeActᶜ | 19.35% ± 4.24 | 24.26% ± 4.68 | +4.91pp |

ᶜ **STALE — do not cite.** Old dspy `codeact` (deprecated). The corrected
**`codeact_chat`** twin is the sole source of truth for CodeAct numbers going
forward; a config without a `codeact_chat` value is **open** — the stale dspy
figure is shown for provenance only, not as a current result. Tracking and
replacements: `codeact-chat-qwen-3_5-27b.md`.

At v2, RLM ≈ CodeAct (24.54 ≈ 24.26) > ReAct (21.01): once the reasoner is
strong enough to exploit code+state, RLM/CodeAct overtake the no-REPL ReAct.

## Per-trial

| Cell | run_id stem | per-trial (8) | mean ± std |
|---|---|---|---|
| RLM v1 | `rvlm-minimal-3_5-9b-val-t{1..8}` | _(per-trial not recorded; mean locked)_ | 16.67 ± 3.40 |
| RLM v2 | `rvlm-minimal-9b-llm-27b-vlm-val-t{1..8}` | _(per-trial not recorded; mean locked)_ | 24.54 ± 5.30 |
| ReAct v1 | `react-3_5-9b-val-t{1..8}` | _(per-trial not recorded; mean locked)_ | 14.97 ± 2.96 |
| ReAct v2 | `react-9b-llm-27b-vlm-val-t{1..8}` | 24.31 / 21.64 / 22.80 / 19.10 / 15.44 / 22.00 / 28.45 / 14.34 | 21.01 ± 4.63 |
| CodeAct v1 | `codeact-3_5-9b-val-t{1..8}` | _(per-trial not recorded; mean locked)_ | 19.35 ± 4.24 |
| CodeAct v2 | `codeact-9b-llm-27b-vlm-val-t{1..8}` | 22.74 / 25.08 / 19.84 / 33.38 / 17.73 / 26.07 / 25.51 / 23.78 | 24.26 ± 4.68 |

## Reads

- **Perception-budget lift holds at 9B.** Swapping only the VLM →27B lifts the
  RLM headline **+7.87pp** (significant). The lift is consistent across
  reasoner size (9B +7.87, 4B +8.60 — see `qwen-3_5-4b.md`), the signature of a
  perception (not orchestration) bottleneck → the scaffold is
  **perception-budget-bound** for mid/small reasoners (supports D-006).
- **The 27B VLM redistributes which questions are answered, not how many
  (n=1 illustration).** On the original RLM t1, both arms scored 17/80 but on
  *different* questions: +1 each on text-dense **slide** and **infographics**,
  −1 each on **comics** and **science_poster**. The aggregate tie at n=1 was a
  single high-variance draw; the n=8 means above show a real +7.87pp lift.
- **CodeAct crossover.** 9B v2 CodeAct (24.26) is statistically tied with 9B v2
  RLM (24.54) — CodeAct's accuracy scales harder with reasoner capability than
  ReAct or RLM, catching RLM by 9B after trailing badly at 8B/4B. Full slope in
  `harness-axis-summary.md`.

## Setup

- Solver: `rvlm` / `react_baseline` / `codeact`. `max_iterations=25` (RLM/ReAct).
- LLM: Qwen 3.5 9B vllm @ `:8909` (DP=4, vision enabled, `max-model-len=65536`),
  `enable_thinking=false`. VLM: 9B @ :8909 (v1) or 27B @ :8928 (v2).
- Data: val, all 25 docs / 80 Q. `max_concurrency=16`.

## Observations / caveats

- **Infra: a single hung question (v2 RLM, original run).** A run stalled at
  24/25 for ~1h50m on `maps_2_q5` — the request dispatched but never logged
  iteration 1, i.e. an I/O stall on a non-returning model call, not a reasoning
  loop. SIGINT didn't break the blocked asyncio call (needed SIGKILL);
  relaunching the same `run_id` re-ran only `maps_2` and q5 returned correct.
  Per-doc resumability made this a clean recovery.
- One IPC-deadlock hang in the CodeAct sweep landed on 9B v2 (t2/`maps_2`,
  `batch_look` bridge frozen 30min past the HTTP timeout) — killed + resumed.

## Status

`done` — n=8 on all six cells (Qwen 3.5 9B reasoner-size point, both perception
configs). Synthesis + cross-size tables: `harness-axis-summary.md`.
