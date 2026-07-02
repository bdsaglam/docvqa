# Qwen 3.5 4B (reasoner) — cross-model perception sweep (val, n=8)

All cross-model cells for the **Qwen 3.5 4B reasoner**, across all three
harnesses (RLM `rvlm` / ReAct / CodeAct) in two perception configs:

- **v1 homog:** LLM = VLM = Qwen 3.5 4B.
- **v2 mixed:** LLM = 4B, VLM = Qwen 3.5 **27B** (swap *only* the VLM — the
  perception-budget test for D-006 prediction 1).

n=8 per cell, val 25 docs / 80 Q, `lm.enable_thinking=false`, per-question
micro-average (`scripts/report.py`). The 27B-VLM-vs-9B-VLM factorial corners
(v2 vs v3) and the cross-size narrative live in
[`harness-axis-summary.md`](harness-axis-summary.md); the 27B-reasoner row in
the per-solver files (`rvlm-qwen-3_5-27b.md`, etc.).

## Results (n=8)

| Harness | v1 homog (VLM=4B) | v2 mixed (VLM=27B) | Δ (v2−v1) |
|---|---|---|---|
| RLM (`rvlm`) | 14.22% ± 3.83 | **21.09% ± 3.16** | **+6.88pp** — Welch t=3.91, 95% CI [+3.1,+10.7], **sig.** |
| ReAct | 13.44% ± 1.98 | 18.12% ± 4.06 | +4.68pp |
| CodeActᶜ | 12.19% ± 3.50 | 15.66% ± 3.00 | +3.47pp |

ᶜ **STALE — do not cite.** Old dspy `codeact` (deprecated). The corrected
**`codeact_chat`** twin is the sole source of truth for CodeAct numbers going
forward; a config without a `codeact_chat` value is **open** — the stale dspy
figure is shown for provenance only, not as a current result. Tracking and
replacements: `codeact-chat-qwen-3_5-27b.md`.

## Per-trial

| Cell | run_id stem | per-trial (8) | mean ± std |
|---|---|---|---|
| RLM v1 | `rvlm-minimal-3_5-4b-val-t{1..8}` | _(recomputed from retained submissions; per-trial not shown)_ | 14.22 ± 3.83 |
| RLM v2 | `rvlm-minimal-4b-llm-27b-vlm-val-t{1..8}` | 25.00 / 20.00 / 21.25 / 23.75 / 23.75 / 20.00 / 15.00 / 20.00 | 21.09 ± 3.16 |
| ReAct v1 | `react-3_5-4b-val-t{1..8}` | _(recomputed from retained submissions; per-trial not shown)_ | 13.44 ± 1.98 |
| ReAct v2 | `react-4b-llm-27b-vlm-val-t{1..8}` | _(recomputed from retained submissions; per-trial not shown)_ | 18.12 ± 4.06 |
| CodeAct v1 | `codeact-3_5-4b-val-t{1..8}` | 10.60 / 9.51 / 16.48 / 6.10 / 14.21 / 16.08 / 11.50 / 13.04 | 12.19 ± 3.50 |
| CodeAct v2 | `codeact-4b-llm-27b-vlm-val-t{1..8}` | 13.40 / 14.84 / 19.53 / 14.38 / 14.10 / 20.31 / 16.90 / 11.80 | 15.66 ± 3.00 |

## Reads

- **The 4B/4B floor — all three harnesses tie (~13%).** RLM 14.22 ± 3.83,
  ReAct 13.44 ± 1.98, CodeAct 12.19 ± 3.50 are statistically
  indistinguishable (every pairwise gap ≪ the stds). With both reasoner and
  perception at 4B, harness type carries no signal — a weak reasoner served by
  a weak VLM collapses to the same floor regardless of how its
  action/observation context is structured. This is the clean **negative
  control** for the harness-lift hypothesis: lift needs a capable-enough base.
- **Perception-budget lift is real but reasoner-gated.** Swapping only the VLM
  →27B lifts RLM **+6.9pp** (significant), but ReAct/CodeAct only **+3.5–4.7pp**.
  The weak 4B reasoner can't exploit a stronger VLM through the append-only-code
  (CodeAct) or no-REPL (ReAct) harnesses as well as through RLM's hidden-state
  REPL — consistent with CodeAct needing a capable reasoner to pay off (see the
  CodeAct reasoner-scaling slope in `harness-axis-summary.md`).

## 4B as Perceiver — reasoner-fixed cell (27B-LM / 4B-VLM), `rvlm`, n=4

The complementary role: fix a **strong 27B reasoner** and use **4B as the VLM
perceiver** — the bottom rung of the reasoner-fixed perception ladder (full grid:
[`rvlm-reasoner-perceiver-3x3.md`](rvlm-reasoner-perceiver-3x3.md)).

**`rvlm` 27B-LM / 4B-VLM = 32.81% ± 3.13 (n=4)** — `rvlm-27b-llm-4b-vlm-val-t{1..4}`
(current `rvlm`, `enable_thinking=false`; this is also the Phase-4 bottom rung).

Ladder (fix 27B reasoner, scale the perceiver):

| Perceiver | `rvlm` | n |
|---|---|---|
| **4B-VLM**  | **32.81 ± 3.13** | 4 |
| 9B-VLM  | 37.2 ± 6.2 | 4 |
| 27B-VLM | 41.88 ± 5.79 | 8 |

**Read:** monotone ~4–5pp/step — even a strong 27B reasoner can't compensate for a
weak 4B perceiver, dropping −9pp vs 27B/27B. Perception is load-bearing under a
fixed reasoner (D-006). Contrast the *perception-budget lift* direction above (fix
the 4B reasoner, upgrade the VLM →27B = +6.88pp): the two directions are the same
mechanism seen from both ends.

## Setup

- Solver: `rvlm` (RLM), `react_baseline` (ReAct), `codeact` (CodeAct).
  `max_iterations=25` (RLM/ReAct); CodeAct default budget.
- LLM: Qwen 3.5 4B vllm, `enable_thinking=false`. VLM: 4B (v1) or 27B (v2).
- Data: val, all 25 docs / 80 Q. `max_concurrency` 6–16.
- Two IPC-deadlock hangs in the CodeAct sweep landed on 4B v2 (t5/`slide_2`,
  `batch_look` bridge frozen past the HTTP timeout) — killed + resumed.

## Status

`done` — n=8 on all six cells (Qwen 3.5 4B reasoner-size point, both perception
configs). Synthesis + cross-size tables: `harness-axis-summary.md`.
