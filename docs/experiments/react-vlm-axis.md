# Loop-type VLM-axis sweep — ReAct & CodeAct vs RLM (val, n=8)

## Question

The RLM (`rvlm`) VLM-axis sweep established the **perception-budget**
result (swap only the VLM →27B, reasoner fixed, +~8pp at 9B & 4B). This
sweep repeats the design with two other agent **loop types**, because
the fine-tuning target is **not** an RLM:

- **ReAct** (`react_baseline`): thought→tool→observation, no code.
- **CodeAct** (`codeact`): action = Python code in a REPL, **append-only
  context** — same tools/prompt/`batch_look` as `rvlm`, the only change
  vs RLM is the context discipline (no `variables_info` sidecar, no
  `RESET_HISTORY`). RLM's hidden Python-namespace state makes it a POMDP;
  an append-only trace is a clean MDP to RL-fine-tune.

Same cells as the RLM sweep: per reasoner size, v1 homog (LLM=VLM) and
v2 mixed (LLM=small, VLM=27B). Qwen3-8B is text-only → v2 only.
`lm.enable_thinking=false`, val 25 docs / 80 Qs, per-question
micro-average.

## Results — v2 mixed (reasoner + 27B VLM)

| Reasoner | RLM | ReAct | CodeAct |
|---|---|---|---|
| Qwen3 8B (text-only) | 11.73% ± 2.96 | **15.79% ± 2.03** (n=8) | _running (CA-8Bv2)_ |
| Qwen3.5 9B | 24.54% ± 5.30 | _queued (R2)_ | _queued (CA-9Bv2)_ |
| Qwen3.5 4B | 21.09% ± 3.16 | _queued (R3)_ | _queued (CA-4Bv2)_ |

## Results — v1 homog (LLM = VLM) + 27B anchor

| Config | RLM | ReAct | CodeAct |
|---|---|---|---|
| Qwen3.5 9B homog | 16.67% ± 3.40 | _queued (R4)_ | _queued (CA-9Bv1)_ |
| Qwen3.5 4B homog | 12.49% ± 3.74 | _queued (R5)_ | _queued (CA-4Bv1)_ |
| Qwen3.5 27B homog | — | ~23.8% (n=4, see `react_baseline-qwen-3_5-27b.md`) | _queued (CA-27B)_ |

## Locked cells

### 8B v2 ReAct (R1) — 2026-06-02
- run_ids `react-8b-llm-27b-vlm-val-t{1..8}`.
- per-trial: 13.00 / 17.67 / 16.74 / 15.54 / 13.28 / 15.54 / 18.98 / 15.58.
- **mean 15.79% ± 2.03, n=8.**
- **vs 8B v2 RLM (11.73% ± 2.96): +4.06pp**, and lower variance.
  For the weak 8B text-only reasoner, the append-only ReAct loop beats
  RLM's hidden-state loop — the first evidence that the loop type (not
  just perception budget) matters, and in the FT-relevant direction.

## Status

In progress. Order (per-user 2026-06-02): R1 ✅ → CodeAct sweep
(8B v2 → 9B v2 → 4B v2 → 27B/27B → 4B v1 → 9B v1) → deferred ReAct
R2–R5. Driver state: `tmp/workspace/qwen-9b-vlm-axis/driver-state.md`.
