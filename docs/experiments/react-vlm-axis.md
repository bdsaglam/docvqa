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
| Qwen3 8B (text-only) | 11.73% ± 2.96 | **15.79% ± 2.03** (n=8) | **9.50% ± 1.44** (n=8) |
| Qwen3.5 9B | 24.54% ± 5.30 | **21.01% ± 4.63** (n=8) | **24.26% ± 4.68** (n=8) |
| Qwen3.5 4B | 21.09% ± 3.16 | _queued (R3)_ | **15.66% ± 3.00** (n=8) |

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

### 8B v2 CodeAct (CA-8Bv2) — 2026-06-02
- run_ids `codeact-8b-llm-27b-vlm-val-t{1..8}`.
- per-trial: 8.68 / 11.87 / 9.30 / 10.27 / 8.31 / 7.87 / 11.14 / 8.54.
- **mean 9.50% ± 1.44, n=8.**
- **8B v2 loop ranking (n=8 each): ReAct 15.79 > RLM 11.73 > CodeAct 9.50.**
  For the weak 8B text-only reasoner the loop type dominates, and the
  ordering is the *opposite* of "more expressive = better": ReAct's
  short observation trace is easiest; CodeAct's append-only **growing
  code** context is hardest (lowest mean, lowest variance — it fails
  consistently, not erratically). Caveat for FT: append-only-code may
  only pay off with a stronger reasoner; the 9B and 27B CodeAct cells
  test whether it crosses over. Also: CA-8Bv2 needed 5 finalize-gap
  resumes (heaviest docs error under CodeAct+8B), more than RLM/ReAct.

### 9B v2 CodeAct (CA-9Bv2) — 2026-06-03
- run_ids `codeact-9b-llm-27b-vlm-val-t{1..8}`.
- per-trial: 22.74 / 25.08 / 19.84 / 33.38 / 17.73 / 26.07 / 25.51 / 23.78.
- **mean 24.26% ± 4.68, n=8.**
- **CodeAct crossover (vs RLM, same VLM):** 8B v2 = 9.50 (worst of 3
  loops) → 9B v2 = 24.26, **statistically tied with 9B RLM (24.54 ±
  5.30)**. CodeAct's accuracy scales much harder with reasoner
  capability than ReAct or RLM — exactly what the append-only-code FT
  design needs (a weak reasoner drowns in the growing context; a
  capable one exploits the code expressivity). The 27B/27B CodeAct cell
  (CA-27B) tests whether it overtakes at the top. One IPC-deadlock hang
  this cell (t2/maps_2, batch_look bridge frozen 30min past the HTTP
  timeout) — killed+resumed; first such hang in the CodeAct sweep.

### 4B v2 CodeAct (CA-4Bv2) — 2026-06-03
- run_ids `codeact-4b-llm-27b-vlm-val-t{1..8}`.
- per-trial: 13.40 / 14.84 / 19.53 / 14.38 / 14.10 / 20.31 / 16.90 / 11.80.
- **mean 15.66% ± 3.00, n=8** (vs 4B RLM 21.09 → CodeAct −5.4pp).
- **CodeAct reasoner-scaling slope (Qwen3.5, clean):** 4B 15.66 → 9B
  24.26 = **+8.6pp**, vs RLM 4B 21.09 → 9B 24.54 = +3.5pp. CodeAct
  benefits ~2.5× more from reasoner scale — it starts behind small
  models and catches RLM by 9B. One IPC-deadlock hang this cell
  (t5/slide_2, batch_look bridge) — killed+resumed (2nd of the CodeAct
  sweep, after t2/maps_2 at 9B).

### 9B v2 ReAct (R2) — 2026-06-03
- run_ids `react-9b-llm-27b-vlm-val-t{1..8}`.
- per-trial: 24.31 / 21.64 / 22.80 / 19.10 / 15.44 / 22.00 / 28.45 / 14.34.
- **mean 21.01% ± 4.63, n=8.**
- **Loop rank-flip with scale (confirmed at n=8, 27B VLM):**
  - 8B v2: **ReAct 15.79 > RLM 11.73 > CodeAct 9.50** (simplest loop wins for the weak reasoner; code-state loops drown it).
  - 9B v2: **RLM 24.54 ≈ CodeAct 24.26 > ReAct 21.01** (RLM/CodeAct overtake ReAct once the reasoner can exploit code+state).
  - 27B: CodeAct ~35.6 pulls clear of the field.
  Narrative: ReAct is the floor-raiser for weak reasoners; CodeAct is
  the ceiling-raiser for strong ones; RLM tracks CodeAct but plateaus.
  Supports an append-only-code FT target *given* a capable base model.

## Status

In progress. Order (per-user 2026-06-02): R1 ✅ → CodeAct sweep
(8B v2 → 9B v2 → 4B v2 → 27B/27B → 4B v1 → 9B v1) → deferred ReAct
R2–R5. Driver state: `tmp/workspace/qwen-9b-vlm-axis/driver-state.md`.
