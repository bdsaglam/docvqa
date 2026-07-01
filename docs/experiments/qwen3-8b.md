# Qwen3 8B (reasoner, older generation) — v2 only (val, n=8)

A reasoner-**quality** point, off the Qwen3.5 size curve. Qwen3-8B is the
older **Qwen3** family and is **text-only**, so it can only serve as the
orchestrator with the VLM swapped in — **v2 mixed only** (LLM = Qwen3-8B,
VLM = Qwen 3.5 27B). No v1 homog (it can't be the VLM).

n=8 per cell, val 25 docs / 80 Q, `lm.enable_thinking=false`, per-question
micro-average. Cross-size narrative: [`harness-axis-summary.md`](harness-axis-summary.md).

## Results — v2 mixed (Qwen3-8B reasoner + 27B VLM), n=8

| Harness | Val (n=8) | run_id stem |
|---|---|---|
| ReAct | **15.79% ± 2.03** | `react-8b-llm-27b-vlm-val-t{1..8}` |
| RLM (`rvlm`) | 11.73% ± 2.96 | `rvlm-minimal-8b-llm-27b-vlm-val-t{1..8}` |
| CodeActᶜ | 9.50% ± 1.44 | `codeact-8b-llm-27b-vlm-val-t{1..8}` |

ᶜ **STALE — do not cite.** Old dspy `codeact` (deprecated). The corrected
**`codeact_chat`** twin is the sole source of truth for CodeAct numbers going
forward; a config without a `codeact_chat` value is **open** — the stale dspy
figure is shown for provenance only, not as a current result. Tracking and
replacements: `codeact-chat-qwen-3_5-27b.md`.

**8B v2 harness ranking: ReAct 15.79 > RLM 11.73 > CodeAct 9.50.**

## Per-trial

| Harness | per-trial (8) |
|---|---|
| ReAct | 13.00 / 17.67 / 16.74 / 15.54 / 13.28 / 15.54 / 18.98 / 15.58 |
| RLM | 18.18 / 12.84 / 10.90 / 9.71 / 10.18 / 10.34 / 12.88 / 8.84 |
| CodeAct | 8.68 / 11.87 / 9.30 / 10.27 / 8.31 / 7.87 / 11.14 / 8.54 |

## Reads

- **An older-generation reasoner, NOT a modality confound.** In v2 the reasoner
  delegates *all* perception to the 27B VLM via `batch_look` and never sees
  pixels in its own context (`rvlm_solver.py` loads `pages` only inside the
  sandbox). So text-only vs multimodal is **irrelevant in v2** — every reasoner
  is a text orchestrator on the same VLM. The only variable vs the Qwen3.5
  9B/4B points is **generation** (Qwen3 vs Qwen3.5). The older 8B is simply a
  **weaker orchestrator**: on RLM it thrashed ~18 iterations/question and
  force-submitted wrong, scoring 11.73% — *below* even the newer 4B (RLM v2
  21.09%) and half of 9B v2 (25.31%). Bug ruled out: `enable_thinking=false`
  correctly applied (`types.py:66` → `chat_template_kwargs`), tool/parse errors
  negligible, `batch_look` returns real content. A clean reasoner-quality
  signal — kept **off** the Qwen3.5 9B↔4B size curve. A clean 8B size point
  would need **Qwen3.5-8B** (same family).
- **The harness ranking is the opposite of "more expressive = better."** For
  the weak 8B reasoner, ReAct's short observation trace is easiest; CodeAct's
  append-only **growing-code** context is hardest (lowest mean, lowest variance
  — it fails consistently, not erratically). This is the first evidence that
  the harness type (not just perception budget) matters — and in the
  FT-relevant direction (the append-only-code MDP only pays off with a stronger
  reasoner; see the CodeAct crossover at 9B in `harness-axis-summary.md`).
  CodeAct on 8B needed 5 finalize-gap resumes (heaviest docs error under
  CodeAct+8B), more than RLM/ReAct.

## Setup

- LLM: Qwen3-8B vllm (text-only), `enable_thinking=false`. VLM: Qwen 3.5 27B.
- Solver: `rvlm` / `react_baseline` / `codeact`. Data: val, 25 docs / 80 Q,
  `max_concurrency=16`.

## Status

`done` — n=8 on all three harnesses (v2 only). Reasoner-quality point.
