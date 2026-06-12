# codeact_chat — Qwen 3.5 27B (val)

## Hypothesis / question

`codeact_chat` is the **corrected** `codeact`: a true multi-turn agentic
chat loop. State is a genuine `messages` array — `system` → `assistant`
(reasoning + fenced ```python```) → `user` (```output```) → … — where
each `assistant` turn **is** the policy action (the literal sampled
tokens), and each `user` turn is the real execution observation. The
loop runs over the same `batch_look` REPL/sandbox as `rvlm`/`codeact`,
but the reasoner is a **direct `litellm.completion`** — **no dspy in the
agent loop**.

Why it exists: the original `codeact` solver was append-only *in content*
but mechanically a single-turn `dspy.Predict` that re-rendered history
into a `trajectory` **string input field** each step — a *derived
observation* of the interaction, not the conversation itself
(POMDP-shaped, and useless as an RL rollout: no clean `(action,
observation)` turns). `codeact_chat` makes the conversation the state →
a clean, fully-observable **MDP**, RL-ready, persisted verbatim to
`trajectory.json`.

Questions: (1) does the corrected MDP loop cost accuracy vs the old
`codeact` / vs `rvlm`? (2) does `enable_thinking=true` help?

## Setup

- Solver: `codeact_chat` (`src/docvqa/solvers/codeact_chat_solver.py`;
  config `solver=codeact_chat`). Reuses `rvlm`'s prompt body, `batch_look`,
  `_build_sandbox_code`, `SubprocessInterpreter`, `SUBMIT()`.
- Model: Qwen 3.5 27B local vllm (lm + vlm), `enable_thinking=false` (headline).
- Profile: DocVQA-2026 (default). `max_iterations=40` (+ page bonus).
- n=8, overlap-the-tail heartbeat. Dates: 2026-06-11 / 06-12.
- Artifacts: each doc writes `trajectory.json` (raw role-tagged transcript;
  runner change, scoped to chat-format trajectories so legacy base64
  trajectories don't bloat disk).

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false lm.timeout=1800 solver=codeact_chat \
  data.split=val data.num_samples=null max_concurrency=24 \
  run_id=codeact-chat-val-tN
```

## Results — A-group (27B homog, val/80-Q)

### Headline — `enable_thinking=false`, n=8

Per-trial: 40.0 / 43.8 / 42.5 / 38.8 / 37.5 / 36.2 / 36.2 / 41.2

**39.53% ± 2.83** (n=8)

| comparison | value | Δ vs codeact_chat |
|---|---|---|
| **codeact_chat (corrected, MDP)** | **39.53% ± 2.83** | — |
| `rvlm` (proposed) | 39.38% ± 1.49 | **+0.15pp — tied** |
| old `codeact` (b40, matched budget, n=7) | 36.96% | +2.6pp |
| old `codeact` (pooled b24/40/56, n=23) | 36.74% ± 4.29 | +2.8pp |

**Reads:**
1. The corrected loop is a **real accuracy gain over the old `codeact`**
   (+2.6–2.8pp at matched budget). Borderline-significant (≈p=0.05 vs the
   high-variance old pooled set), but the direction is consistent and it
   sheds the dspy adapter fragility (no `AdapterParseError` wasted steps).
   So it is **not only POMDP→MDP hygiene** — fixing the mechanism also
   helped accuracy. Plausible mechanisms: native multi-turn chat (the
   model re-reads its own prior actions as real `assistant` turns) vs a
   re-rendered string field; fewer parse-failure dead steps.
2. It is **statistically tied with `rvlm`** (39.53 vs 39.38). So the
   true append-only **MDP matches `rvlm`'s compacted POMDP at no accuracy
   cost** — which *strengthens* the codeact-as-RL-target narrative without
   disturbing `rvlm` as the proposed method. (An early n=3 read of 42.1%
   suggested it might *exceed* `rvlm`; that regressed to parity by n=8 —
   it was top-of-noise, not a real edge.)

### Thinking ablation — `enable_thinking=true`, n=7

Per-trial: 43.8 / 41.2 / 38.8 / 37.5 / 35.0 / 37.5 / 30.0

**37.68% ± 4.42** (n=7; t8 excluded — pathological `science_paper_1`
`batch_look` hang that survived 3 resume attempts). **Δ vs no-think =
−1.85pp.**

**Verdict: thinking does NOT help** — marginally *worse* on the mean,
**higher variance** (±4.42 vs ±2.83), **~2× slower** (long think blocks =
far more generated tokens), and **pathologically hang-prone** on heavy
docs (`science_paper_1`, `maps_2`, `engineering_drawing_1`) via the
`batch_look` IPC bridge. → **Use `enable_thinking=false`** for all
downstream cells.

> **Server config for thinking runs:** run vllm **without**
> `--reasoning-parser qwen3` so `<think>…</think>` stays inline in
> `content` (byte-faithful for the RL target). The chat template
> pre-fills the *opening* `<think>` in the prompt, so `content` carries
> only the closing `</think>`; the solver restores the opener →
> well-formed `<think>…</think>` block in `trajectory.json`. (With the
> parser on, the solver instead re-wraps `reasoning_content`.) Both paths
> verified.

## B/C cross-model campaign — IN PROGRESS (no-think)

Topology: one model per GPU — 27B@:8927 (GPU0), 9B@:8909 (GPU2),
4B@:8904 (GPU1) — running 27B-dependent and homog-small tracks
concurrently on disjoint GPUs.

| cell | config | status | partial | old `codeact` ref |
|---|---|---|---|---|
| v3 | 27B-LM / 9B-VLM | n=2 + t3 in-flight | 28.7 / 37.5 | 30.43% |
| 4b-homog | 4B / 4B | n≈2–3 (winding to n=8) | 15.0 / 17.5 | 12.19% |
| 4b/27b | 4B-LM / 27B-VLM | **priority, running** | — | 15.66% |

**Paused per user (2026-06-12)** after `4b/27b` n=8 completes. **Deferred**
(resumable): v3 beyond n=3, 9b/27b, 9b-homog, qwen3-8B (8b/27b), gemma
E4B/31B homog. Old `codeact` refs for those: 9b-homog 19.35, 9b/27b 24.26,
8b/27b 9.50, gemma-E4B 7.66, gemma-31B 29.25 (n=5).

## Key findings

1. **Corrected MDP loop ≥ old `codeact` (+2.7pp), ties `rvlm`** — fixing
   the POMDP cost no accuracy and plausibly helped.
2. **Thinking: no benefit** on DocVQA val; use no-think (also avoids the
   thinking-amplified `batch_look` hangs).
3. `trajectory.json` persists the clean role-tagged MDP transcript — the
   intended RL fine-tuning target.

## Operational caveats

- `science_paper_1` (19pp) and `business_report_1` (89pp) are the slow
  tail; `science_paper_1` + thinking deterministically hangs the
  `batch_look` IPC bridge (needed multiple kill+resume; t8 of the
  thinking sweep was abandoned).
- High variance (~3pp no-think, ~4.4pp think) — always n≥8; an early n=3
  read mis-signaled a `rvlm`-beating result that regressed to parity.
