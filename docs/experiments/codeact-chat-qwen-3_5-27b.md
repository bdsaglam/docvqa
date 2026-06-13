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

## B/C cross-model campaign (no-think)

Topology: one model per GPU — 27B@:8927 (GPU0), 9B@:8909 (GPU2),
4B@:8904 (GPU1) — ran 27B-dependent and homog-small tracks concurrently
on disjoint GPUs.

| cell | config | `codeact_chat` | old `codeact` | Δ vs old | `rvlm` (proposed) | vs `rvlm` |
|---|---|---|---|---|---|---|
| 27B-homog | 27B / 27B | 39.53% ± 2.83 (n=8) | 36.74% | +2.8pp | 39.38% ± 1.49 | +0.15 — **tied** |
| **4b/27b** | 4B-LM / 27B-VLM | **22.34% ± 3.44 (n=8)** | 15.66% | **+6.7pp** | 21.09% ± 3.16 | +1.25 — **tied** |
| 4b-homog | 4B / 4B | 15.83% ± 2.20 (n=6) | 12.19% | +3.6pp | 12.49% ± 3.74 | +3.34 — borderline |
| v3 | 27B-LM / 9B-VLM | 32.9% (n=3) | 30.43% | +2.5pp | _(no clean rvlm v3 ref)_ | — |

- **4b/27b** per-trial: 25.0 / 18.8 / 22.5 / 26.2 / 23.8 / 25.0 / 21.2 / 16.2.
  Swapping the 4B VLM → 27B VLM under a fixed 4B reasoner lifts **+6.5pp**
  over `4b-homog` — the perception-budget signature (supports D-006), and
  `codeact_chat` clears old `codeact` by +6.7pp.
- **v3** per-trial: 28.7 / 37.5 / 32.5 (n=3; stopped at n=3 per pause).
- **4b-homog** per-trial: 15.0 / 17.5 / 18.8 / 15.0 / 12.5 / 16.2 (n=6;
  **t7/t8 deferred** — user held t7 mid-run, "resume later").

- **vs `rvlm` (the key comparison):** old `codeact` *trailed* `rvlm` at
  every config; the corrected `codeact_chat` **catches up to a statistical
  tie** — 27B +0.15, 4b/27b +1.25 (Δ ≪ combined std), and only a borderline
  nominal edge at 4b-homog (+3.34, Δ/SE≈2.1, p≈0.05, but `codeact_chat`
  n=6 vs `rvlm` n=8 and overlapping stds → don't lean on it). So the
  finding is **`codeact_chat` ≈ `rvlm` across the model axis** — the
  append-only MDP matches the compacted-POMDP proposed method at no
  accuracy cost. It does **not** beat `rvlm`; `rvlm` remains the proposed
  method, and this *strengthens* the codeact-as-RL-target narrative.
  Caveat: `rvlm` cross-model is clean n=8; `codeact_chat` 4b/27b is a
  5-old/3-new-code mix (`f7f497e`) and 4b-homog is n=6 — a homogeneous
  re-run would firm up the smaller-model rows.

**Code provenance / 10-min exec-timeout fix (2026-06-12, commit
`f7f497e`).** The 4B reasoner intermittently writes a **degenerate
per-page `batch_look` scan** (e.g. 120 sequential VLM calls on a 60–89pp
doc) that ran ~40min/cell and stalled trials; and heavy docs
(`science_paper_1`, `maps_2`, `engineering_drawing_1`) drop under VLM
saturation when a call exceeds the 120s per-message timeout. Fix: a
per-cell **wall-clock `exec_timeout=600s`** (`SubprocessInterpreter`) that
aborts the cell with a corrective message, plus a **`_kill_and_reset`** so
both timeout paths restart a clean subprocess (re-runs `sandbox_code`,
restores `pages`) instead of contaminating the rest of the question.
Within this n=8: **t1/t6/t7/t8 ran (or were resumed) on the fixed code**;
t2–t5 on the prior code. The fix only changes behavior on >600s degenerate
cells, and t1/t6 specifically *needed* it to complete validly (they'd hung
/ short-exited otherwise) — so the mix gives valid completions, not a
solver change. A fully-homogeneous re-run is optional/pending.

**Paused per user (2026-06-12)** after `4b/27b` n=8. **Deferred**
(resumable): 4b-homog t7/t8, v3 beyond n=3, 9b/27b, 9b-homog, qwen3-8B
(8b/27b), gemma E4B/31B homog. Old `codeact` refs for those: 9b-homog
19.35, 9b/27b 24.26, 8b/27b 9.50, gemma-E4B 7.66, gemma-31B 29.25 (n=5).

### VLM-load diagnosis (2026-06-12, amax1 27B@:8927)
The doc-drops under multi-trial contention are **VLM saturation, not a
deadlock**: 27B mean e2e **70s/call**, 36s of it queue wait, TTFT 39s;
**~18% of calls exceed the 120s** per-message timeout (→ dropped heavy
docs). Responses are short (mean **78 output tok**, p75 ≤100, thinking
off) — the load is **prefill-bound** (33:1 prompt:gen ratio, large image
prompts). Lever = concurrency, not generation: serial trials / a DP=3 27B
across all GPUs removes the queue (see infra notes).

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
