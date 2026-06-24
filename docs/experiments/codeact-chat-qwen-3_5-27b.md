# codeact_chat — Qwen 3.5 model-axis (val/80-Q)

## (a) What `codeact_chat` is — the MDP / RL-target framing

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
`codeact` / vs `rvlm`? (2) does `enable_thinking=true` help? (3) how does
it behave across the model axis (homog 4B/9B/27B + cross-model ladders)?

### Setup

- Solver: `codeact_chat` (`src/docvqa/solvers/codeact_chat_solver.py`;
  config `solver=codeact_chat`). Reuses `rvlm`'s prompt body, `batch_look`,
  `_build_sandbox_code`, `SubprocessInterpreter`, `SUBMIT()`.
- Profile: DocVQA-2026 (default). `max_iterations=40` (+ page bonus).
- val 25 docs / 80 questions, `data.num_samples=null`,
  `enable_thinking=false` (headline). n=8, overlap-the-tail heartbeat.
- Artifacts: each doc writes `trajectory.json` (raw role-tagged transcript;
  runner change, scoped to chat-format trajectories so legacy base64
  trajectories don't bloat disk) — the intended RL fine-tuning target.

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false lm.timeout=1800 solver=codeact_chat \
  data.split=val data.num_samples=null max_concurrency=24 \
  run_id=codeact-chat-val-tN
# cross-model: set lm=/vlm= to the desired pair (e.g. lm=qwen-3_5-4b-vllm-local
#   vlm=qwen-3_5-27b-vllm-local), run_id=codeact-chat-<tag>-val-tN
```

---

## (b) 27B-homog headline (27B-LM / 27B-VLM, val/80-Q)

### Headline — `enable_thinking=false`, n=8

Per-trial: 40.0 / 43.8 / 42.5 / 38.8 / 37.5 / 36.2 / 36.2 / 41.2

**39.53% ± 2.83** (n=8) · **pass@8 63.75% · SC@8 45.00%** (diagnostic, not
headline per D-003 — oracle ceiling +24.2pp over avg@1 → large recoverable
headroom for a verifier/RL reward; SC@8 +5.5pp. Full axis in
[`../pass-at-k.md`](../pass-at-k.md): 4b/27b pass@8 55.0/SC 26.2; 4b-homog 47.5/20.0.)

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
`batch_look` hang that survived 3 resume attempts). **pass@7 58.75 · SC@7
38.75.** **Δ vs no-think = −1.85pp.**

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

---

## (c) Cross-model / model-axis results (no-think, val/80-Q)

Two ladders intersect at 27B/27B: a **perception-fixed** ladder (fix the
reasoner, scale the VLM) and a **reasoner-fixed** ladder (fix the VLM,
scale the reasoner). Cells run smaller-models-first, DP=3-per-model on
amax1 (swap model → run to n=8 → swap). Queue:
`tmp/workspace/codeact-chat-remaining/QUEUE.md`.

### Homogeneous (same model both roles)

| cell | model | n | `codeact_chat` | old `codeact` ref | status |
|---|---|---|---|---|---|
| 4b-homog | 4B / 4B | 8 | **16.25% ± 2.00** | 12.19 | **DONE** |
| 9b-homog | 9B / 9B | 8 | **22.97% ± 2.75** | 19.35 | **DONE** |
| 27B-homog | 27B / 27B | 8 | **39.53% ± 2.83** | 36.74 | **DONE** |
| gemma-E4B | gemma-4-E4B / E4B | 8 | **7.81% ± 1.86** | 7.66 | **DONE** |
| gemma-31B | gemma-4-31B / 31B | 4 | **30.31% ± 2.13** | 29.25 (n=5) | **DONE** |

> **Stop-token fix (2026-06-24, commit in `codeact_chat_solver.py`).** gemma-4
> uses a harmony/channel format and never emits a turn-final EOS in the
> hand-rolled `litellm` chat loop, so it generated the **entire multi-turn
> rollout in one completion** — fabricating observations + many code blocks +
> a premature `SUBMIT` — which ran concatenated → garbage. A first pass scored
> gemma-31B at a bogus **5.00%** (vs `rvlm` 33). Diagnosis was decisive: direct
> image probe shows perception works; the trajectory showed gemma role-playing
> both sides of the loop. Fix = `_split_first_turn` truncates each assistant
> turn to its first code block (enforcing the one-action-per-turn protocol);
> no-op for self-stopping Qwen. After the fix gemma-31B → **30.31%**, in line
> with `rvlm`/old `codeact`. The gemma-E4B 6.56% below is from the **pre-fix**
> loop and is being re-run.

- **gemma-31B** (n=4, post-fix): **30.31% ± 2.13**, **pass@4 47.50 · SC@4
  28.75**, Unknown 8% (gemma4 image, TP=2 GPU0+1 @ `:8931`, shm32g/ipc=host,
  `c=2` for TP stability). Matches `rvlm` 33.04 and old `codeact` 29.25 — gemma-4
  is a competent codeact_chat reasoner once it actually takes turns. Confirms
  `codeact_chat` ties `rvlm` for **both** model families at the strong-reasoner
  end (Qwen-27B 39.5≈39.4; gemma-31B 30.3≈33.0).

- **gemma-E4B** (post-fix, n=8): **7.81% ± 1.86**, **pass@8 20.00 · SC@8 8.75**,
  Unknown 40% (gemma4 E4B image, DP=3 @ `:8904`). The stop-token fix barely
  moves E4B (pre-fix 6.56 → 7.81), because a 4B-class model scores at the floor
  whether it takes proper turns or runs away — it is simply too weak to exploit
  the scaffold. So the **"no harness lift at 4B" negative control holds**: E4B
  (7.81) ≈ old `codeact` (7.66) ≈ `official_baseline` floor (6.25). Clean
  contrast with gemma-31B, where the *same* bug suppressed a genuinely capable
  model from 30.31 → 5.00 — the bug's impact scales with how much real
  multi-turn reasoning the model would otherwise contribute.

- **4b-homog** per-trial: 15.0 / 17.5 / 18.8 / 15.0 / 12.5 / 16.2 / 17.5 /
  17.5 (n=8). Was n=6 (15.83 ± 2.20) when paused; t7/t8 ran on the
  resumed DP=3 4B server → **16.25% ± 2.00 (n=8)**, **pass@8 47.50 · SC@8
  20.00**, +4.1pp vs old `codeact`. This is the matrix floor among the
  codeact_chat cells.
- **9b-homog** per-trial: 20.0 / 27.5 / 18.8 / 25.0 / 22.5 / 23.8 / 22.5 /
  23.8 (n=8) → **22.97% ± 2.75**, **pass@8 61.25 · SC@8 32.50** (2026-06-16,
  DP=3 9B). +1.4pp vs old `codeact` (19.35) and **+4.1pp vs `rvlm` 9b-homog
  (18.91)** — at 9B homog the corrected MDP loop leads the proposed method.
  `engineering_drawing_1` + `science_paper_1` were stochastic blockers (9B
  degenerate `batch_look` scans → repeated 10-min exec-timeouts) that needed
  several resume cycles per trial; the exec-timeout guard kept them bounded.

### Perception-fixed ladder (fixed 4B reasoner, scale the VLM)

| cell | lm / vlm | n | `codeact_chat` | old `codeact` ref | status |
|---|---|---|---|---|---|
| 4b-homog | 4B / 4B | 8 | **16.25% ± 2.00** | 12.19 | **DONE** |
| 4b/27b | 4B-LM / 27B-VLM | 8 | **22.34% ± 3.44** | 15.66 | **DONE** |
| 9b/27b | 9B-LM / 27B-VLM | 8 | **26.56% ± 4.21** | 24.26 | **DONE** |
| 8b/27b | qwen-3-8B-LM / 27B-VLM | 8 | **16.72% ± 3.20** | 9.50 | **DONE** |

- **4b/27b** per-trial: 25.0 / 18.8 / 22.5 / 26.2 / 23.8 / 25.0 / 21.2 /
  16.2 (n=8) → **22.34% ± 3.44**, **pass@8 55.00 · SC@8 26.25**. Swapping
  the 4B VLM → 27B VLM under a fixed 4B reasoner lifts **+6.1pp** over
  `4b-homog` (22.34 vs 16.25) — the perception-budget signature (supports
  D-006); `codeact_chat` clears old `codeact` by **+6.7pp**.
- **9b/27b** (n=8): **26.56% ± 4.21**, **pass@8 62.50 · SC@8 35.00** —
  **+2.3pp** over old `codeact` (24.26) and **+3.6pp** over `9b-homog`
  (22.97), continuing the perception-budget lift one rung up the reasoner
  ladder. Run on the cross-model topology: **9B reasoner local (131k) +
  27B VLM on the remote**. (The 9B needed its full 131k window — at 64k the
  append-only trajectory overflowed on image/question-heavy docs.)
- **8b/27b** (n=8): **16.72% ± 3.20**, **pass@8 37.50 · SC@8 17.50** —
  **+7.2pp** over old `codeact` (9.50). Off the main Qwen-3.5 axis: this is
  **Qwen3-8B** (older family), served at 131k via YaRN
  (`original_max_position_embeddings=32768, factor=4.0`). It underperforms the
  Qwen-3.5-**4B**/27B rung (22.34) despite being larger — the Qwen3-8B reasoner
  degenerates into page-by-page `batch_look` scans (one page per step, marching
  to the 50-step cap then defaulting to "Unknown"; 32% Unknown overall), so it
  floods the VLM and self-limits. A weak-reasoner data point, not a perception
  one; still clears old `codeact` decisively. Run fully local on amax1 (8B-LM @
  `:8908` GPU2, 27B-VLM **DP=2** @ `:8927` GPU0+1).

### Reasoner-fixed ladder (fixed VLM at the strong end, scale the reasoner)

| cell | lm / vlm | n | `codeact_chat` | old `codeact` ref | status |
|---|---|---|---|---|---|
| 27b/4b | 27B-LM / 4B-VLM | 8 | _queued (Phase 4, deprioritized)_ | — | QUEUED |
| v3 (27B/9B) | 27B-LM / 9B-VLM | 8 | **32.81% ± 3.04** | 30.43 | **DONE** |
| 27B-homog | 27B / 27B | 8 | **39.53% ± 2.83** | 36.74 | **DONE** |

- **v3** (n=8): **32.81% ± 3.04**, **pass@8 67.50 · SC@8 37.50** — **+2.4pp**
  over old `codeact` (30.43). Fills the reasoner-fixed ladder's middle rung
  (27B-LM holding, VLM 27B→9B): 39.53 (27B-VLM) → **32.81** (9B-VLM), a
  **−6.7pp** perception drop from halving-and-then-some the VLM, mirroring the
  perception-budget signature on the reasoner-fixed axis. Run entirely local on
  amax1 (27B-LM TP=2 @ `:8927`, 9B-VLM @ `:8909`, both 131k/64k windows).
- **27b/4b** (strongest reasoner × weakest VLM) is the ladder's bottom
  rung; it was **never run for any harness**. Phase 4 (deprioritized) runs
  it across **all 3 harnesses** (rvlm + codeact_chat + react) so the rung
  is comparable, not an orphan.

---

## (d) vs `rvlm` — the key comparison

These `rvlm` numbers are already-locked references (don't recompute).

| cell | `codeact_chat` | `rvlm` (proposed) | Δ (cc − rvlm) | read |
|---|---|---|---|---|
| 27B-homog | 39.53% ± 2.83 (n=8) | 39.38% ± 1.49 (n=8) | +0.15 | **tied** |
| 4b/27b | 22.34% ± 3.44 (n=8) | 21.09% ± 3.16 (n=8) | +1.25 | tied (Δ ≪ combined std) |
| 4b-homog | 16.25% ± 2.00 (n=8) | 12.49% ± 3.74 (n=8) | +3.76 | borderline (overlapping std) |

Old `codeact` *trailed* `rvlm` at every config; the corrected
`codeact_chat` **catches up to a statistical tie** — 27B +0.15, 4b/27b
+1.25 (both Δ ≪ combined std), and only a borderline nominal edge at
4b-homog (+3.76, both n=8 now, but overlapping stds → don't lean on it).
So the finding is **`codeact_chat` ≈ `rvlm` across the model axis** — the
append-only MDP matches the compacted-POMDP proposed method at no accuracy
cost. It does **not** beat `rvlm`; `rvlm` remains the proposed method, and
this *strengthens* the codeact-as-RL-target narrative.

Caveat: `rvlm` cross-model is clean n=8; `codeact_chat` 4b/27b is a
5-old/3-new-code mix (`f7f497e`, see below). The 4b-homog row is now a
clean n=8 (t7/t8 ran on the resumed server).

---

## (e) Exec-timeout fix — commit `f7f497e` (2026-06-12) + provenance

The 4B reasoner intermittently writes a **degenerate per-page
`batch_look` scan** (e.g. 120 sequential VLM calls on a 60–89pp doc) that
ran ~40min/cell and stalled trials; and heavy docs (`science_paper_1`,
`maps_2`, `engineering_drawing_1`) drop under VLM saturation when a call
exceeds the 120s per-message timeout.

**Fix:** a per-cell **wall-clock `exec_timeout=600s`**
(`SubprocessInterpreter`) that aborts the cell with a corrective message,
plus a **`_kill_and_reset`** so both timeout paths restart a clean
subprocess (re-runs `sandbox_code`, restores `pages`) instead of
contaminating the rest of the question. The fix only changes behavior on
>600s degenerate cells. It is in main.

**Provenance note (4b/27b n=8):** within that cell, **t1/t6/t7/t8 ran (or
were resumed) on the fixed code**; t2–t5 on the prior code. t1/t6
specifically *needed* it to complete validly (they'd hung / short-exited
otherwise) — so the mix gives valid completions, not a solver change. A
fully-homogeneous re-run is optional/pending. The 4b-homog n=8 and
27B-homog n=8 cells are not affected by this mix.

---

## (f) VLM-load diagnosis (2026-06-12, amax1 27B@:8927)

The doc-drops under multi-trial contention are **VLM saturation, not a
deadlock**: 27B mean e2e **70s/call**, 36s of it queue wait, TTFT 39s;
**~18% of calls exceed the 120s** per-message timeout (→ dropped heavy
docs). Responses are short (mean **78 output tok**, p75 ≤100, thinking
off) — the load is **prefill-bound** (33:1 prompt:gen ratio, large image
prompts). Lever = concurrency, not generation: serial trials / a DP=3 27B
across all GPUs removes the queue (see infra notes). The campaign now runs
one model on all 3 GPUs (DP=3) at a time to avoid cross-trial contention.

---

## (g) Operational caveats

- `science_paper_1` (19pp) and `business_report_1` (89pp) are the slow
  tail; `science_paper_1` + thinking deterministically hangs the
  `batch_look` IPC bridge (needed multiple kill+resume; t8 of the
  thinking sweep was abandoned).
- High variance (~3pp no-think, ~4.4pp think) — always n≥8; an early n=3
  read mis-signaled a `rvlm`-beating result that regressed to parity.
- **Completeness gate:** a trial counts only at **80/80 questions**; heavy
  docs can drop under load → resume the same `run_id`.

## Key findings

1. **Corrected MDP loop ≥ old `codeact` (+2.7pp at 27B), ties `rvlm`** —
   fixing the POMDP cost no accuracy and plausibly helped.
2. **`codeact_chat` ≈ `rvlm` across the model axis** (27B, 4b/27b,
   4b-homog all tied within combined std) — append-only MDP matches the
   compacted-POMDP proposed method at no accuracy cost.
3. **Perception-budget signature reproduces** (4b-homog 16.25 → 4b/27b
   22.34, +6.1pp from swapping only the VLM → 27B) — supports D-006.
4. **Thinking: no benefit** on DocVQA val; use no-think (also avoids the
   thinking-amplified `batch_look` hangs).
5. `trajectory.json` persists the clean role-tagged MDP transcript — the
   intended RL fine-tuning target.
