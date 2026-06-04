# Harness-type VLM-axis sweep — ReAct & CodeAct vs RLM (val, n=8)

## Question

The RLM (`rvlm`) VLM-axis sweep established the **perception-budget**
result (swap only the VLM →27B, reasoner fixed, +~8pp at 9B & 4B). This
sweep repeats the design with two other agent **harness types**, because
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
| Qwen3.5 4B | 21.09% ± 3.16 | **15.66% ± 4.73** (n=8) | **15.66% ± 3.00** (n=8) |

## Results — v1 homog (LLM = VLM) + 27B anchor

| Config | RLM | ReAct | CodeAct |
|---|---|---|---|
| Qwen3.5 9B homog | 16.67% ± 3.40 | **14.97% ± 2.96** (n=8) | **19.35% ± 4.24** (n=8) |
| Qwen3.5 4B homog | 12.49% ± 3.74 | **11.94% ± 2.23** (n=8) | **12.19% ± 3.50** (n=8) |
| Qwen3.5 27B homog | **39.38 ± 1.49** (`rvlm`, n=8) | **25.16 ± 4.60** (n=8) | **37.25 ± 4.63** (b=40, n=5/8) |

## Locked cells

### 8B v2 ReAct (R1) — 2026-06-02
- run_ids `react-8b-llm-27b-vlm-val-t{1..8}`.
- per-trial: 13.00 / 17.67 / 16.74 / 15.54 / 13.28 / 15.54 / 18.98 / 15.58.
- **mean 15.79% ± 2.03, n=8.**
- **vs 8B v2 RLM (11.73% ± 2.96): +4.06pp**, and lower variance.
  For the weak 8B text-only reasoner, the append-only ReAct harness beats
  RLM's hidden-state harness — the first evidence that the harness type (not
  just perception budget) matters, and in the FT-relevant direction.

### 8B v2 CodeAct (CA-8Bv2) — 2026-06-02
- run_ids `codeact-8b-llm-27b-vlm-val-t{1..8}`.
- per-trial: 8.68 / 11.87 / 9.30 / 10.27 / 8.31 / 7.87 / 11.14 / 8.54.
- **mean 9.50% ± 1.44, n=8.**
- **8B v2 harness ranking (n=8 each): ReAct 15.79 > RLM 11.73 > CodeAct 9.50.**
  For the weak 8B text-only reasoner the harness type dominates, and the
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
  harnesses) → 9B v2 = 24.26, **statistically tied with 9B RLM (24.54 ±
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
- **Harness rank-flip with scale (confirmed at n=8, 27B VLM):**
  - 8B v2: **ReAct 15.79 > RLM 11.73 > CodeAct 9.50** (simplest harness wins for the weak reasoner; code-state harnesses drown it).
  - 9B v2: **RLM 24.54 ≈ CodeAct 24.26 > ReAct 21.01** (RLM/CodeAct overtake ReAct once the reasoner can exploit code+state).
  - 27B (n=8 RLM/ReAct; CodeAct n=5/8 preliminary): **RLM 39.38 ≳ CodeAct
    ~37 ≫ ReAct 25.16** — RLM and CodeAct close at the top, ReAct far back.
  Narrative: ReAct is the floor-raiser for weak reasoners; at the top,
  CodeAct draws level with RLM (provisional, locks at n=8) rather than
  pulling clear. Supports an append-only-code FT target *given* a capable
  base model — at ~no cost vs RLM if the 27B CodeAct number holds.

### 4B v2 ReAct (R3) — 2026-06-03
- run_ids `react-4b-llm-27b-vlm-val-t{1..8}`.
- per-trial: 11.38 / 17.97 / 13.60 / 26.07 / 16.55 / 12.64 / 12.70 / 14.37.
- **mean 15.66% ± 4.73, n=8** (identical mean to CodeAct 4B; both below RLM 4B 21.09).

### v2 column complete — harness × reasoner-size summary (27B VLM, n=8)
| Reasoner | RLM | ReAct | CodeAct | best |
|---|---|---|---|---|
| 8B (text-only) | 11.73 | **15.79** | 9.50 | ReAct |
| 4B | **21.09** | 15.66 | 15.66 | RLM |
| 9B | **24.54** | 21.01 | 24.26 | RLM≈CodeAct |
| 27B/27B | **39.38** (`rvlm`, n=8) | 25.16 (n=8) | 37.25 (b=40, **n=5/8 — preliminary**) | RLM ≳ CodeAct* ≫ ReAct |

\* CodeAct-27B is **not yet locked** — see the preliminary note below.

> **⚠ CodeAct-27B is PRELIMINARY (n<8).** It is being run on amax1 as a
> 3-budget sweep (`max_iterations` ∈ {24, 40, 56}) to also check the
> budget axis. Current partial means: **b24 37.92 ± 5.90 (n=6), b40
> 37.25 ± 4.63 (n=5), b56 36.00 ± 2.40 (n=5)**; pooled 37.11 ± 4.42
> (n=16). Budget looks flat (~36–38%, all within ~2pp). The harness-cell
> number above uses **b=40** (the default budget, comparable to the other
> CodeAct cells) and **will be refreshed when b40 hits n=8.** Treat the
> 27B CodeAct reading below as provisional.

**Reading (with the 27B row filled from the main matrix; CodeAct-27B provisional):**
- **CodeAct scales hardest, and at 27B it looks set to land *near* RLM —
  not clear of it.** Worst at 8B (9.50), catches RLM at 9B (24.26 ≈
  24.54), and at 27B is **tracking ~37 (n=5/8), ~2pp under RLM's 39.38**.
  (An earlier `~35.6 (n=4)` estimate suggested CodeAct "pulled clear" —
  with the RLM-27B number now in hand, RLM ≳ CodeAct at the top, both
  close. **This locks at n=8.**) If the n=8 number holds, the FT-relevant
  read is: the clean append-only-code MDP target costs ~no accuracy vs
  hidden-state RLM *given a strong reasoner* (≥9B).
- **RLM is robust across the range** — best at 4B, 9B, and 27B (39.38);
  CodeAct converges toward it as the reasoner scales but (on current
  partial data) does not overtake.
- **ReAct is the floor** — only wins at the weakest reasoner (8B); at 27B
  it is far back (25.16, ~12-14pp below RLM) — no REPL to compose
  multi-step perception.
- FT implication (provisional, pending CodeAct-27B n=8): an
  append-only-code harness is the right target **iff** the base reasoner
  is strong enough (≥9B here); at 27B it appears to cost only ~2pp vs RLM
  for a much cleaner MDP trajectory; below 9B, RLM or plain ReAct is safer.

### 4B v1 CodeAct (CA-4Bv1) — 2026-06-04
- run_ids `codeact-3_5-4b-val-t{1..8}`.
- per-trial: 10.60 / 9.51 / 16.48 / 6.10 / 14.21 / 16.08 / 11.50 / 13.04.
- **mean 12.19% ± 3.50, n=8.**
- **CodeAct ties RLM at the 4B homog floor:** CodeAct 12.19 ≈ RLM 4B
  homog 12.49 ± 3.74 (statistically indistinguishable). With both LLM
  and VLM at 4B (weak perception + weak reasoner), the harness type
  stops mattering — every harness collapses to the same low floor.
- **Perception-budget lift under CodeAct (4B):** homog 12.19 → v2 (27B
  VLM) 15.66 = **+3.5pp**. The swap-only-the-VLM lift replicates under
  CodeAct but is smaller than RLM's (4B RLM homog 12.49 → v2 21.09 =
  +8.6pp). The weak 4B reasoner can't exploit a stronger VLM through the
  append-only-code harness as well as through RLM's hidden-state REPL —
  consistent with CodeAct needing a capable reasoner.

### 4B v1 ReAct (R5) — 2026-06-04
- run_ids `react-3_5-4b-val-t{1..8}`.
- per-trial: 8.97 / 12.50 / 11.91 / 9.60 / 10.04 / 14.50 / 13.10 / 14.90.
- **mean 11.94% ± 2.23, n=8.**
- **The 4B/4B floor — all three harnesses tie:** RLM 12.49 ± 3.74,
  ReAct 11.94 ± 2.23, CodeAct 12.19 ± 3.50 — statistically
  indistinguishable (~12%, every pairwise gap ≪ the stds). With both
  reasoner and perception at 4B, the harness type carries no signal: a
  weak reasoner served by a weak VLM collapses to the same floor
  regardless of how its action/observation context is structured. The
  harness-type effects (the 8B rank-flip, the CodeAct scaling slope)
  only emerge once *either* perception (v2's 27B VLM) or the reasoner
  (9B/27B) is strong enough to exploit the harness.

## v3 — reasoning vs perception (LLM=27B / VLM=9B), in progress

The missing factorial corner: a **strong reasoner on weak perception**,
to pair against v2's weak-reasoner-on-strong-perception (9B-LM/27B-VLM).
run_ids `{rvlm-minimal,react,codeact}-27b-llm-9b-vlm-val-t*`, n=8 target,
cap 3 on 27B DP=3 @8927 + 9B @8909.

**Current (ReAct locked n=8; RLM/CodeAct partial — rest on amax1):**

| Harness | 9B-LM / 27B-VLM (v2) | 27B-LM / 9B-VLM (v3) | Δ | lean |
|---|---|---|---|---|
| RLM | 24.54 | 33.67 (n=2: 32.60, 34.74) | **+9.1** | reasoning-bound |
| CodeAct | 24.26 | 30.14 (n=4: 27.87, 35.24, 32.81, 24.64) | **+5.9** | reasoning-bound |
| ReAct | 21.01 | **17.96 ± 3.94 (n=8)** | **−3.05** | **perception-bound** |

ReAct v3 per-trial: 14.83 / 17.91 / 21.44 / 19.27 / 14.23 / 15.71 / 25.48
/ 14.84. **ReAct v3 is the only harness that *loses* going from
9B-LM/27B-VLM → 27B-LM/9B-VLM** (−3.05pp, n=8) — a stronger reasoner on
weaker perception can't recover the lost VLM acuity because it has no
crop/zoom actuator. RLM (+9.1) and CodeAct (+5.9) both *gain* from the
reasoner swap (partial n; locks on amax1). RLM/CodeAct v3 remaining
trials run on amax1 (see `coordination/amax7.md` 2026-06-04 note).

**Mechanism — the REPL is what converts reasoning into perception
(why ReAct is the lone perception-bound harness).** ReAct's only
perception actuators are `look(page, query)` / `look_many(pages, query)`
— **whole-page** VLM queries at fixed page granularity. It cannot crop a
region, zoom, composite, or do coordinate arithmetic on an image (no
Python execution; `react_baseline_solver.py` docstring: *"no crops … no
PIL.crop on retrieved page images"*). So ReAct's perception ceiling **is**
the VLM's whole-page acuity: a stronger reasoner has no actuator to
direct finer-grained perception, so swapping 9B→27B reasoner while the
VLM stays at 9B buys nothing (it even regresses) — ReAct is
**perception-bound**, and to improve it you must improve the raw VLM.

RLM and CodeAct write Python around the same `batch_look`: they crop to
the evidence region, zoom, retry tighter, composite panels, and arithmetic
on coordinates. A stronger reasoner therefore produces **better-targeted
sub-images** and extracts more *even from a weaker (9B) VLM* — so the
27B-reasoner/9B-VLM corner *gains* (RLM +8, CodeAct +3.6) → these harnesses
are **reasoning-bound**. This is a direct corollary of the D-006
active-perception thesis: the REPL crop/zoom loop is the mechanism that
**turns reasoning capability into perception quality**; remove it (ReAct)
and reasoning can no longer buy perception. Locks at n=8.

## Status

In progress. Order (per-user 2026-06-02): R1 ✅ → CodeAct sweep
(8B v2 → 9B v2 → 4B v2 → 27B/27B → 4B v1 → 9B v1) → deferred ReAct
R2–R5. Driver state: `tmp/workspace/qwen-9b-vlm-axis/driver-state.md`.

**27B/27B CodeAct (CA-27B) — in progress on amax1 (2026-06-04).** Run as
a 3-budget sweep (`max_iterations` ∈ {24, 40, 56}), n=8 each, at high
concurrency. Partial: b24 n=6, b40 n=5, b56 n=5 — all ~36–38% (budget
flat). Per-trial data accumulates in `codeact-qwen-3_5-27b.md`; the
27B-row numbers here use **b=40** (default budget) and **refresh to n=8
when the b40 cell completes** (autonomous cron `295678ee`). RLM-27B
(39.38, `rvlm`) and ReAct-27B (25.16) are final from the main 8-solver
matrix (`docs/results.md`).
