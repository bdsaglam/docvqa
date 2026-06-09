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
| Qwen3.5 27B homog | **39.38 ± 1.49** (`rvlm`, n=8) | **25.16 ± 4.60** (n=8) | **36.96 ± 5.25** (b=40, n=7) |
| Gemma-4 E4B homog | **7.34% ± 3.30** (n=8) | **6.09% ± 2.36** (n=8) | **7.66% ± 1.94** (n=8) |
| Gemma-4 31B homog | **32.50% ± 4.48** (n=8) | **18.44% ± 3.58** (n=8) | _n=8 in progress_ (n=1 pilot 37.50) |

**Cross-family Gemma points (n=8, escalated 2026-06-09 per user request;
supersedes the n=1/n=2 pilots):** homogeneous Gemma (lm=vlm=Gemma) ×
{rvlm, react, codeact}, n=8 each. Baselines (`raw_vlm_multi_baseline`,
`official_baseline`) and the 31B codeact cell are **still running** — the
per-model harness-LIFT table lands when they finish.

- **Gemma-4 E4B (n=8): rvlm 7.34 / react 6.09 / codeact 7.66 — all three
  harnesses statistically tied** (within ~1 std of each other, 6–8% band).
  A 4B model is **too weak to exploit any scaffold**: it burns its iteration
  budget on coding mistakes and has weak homogeneous vision, so harness type
  doesn't separate. (Per-cell stds 1.9–3.3pp; the gaps are noise.) This is
  itself the clean negative control for the lift hypothesis — lift needs a
  capable-enough base model.
- **Gemma-4 31B (n=8): rvlm 32.50 ≫ react 18.44 — a +14.1pp gap, ≫ the
  combined std.** This is the headline cross-family result: the recursive
  VLM sub-call (rvlm) is **load-bearing**, and the REPL-only ReAct harness
  collapses to the no-recursion tier — exactly mirroring Qwen 27B (rvlm 39.4
  ≫ react 25.2). The harness ordering (recursive-perception ≫ tool-only ReAct)
  is **robust across model families and is sharp once the base model is
  strong enough** (31B), while it vanishes into noise at 4B.
- Numbers moved modestly from the pilots (E4B rvlm 6.88→7.34, react 4.38→6.09;
  31B rvlm 30.00→32.50, react 20.00→18.44) — the n=1/n=2 pilots were inside
  the ~3–4pp trial noise, as expected.

**31B ReAct slow-doc note:** ReAct on Gemma-31B is expensive on image-heavy
docs (`engineering_drawing_1`, `science_poster_2`) — it can burn its full
`max_iters` (~34) without emitting an answer (a single doc can run ~30–60min).
Trials that stalled >10min on such a doc past 40min total runtime were
accepted as **timeout = failure** (that doc scored 0/N via an empty-prediction
placeholder, keeping a consistent /80 denominator). 3 of 8 react trials used
this; the other 5 completed all docs naturally and landed in the same range
(13.75–22.50), so the placeholder does not bias the cell. rvlm/codeact do not
hit this grind.

Config verified non-garbage (batch_look fires, model self-corrects from real
sandbox errors; real ReAct predictions e.g. map place-names read off the
page). **Serving:** canonical Gemma image is `vllm/vllm-openai:gemma4` +
`--reasoning-parser gemma4` (per `docs/scratchpad.md`); rvlm/codeact/react all
parse tool calls from text (no native tool parser needed). 31B runs TP=2
(`--enforce-eager --shm-size=16g`, one trial at a time — an inherent shm
crash recurs under bursty multi-image load and is handled by restart+resume).
Detail: `gemma-model-axis.md`.

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
  - 27B (all n=8 except CodeAct b40 n=7): **RLM 39.38 ≳ CodeAct
    36.96 ≫ ReAct 25.16** — RLM and CodeAct close at the top, ReAct far back.
  Narrative: ReAct is the floor-raiser for weak reasoners; at the top,
  CodeAct draws *level* with RLM (−2.4pp, within noise) rather than
  pulling clear. Supports an append-only-code FT target *given* a capable
  base model — at ~no cost vs RLM.

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
| 27B/27B | **39.38** (`rvlm`, n=8) | 25.16 (n=8) | 36.96 (b=40, n=7) | RLM ≳ CodeAct ≫ ReAct |

> **CodeAct-27B locked (2026-06-04).** Run on amax1 as a 3-budget sweep
> (`max_iterations` ∈ {24, 40, 56}): **b24 37.66 ± 5.02 (n=8), b40 36.96 ±
> 5.25 (n=7), b56 35.62 ± 2.59 (n=8)**; pooled **36.74 ± 4.29 (n=23)**.
> Budget is flat (~36–38%, noise-dominated) and the cap never binds
> (~1% @cap at every budget). The harness cell uses **b=40** (default
> budget, comparable to the 4B/9B CodeAct cells). b40 is n=7 — its t8
> `maps_2` doc hit the 14400s task-timeout twice and was dropped. Detail:
> `docs/experiments/codeact-qwen-3_5-27b.md`.

**Reading (27B row now fully locked):**
- **CodeAct scales hardest, and at 27B it lands *near* RLM — not clear of
  it.** Worst at 8B (9.50), catches RLM at 9B (24.26 ≈ 24.54), and at 27B
  is **36.96, ~2.4pp under RLM's 39.38 (within noise)**. With the RLM-27B
  number in hand, RLM ≳ CodeAct at the top, both close. The FT-relevant
  read: the clean append-only-code MDP target costs ~no accuracy vs
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

## v3 — reasoning vs perception (LLM=27B / VLM=9B), DONE (n=8)

The missing factorial corner: a **strong reasoner on weak perception**,
to pair against v2's weak-reasoner-on-strong-perception (9B-LM/27B-VLM).
run_ids `{rvlm-minimal,react,codeact}-27b-llm-9b-vlm-val-t*`, **n=8 locked**
(RLM/CodeAct completed on amax1 2026-06-06, 27B DP=2 @8927 + 9B @8909).

**Locked n=8 — all three harnesses:**

| Harness | 9B-LM / 27B-VLM (v2) | 27B-LM / 9B-VLM (v3) | Δ | lean |
|---|---|---|---|---|
| RLM | 24.54 | **34.82 ± 3.01 (n=8)** | **+10.3** | **reasoning-bound** |
| CodeAct | 24.26 | **30.43 ± 2.86 (n=8)** | **+6.2** | **reasoning-bound** |
| ReAct | 21.01 | **17.96 ± 3.94 (n=8)** | **−3.05** | **perception-bound** |

RLM v3 per-trial: 32.60 / 34.74 / 32.50 / 35.00 / 32.50 / 32.50 / 40.00 / 38.75.
CodeAct v3 per-trial: 27.87 / 35.24 / 32.81 / 28.75 / 30.00 / 31.25 / 31.25 / 26.25.
(amax7 ran t1–t2 RLM / t1–t3 CodeAct + ReAct; amax1 finished the rest.)

ReAct v3 per-trial: 14.83 / 17.91 / 21.44 / 19.27 / 14.23 / 15.71 / 25.48
/ 14.84. **ReAct v3 is the only harness that *loses* going from
9B-LM/27B-VLM → 27B-LM/9B-VLM** (−3.05pp, n=8) — a stronger reasoner on
weaker perception can't recover the lost VLM acuity because it has no
crop/zoom actuator. RLM (+10.3) and CodeAct (+6.2) both *gain* from the
reasoner swap (n=8, locked).

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
27B-reasoner/9B-VLM corner *gains* (RLM +10.3, CodeAct +6.2) → these
harnesses are **reasoning-bound**. This is a direct corollary of the D-006
active-perception thesis: the REPL crop/zoom loop is the mechanism that
**turns reasoning capability into perception quality**; remove it (ReAct)
and reasoning can no longer buy perception. **Locked at n=8.**

## Status

In progress. Order (per-user 2026-06-02): R1 ✅ → CodeAct sweep
(8B v2 → 9B v2 → 4B v2 → 27B/27B → 4B v1 → 9B v1) → deferred ReAct
R2–R5. Driver state: `tmp/workspace/qwen-9b-vlm-axis/driver-state.md`.

**27B/27B CodeAct (CA-27B) — LOCKED on amax1 (2026-06-04).** Ran as a
3-budget sweep (`max_iterations` ∈ {24, 40, 56}): b24 37.66 ± 5.02 (n=8),
b40 36.96 ± 5.25 (n=7), b56 35.62 ± 2.59 (n=8); pooled 36.74 ± 4.29
(n=23). Budget flat, cap never binds. The 27B-row CodeAct number uses
**b=40** (default budget, comparable to the 4B/9B cells); b40 is n=7 (t8's
`maps_2` hit the 14400s task-timeout twice → dropped). Per-trial data:
`codeact-qwen-3_5-27b.md`. RLM-27B (39.38, `rvlm`) and ReAct-27B (25.16)
are final from the main 8-solver matrix (`docs/results.md`). This was the
config-identical reuse of amax1's `codeact-b40` budget arm per the
2026-06-04 27B-only directive (no separate `codeact-3_5-27b-val` re-run).
