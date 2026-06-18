# Harness × model axis — cross-cutting synthesis (val, n=8)

The cross-model sweeps repeat the active-perception design across three agent
**harness types** and several reasoner/VLM sizes. The raw per-cell numbers and
per-trial data live in the **per-model files** below; this doc holds the
**cross-cutting tables and narratives** that span them.

- **Harnesses.** RLM (`rvlm`, hidden Python-namespace state → POMDP) · ReAct
  (`react_baseline`, thought→tool→observation, **no code**) · CodeAct
  (`codeact`, action = Python in a REPL, **append-only context** → clean MDP;
  same tools/prompt/`batch_look` as `rvlm`, only the context discipline differs).
  The fine-tuning target is **not** an RLM — CodeAct's append-only trace is the
  RL-friendly MDP form, so the sweep asks whether it costs accuracy.
- **Perception configs.** v1 homog (LLM = VLM) · v2 mixed (small LLM + 27B VLM)
  · v3 (27B LLM + 9B VLM). v2↔v3 is the reasoning-vs-perception factorial.
- All n=8, val 25 docs / 80 Q, `enable_thinking=false`, per-question micro-avg.

**Per-model detail (raw numbers + per-trial):**
[`qwen-3_5-4b.md`](qwen-3_5-4b.md) · [`qwen-3_5-9b.md`](qwen-3_5-9b.md) ·
[`qwen3-8b.md`](qwen3-8b.md) · [`gemma-4-e4b.md`](gemma-4-e4b.md) ·
[`gemma-4-31b.md`](gemma-4-31b.md). The 27B-reasoner row is in the per-solver
files (`rvlm-qwen-3_5-27b.md`, `react_baseline-qwen-3_5-27b.md`,
`codeact-qwen-3_5-27b.md`) and `docs/results.md`.

> **⚠ STALE CodeAct — do not cite.** Every CodeAct (`CodeActᶜ`) number in
> this document is an **old-dspy `codeact`** result. The corrected
> **`codeact_chat`** twin is the sole source of truth for CodeAct numbers
> going forward, and these are retained for **provenance only**. Every
> finding derived from CodeAct numbers below — the **reasoner-scaling
> slope** (Finding 2), the **v3 reasoning-bound delta** (Finding 3,
> CodeActᶜ +6.2), the **Gemma CodeAct lifts** (Findings 4–5), and the
> **clean append-only-code MDP fine-tuning-target argument** — is
> therefore **provisional**, pending the `codeact_chat` re-run that
> replaces these rows. The analysis is kept intact, but no CodeAct number
> here is a current result.

## Table 1 — v1 homog (LLM = VLM) + 27B anchor (n=8)

| Config | RLM (`rvlm`) | ReAct | CodeActᶜ |
|---|---|---|---|
| Qwen3.5 4B homog | 12.49 ± 3.74 | 11.94 ± 2.23 | 12.19 ± 3.50 |
| Qwen3.5 9B homog | 16.67 ± 3.40 | 14.97 ± 2.96 | 19.35 ± 4.24 |
| Qwen3.5 27B homog | **39.38 ± 1.49** | 25.16 ± 4.60 | 36.96 ± 5.25 (b=40, n=7) |
| Gemma-4 E4B homog | 7.34 ± 3.30 | 6.09 ± 2.36 | 7.66 ± 1.94 |
| Gemma-4 31B homog | **32.50 ± 4.48** | 18.44 ± 3.58 | 29.25 ± 5.77 (n=5†) |

† Gemma-31B CodeAct n=5 (stopped early per user); score depressed by slow-doc
guards + gemma4-31B CodeAct operational instability — see below and
[`gemma-4-31b.md`](gemma-4-31b.md).

ᶜ **STALE — do not cite.** Old dspy `codeact` (deprecated). The corrected
**`codeact_chat`** twin is the sole source of truth for CodeAct numbers going
forward; a config without a `codeact_chat` value is **open** — the stale dspy
figure is shown for provenance only, not as a current result. Tracking and
replacements: `codeact-chat-qwen-3_5-27b.md`.

## Table 2 — v2 mixed (reasoner + 27B VLM), n=8

| Reasoner | RLM | ReAct | CodeActᶜ | best |
|---|---|---|---|---|
| Qwen3 8B (text-only, older gen) | 11.73 ± 2.96 | **15.79 ± 2.03** | 9.50 ± 1.44 | ReAct |
| Qwen3.5 4B | **21.09 ± 3.16** | 15.66 ± 4.73 | 15.66 ± 3.00 | RLM |
| Qwen3.5 9B | **24.54 ± 5.30** | 21.01 ± 4.63 | 24.26 ± 4.68 | RLM ≈ CodeAct |
| (27B/27B homog, for reference) | **39.38** | 25.16 | 36.96ᶜ (b40) | RLM ≳ CodeAct ≫ ReAct |

## Finding 1 — perception-budget lift (v1 → v2, swap only the VLM)

Holding the reasoner fixed and swapping **only** the VLM → 27B lifts the RLM
headline ~8pp at both Qwen3.5 sizes (**9B +7.87pp**, Welch t=3.54, sig.;
**4B +8.60pp**, t=4.96, sig.). The lift's consistency across reasoner size is
the signature of a **perception (not orchestration) bottleneck** → the scaffold
is **perception-budget-bound** for mid/small reasoners (supports D-006). Under
ReAct/CodeAct the same swap lifts less (4B +~3.5pp, 9B +5–6pp) — the weaker
harnesses can't convert better perception into accuracy as efficiently as RLM's
hidden-state REPL.

## Finding 2 — harness rank-flip with reasoner scale (v2, 27B VLM)

The best harness changes with reasoner capability:

- **8B (text-only): ReAct 15.79 > RLM 11.73 > CodeAct 9.50ᶜ.** The simplest
  harness wins for the weakest reasoner; CodeAct's append-only growing-code
  context is *hardest* (fails consistently — lowest mean, lowest variance).
- **4B: RLM 21.09 > ReAct ≈ CodeAct 15.66ᶜ.** RLM already leads.
- **9B: RLM 24.54 ≈ CodeAct 24.26ᶜ > ReAct 21.01.** RLM/CodeAct overtake ReAct
  once the reasoner can exploit code + state.
- **27B: RLM 39.38 ≳ CodeAct 36.96ᶜ ≫ ReAct 25.16.** RLM and CodeAct close at
  the top; ReAct far back.

**CodeAct scales hardest.** Worst at 8B → catches RLM at 9B → lands ~2.4pp under
RLM at 27B (within noise). **CodeAct reasoner-scaling slope (Qwen3.5):** 4B
15.66 → 9B 24.26 = **+8.6pp**ᶜ, vs RLM 4B 21.09 → 9B 24.54 = +3.5pp — CodeAct
benefits ~2.5× more from reasoner scale.

**FT implication.** The clean append-only-code **MDP** target (CodeAct) costs
~no accuracy vs hidden-state RLM *given a strong reasoner* (≥9B); below 9B, RLM
or plain ReAct is safer. ReAct is the floor-raiser for weak reasoners only; at
27B it lacks a REPL to compose multi-step perception and falls ~12–14pp behind.

## Finding 3 — reasoning vs perception (v2 vs v3), the mechanism

The missing factorial corner: **strong reasoner on weak perception** (v3:
27B-LM / 9B-VLM), paired against v2's weak-reasoner-on-strong-perception
(9B-LM / 27B-VLM). run_ids `{rvlm-minimal,react,codeact}-27b-llm-9b-vlm-val-t*`,
**n=8 locked**.

| Harness | v2 (9B-LM / 27B-VLM) | v3 (27B-LM / 9B-VLM) | Δ | lean |
|---|---|---|---|---|
| RLM | 24.54 | **34.82 ± 3.01** | **+10.3** | **reasoning-bound** |
| CodeActᶜ | 24.26 | **30.43 ± 2.86** | **+6.2** | **reasoning-bound** |
| ReAct | 21.01 | **17.96 ± 3.94** | **−3.05** | **perception-bound** |

Per-trial — RLM v3: 32.60 / 34.74 / 32.50 / 35.00 / 32.50 / 32.50 / 40.00 /
38.75. CodeAct v3: 27.87 / 35.24 / 32.81 / 28.75 / 30.00 / 31.25 / 31.25 /
26.25. ReAct v3: 14.83 / 17.91 / 21.44 / 19.27 / 14.23 / 15.71 / 25.48 / 14.84.

**The REPL is what converts reasoning into perception (why ReAct is the lone
perception-bound harness).** ReAct's only perception actuators are
`look(page, query)` / `look_many(pages, query)` — **whole-page** VLM queries at
fixed page granularity. It cannot crop a region, zoom, composite, or do
coordinate arithmetic on an image (no Python; `react_baseline_solver.py`
docstring: *"no crops … no PIL.crop on retrieved page images"*). So ReAct's
perception ceiling **is** the VLM's whole-page acuity: a stronger reasoner has
no actuator to direct finer-grained perception, so swapping 9B→27B reasoner
while the VLM stays at 9B buys nothing (it even regresses −3.05pp) — ReAct is
**perception-bound**.

RLM and CodeAct write Python around the same `batch_look`: they crop to the
evidence region, zoom, retry tighter, composite panels, do coordinate
arithmetic. A stronger reasoner produces **better-targeted sub-images** and
extracts more *even from a weaker (9B) VLM* — so the 27B-LM/9B-VLM corner
*gains* (RLM +10.3, CodeAct +6.2) → these harnesses are **reasoning-bound**.
This is a direct corollary of the D-006 active-perception thesis: the REPL
crop/zoom loop is the mechanism that **turns reasoning capability into
perception quality**; remove it (ReAct) and reasoning can no longer buy
perception.

## Finding 4 — cross-family robustness (Gemma vs Qwen)

The harness ordering reproduces on a second model family:

- **31B: rvlm 32.50 ≫ react 18.44 (+14.1pp ≫ std)** — the recursive VLM
  sub-call is **load-bearing**; REPL-only ReAct collapses to the no-recursion
  tier, mirroring Qwen 27B (rvlm 39.4 ≫ react 25.2). "Recursive-perception ≫
  tool-only ReAct" is **robust across model families**.
- **E4B: all three harnesses tied 6–8%** (within ~1 std) — a 4B model is too
  weak to exploit any scaffold. Clean negative control: capacity gates whether
  the scaffold can be driven; lift is *sharp* at 31B, *absent* at 4B.

## Finding 5 — harness-lift vs no-scaffold baselines (per-model)

Each harness measured against the two no-scaffold baselines on the same model.
**Baseline = max(`raw_vlm_multi_baseline`, `official_baseline`)** (the stronger
no-scaffold point). All n=8 except Gemma-31B CodeAct (n=5, stopped early).

| Model | rawvlm | official | base | rvlm (lift) | CodeActᶜ (lift) | ReAct (lift) |
|---|---|---|---|---|---|---|
| **Gemma-4 31B** | 10.78 ± 0.93 | 11.09 ± 1.82 | 11.09 | **32.50 (+21.4)** | 29.25 n=5 (+18.2) | 18.44 (+7.4) |
| **Gemma-4 E4B** | 3.75 ± 0.00 | 6.25 ± 1.16 | 6.25 | 7.34 (+1.1 n.s.) | 7.66 (+1.4 n.s.) | 6.09 (−0.2 n.s.) |

**The lift is a capacity gate, cleanly bracketed by one model family.** At 31B
**every harness clears both no-scaffold baselines by ≫ the std** — scaffolding
buys +7 to +21pp. At 4B **no harness clears the `official_baseline`** — every
lift is within noise. Same family, same code, same prompts: the only thing that
changes is base capability, and it gates whether *any* scaffold pays off. This
is the clean cross-family confirmation of the D-006 active-perception thesis.

> **Gemma-31B CodeAct operational caveat.** CodeAct n=5 (stopped early per user
> call). Its 29.25 is **depressed by slow-doc guards** (placeholder-zeroed docs:
> t2 4-doc, t3 1-doc, t4 3-doc, t5 1-doc) **and by gemma4-31B CodeAct
> instability** — across the sweep CodeAct hit **8 shm-crashes + repeated
> degenerate-generation and max-iteration runaways** (single question emitting
> 10k+ tokens at ~20 tok/s, never terminating). `rvlm`/`react`/the baselines ran
> clean. **CodeAct-on-gemma4-31B is operationally fragile** — itself a finding;
> true accuracy is likely somewhat above 29.25 but clearly in the scaffold tier.

## Status

n=8 locked for: Qwen3.5 {4B, 9B} (v1+v2, all harnesses), Qwen3 8B (v2),
v2↔v3 factorial (all harnesses), Gemma {E4B (3 harnesses + 2 baselines), 31B
(rvlm+react + 2 baselines)}; Gemma-31B CodeAct n=5 (stopped early per user).
**Per-model harness-lift table (Finding 5) complete for both Gemma sizes.**
Coordination: `coordination/amax1.md`, `coordination/amax7.md`.
