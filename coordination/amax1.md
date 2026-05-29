# amax1 queue (throughput host)

amax1 is the throughput host — runs side-track experiments where the
direction is already known and we just need to lock numbers. No adaptive
iteration; if a cell shows an unexpected direction, **halt and append a
`## NOTE FOR AMAX7`** section at the bottom of this file.

**vllm:** brings up its own per-model containers as needed. Set
`MODEL_TAG` per cell.

## In progress

### `[→]` R4: direct_vlm cap=40 @ images_for_last_n=1

run_id `direct-vlm-iln1-val-iter40-t1`, tmux `docvqa-dvm:iln1`.
Launched 2026-05-29 overlapping R3's tail. Tests whether il_n=1 fixes
the comics crash (BadReq64=0 so far) and how trimming kept-image
context affects accuracy. R5 (il_n=2) overlaps when R4 hits 23/25.

## Queued

### images_for_last_n sweep for direct_vlm (replaces old n=2 trials)

User directive 2026-05-29: run `direct_vlm` cap=40 at
`images_for_last_n=1` and `=2` **instead of** the confirmatory n=2
trials (old R4 minimal-t2 / R5 direct_vlm-t2 are CANCELLED). Motivation:
R3 (il_n=3 default) crashes on long comics docs — the solver
`display()`s images into its own context and exceeds vllm's 64-image
limit. Trimming kept-image-context should fix the crash; the sweep also
measures the accuracy effect.

- **R4** `direct-vlm-iln1-val-iter40-t1`: `solver=direct_vlm
  solver.max_iterations=40 solver.images_for_last_n=1`
- **R5** `direct-vlm-iln2-val-iter40-t1`: `solver=direct_vlm
  solver.max_iterations=40 solver.images_for_last_n=2`

Comparison: R3 (il_n=3, crashes on comics) vs R4 (il_n=1) vs R5 (il_n=2),
all direct_vlm cap=40. Driven by the cron chain (afae64f5).

### 3. `[ ]` Gemma 4 E4B baseline + scaffold n=1 val (task #8 part 1)

Re-run on clean prompts (D-009). Original 2026-05-09 cells used
pre-scrub prompts. Direction is robust (+5.83pp lift in original n=3);
just locking the magnitude under clean prompts.

```bash
# Bring up vllm for Gemma 4 E4B (see existing setup notes in
# docs/experiments/gemma-4-e4b-baseline-scaffold.md)
TRIALS=1 SOLVER=rvlm bash scripts/run_gemma_chain.sh gemma-4-e4b-vllm-local 4-e4b
```

- Expected wall: ~2-3h (baseline + scaffold)
- Expected direction: lift sign preserved (~+5pp baseline → scaffold)

### 4. `[ ]` Qwen 3.5 9B baseline + scaffold n=1 val (task #8 part 2)

```bash
TRIALS=1 SOLVER=rvlm bash scripts/run_gemma_chain.sh qwen-3_5-9b-vllm-local 3_5-9b
```

- Expected wall: ~2-3h
- Expected direction: lift sign preserved (~+6pp from original n=3)

### 5. `[ ]` Gemma 4 31B baseline + scaffold n=1 val (task #8 part 3)

```bash
# Per docs/experiments/gemma-4-31b-baseline-scaffold.md: needs
# vllm --tensor-parallel-size 4 --enforce-eager to survive scaffold load
TRIALS=1 SOLVER=rvlm bash scripts/run_gemma_chain.sh gemma-4-31b-vllm-local 4-31b
```

- Expected wall: ~3-4h
- Expected direction: lift sign preserved (~+25pp from original n=3)

## Done

### R3. `[✓ PARTIAL]` direct_vlm n=1 val @ cap=40, il_n=3 (task #19) — 2026-05-30

run_id `direct-vlm-val-iter40-t1`. **23/25 docs (74 q); `comics_2` &
`comics_4` UNSCOREABLE** — `direct_vlm` at il_n=3 crashed 4× on those
two long comics docs (>64-image / sandbox kill; 3 `BadRequestError:
64 images` logged). Resumed across attempts (16→21→23 docs). 47 cap40
hits. **q-weighted accuracy over the 74 completed q = 43.2%.**
Per-cat: infographics/science_poster/slide 60%, eng_drawing/science_paper
40%, business_report 30%, comics 50%(2/4), maps 10%.

**★ HEADLINE FINDING — full prompt is hugely load-bearing for the
DIRECT architecture (opposite of rvlm).** R3 and R2 cover the
**identical 74-q subset** (both miss only comics_2/comics_4), so paired:
| cap40, same 74 q | prompt | score |
|---|---|---|
| R2 `direct_vlm_minimal` | stripped | 21.6% (16/74) |
| R3 `direct_vlm` | full (TOOL_HINTS+tips) | **43.2% (32/74)** |
**Δ ≈ +21.6pp** from the full prompt — far beyond n=1 noise (~3pp).
This is the OPPOSITE of the rvlm family, where minimal ≈ unified
(prompt content NOT load-bearing, Δ=+1.09pp n.s.). Coherent story:
the **recursive (VLM-sub-call) architecture is robust to prompt
minimization; the direct (display-into-own-context) architecture
depends heavily on prompt scaffolding.** Caveats: n=1 each, partial
74/80 (internally valid — identical subset), single model (Qwen 27B).
Worth a paired n≥2 confirmation before it goes in the paper, but the
magnitude is striking. (Also note R3's 43.2% > R2's 21.6% > R1's
27.5%cap20 — direct+full-prompt is the strongest direct-VLM config so
far on this subset.)

### R2. `[✓]` direct_vlm_minimal n=1 val @ cap=40 (task #34) — 2026-05-29

run_id `direct-vlm-minimal-val-iter40-t1`, c=24, `max_iterations=40`.
**Overall 21.6% (16/74)** — per-cat: infographics 60%, science_paper
30%, science_poster/slide 20%, business_report/eng_drawing/maps 10%,
**comics 0% (0/4)**.

**⚠ INCOMPLETE DENOMINATOR (74, not 80).** `comics_2` and `comics_4`
each hit the 4h `task_timeout` (14400s) at 18:47Z and errored out
(6 questions never scored). So 21.6%/74 is NOT directly comparable to
R1's 27.5%/80. On the shared 74 questions the comparison is still
roughly flat-to-down.

**Headline finding (R1 vs R2 — iteration-budget effect): doubling the
cap did NOT help.** 58/74 questions still hit the 40/40 cap (vs 62/80
hitting 20/20 in R1), and the score did not rise. The agent does not
converge at *either* budget — it loops/wanders to whatever ceiling it
is given. **R1's low score was therefore NOT cap-truncation**; the
iteration budget was not the bottleneck. This weakens the original
"cap=20 was binding" hypothesis and shifts the read toward solver
behavior / task difficulty. NOTE for the chain: R4/R5 (more cap=40
trials) are now lower-value given this — flag for user on return.

Operational: logfire/otlp telemetry threw protobuf DecodeErrors
throughout (harmless to scoring). comics docs are pathological for the
direct-VLM family at cap=40 (4h timeout) — R3/R4/R5 will likely share
the <80 denominator problem.

### R1. `[✓]` direct_vlm_minimal n=1 val @ cap=20 (task #34) — 2026-05-29

run_id `direct-vlm-minimal-val-t1`, c=24, default `max_iterations=20`.
**Overall 27.5% (22/80).** Per-category: slide 60%, infographics 50%,
science_paper 30%, business_report/comics/science_poster 20%,
engineering_drawing/maps 10%. Wall ~96min.

**Key signal: 62/80 questions (78%) hit the 20/20 iteration cap.** The
low score is therefore confounded — cannot attribute to prompt
content until the cap is relaxed. Hence the cap=40 re-run (R2) and the
decision to run the whole direct-VLM family at cap=40. NO direct_vlm
baseline comparison yet (R3 will provide it at equal cap). Logfire
telemetry threw protobuf DecodeErrors throughout (broken span export)
but did NOT affect scoring/submission — harmless.

### Autonomous chain plan (R1→R5, all on the 27B @ 8927; user away 2026-05-29)

Strictly sequential on local vllm; driven by heartbeat cron. No vllm
swaps (safe while unattended).
- R1 `direct-vlm-minimal-val-t1` cap=20 — ✓ 27.5%
- R2 `direct-vlm-minimal-val-iter40-t1` cap=40 — ✓ 21.6%/74
- R3 `direct-vlm-val-iter40-t1` (direct_vlm, il_n=3 default) cap=40 — running (crashes on comics)
- R4 `direct-vlm-iln1-val-iter40-t1` (direct_vlm, **images_for_last_n=1**) cap=40
- R5 `direct-vlm-iln2-val-iter40-t1` (direct_vlm, **images_for_last_n=2**) cap=40

(R4/R5 changed 2026-05-29 from the original n=2 confirmatory trials to
the images_for_last_n sweep, per user "instead of more trials".)

Reads: R1-vs-R2 = iteration-budget effect (≈flat → budget not the
bottleneck); R3 vs R4 vs R5 = images_for_last_n sweep (crash-fix +
accuracy effect of trimming kept image-context); R2(minimal) vs
direct_vlm runs at equal cap = prompt/architecture; rvlm headline is
cap=20 (flagged in amax7.md).

### NOTE: model-axis cells #3-5 DEFERRED (need attention on return)

Not run unattended. Reasons in
`tmp/workspace/amax1-model-axis/vllm-bringup-notes.md`: (1) per-model
vllm bringup is unscripted and Gemma-4 needs an undocumented tool-call
parser (wrong parser → silent tool-call failure → fake-low scores);
(2) #5 Gemma-31B as-written needs TP=4 but amax1 has only 3 GPUs. The
vllm `docker run` template + per-model ports are in that file.

### A. `[✓]` ReAct baseline n=8 val — REPL-vs-no-REPL ablation

Finished 2026-05-29T01:53. c=24, `lm.timeout=1800` (overridden via the
new `LMConfig.timeout` field), run_ids `react-val-t1..t8`. **n=8 mean
= 30.47%, std 3.06pp, range 25.0–33.75%.** Paired vs rvlm n=8: **Δ =
−10.47pp, 95% CI [−13.42, −7.52]pp** (cleanly outside ±1.5pp noise).
Paired vs rvlm_unified n=8: **Δ = −10.47pp, 95% CI [−14.46, −6.48]pp**.
Lands in "≤ −5pp → REPL is load-bearing" per the pre-set decision
table. Largest per-category gaps (rvlm−react): `business_report`
(+23.8pp), `engineering_drawing` (+23.8pp), `comics`/`science_poster`
(+13–14pp) — exactly the zoom-then-read categories where the REPL's
`pages[i].crop()` matters. Operational notes: 3 `litellm.Timeout`s on
original chain (`science_poster_2`/`business_report_2`), recovered via
2 backfill passes; the second backfill exposed that the 600s default
timeout doesn't fit ReAct's long-trajectory completions, fixed via
config-driven `LMConfig.timeout` (commit `3acee78`). Full per-trial
tables, paired Δ stats, per-category breakdown, and the timeout
calibration story in [docs/experiments/react-baseline.md](../docs/experiments/react-baseline.md).

### 1. `[✓]` rvlm n=8 val — paired-comparison anchor for unified-tips (task #28)

Finished 2026-05-28T16:45 (c=24, run_ids `rvlm-val-t1`..`rvlm-val-t8`).
**n=8 mean = 40.94%, std 2.29pp, range 36.25%–43.75%.** Paired vs amax7
`rvlm_unified` n=8: **Δ = 0.00pp** (both arms 262/640 total correct),
95% CI [−3.91, +3.91]pp — lands cleanly in "promote unified to default"
per the pre-set decision table. Operational note: t6→t7 transition hit
a same-`run_id` contamination incident (chain's auto-t7 launched in
parallel with a standalone t7); recovered via clean restart on
`science_paper_1`. Runner's silent-timeout fallback was changed to
record-error-and-retry mid-chain (commit `8309710`). Full per-trial
tables, paired Δ stats, contamination story, and silent-failure audit
in [docs/experiments/unified-category-tips-ablation.md](../docs/experiments/unified-category-tips-ablation.md).

## Decision rules (set in advance)

- **All three cells: lift sign preserved** → model-axis prediction 1 is
  robust under clean prompts. Escalate to n=2 (then n=8 only after the
  paper headline locks).
- **Any cell: lift sign reverses** → halt this queue and `## NOTE FOR
  AMAX7`. Pre-scrub prompts were inflating the lift differentially
  somehow; need to investigate before drawing model-axis conclusions.
- **Magnitude shifts by 5+pp** → expected (per-trial std ~3-4pp on
  Qwen 27B; smaller models can be noisier). Note in the result line;
  still escalate to n=2.
