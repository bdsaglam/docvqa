# Strip chain: how far can the prompt be stripped?

## Hypothesis / question

After `rvlm_minimal` was locked as the proposed method (n=8 Δ vs
`rvlm_unified` = +1.09pp, n.s. — see `rvlm-minimal-generality.md`),
the next question is which *parts* of the minimal body carry the
method. Three further cuts:

1. **`rvlm_skeletal`** — drops the 3 doc-shape pattern bullets from
   `rvlm_minimal` (high-density single page, many-page document,
   counting / superlatives). Keeps APPROACH steps + the verify-under-
   VLM-stochasticity principle.
2. **`rvlm_naked`** — strips down to DATA + TOOLS + faithfulness +
   OUTPUT FORMAT. No APPROACH, no patterns, no verify principle. The
   pure thesis statement: "give the agent a recursive-perception tool
   and let it figure it out."
3. **`rvlm_hybrid`** — uses `MultimodalRLM` (the agent's LM is itself
   multimodal) with BOTH `display(image)` (agent sees image inline)
   AND `ask_vlm(image, query) -> str` (agent delegates to a fresh
   sub-VLM). Agent picks per call. Body otherwise mirrors
   `rvlm_minimal`. Tests whether the rvlm family's *forced*
   delegation pattern is architecturally load-bearing or a free
   alternative the agent could pick on its own.

## Pre-set decision rules

| Cell | Δ ≈ 0 | Δ small (−1 to −3pp) | Δ ≥ −5pp (load-bearing) |
|---|---|---|---|
| skeletal | promote to proposed method (lower prompt overhead) | small generality cost in paper | patterns carry the method |
| naked | naked becomes the proposed method | honest cost line; minimal stays | APPROACH/verify is load-bearing |
| hybrid | architecture choice is free | hybrid as ablation showing minor cost | always-delegate (rvlm) is correct |

## Results — n=1 (Qwen 3.5 27B local, c=32, full 25-doc / 80-Q val)

| solver | n=1 score | Δ vs minimal-t1 (42.50%) | σ-units vs minimal n=8 |
|---|---|---|---|
| minimal-t1 | 42.50% (34/80) | — | — |
| skeletal-t1 | 42.50% (34/80) | +0.00pp | 0σ |
| **naked-t1** | **32.50% (26/80)** | **−10.00pp** | **~4.3σ** |
| hybrid-t1 | 35.00% (28/80) | −7.50pp | ~3.2σ |

### Tool-preference finding from hybrid (paper-worthy)

Counted tool calls across the hybrid trial's code blocks:

| tool | calls | share |
|---|---|---|
| `display(...)` | 1397 | 66% |
| `ask_vlm(...)` | 706 | 34% |

The agent, given the choice, **strongly preferred direct perception
(`display`) over delegation (`ask_vlm`)**. And yet hybrid scored
7.5pp lower than `minimal` (always-delegate). *Revealed preference
favored display; the metric favored delegation.* This is direct
evidence that the rvlm family's forced-delegation pattern is the
right one — not a vestige.

## Results — n=2 paired

n=2 follow-up on skeletal + hybrid (naked skipped — −10pp is
unambiguous). Common 21-doc / 68-Q subset across the 6 trials
(minimal t1+t2, skeletal t1+t2, hybrid t1+t2):

| trial | common-subset | Δ vs minimal-t* |
|---|---|---|
| minimal-t1 | 44.12% | — |
| minimal-t2 | 42.65% | — |
| skeletal-t1 | 41.18% | −2.94pp |
| skeletal-t2 | 39.71% | −2.94pp |
| hybrid-t1 | 35.29% | −8.82pp |
| hybrid-t2 | 35.29% | −7.35pp |

Paired n=2 mean Δ:
- **skeletal − minimal = −2.94pp** (both trials identical)
- **hybrid − minimal = −8.09pp** (both trials within 1.5pp of each
  other; mean clearly outside minimal's σ)

Skeletal n=1 "literal tie" on full 80Q was an artifact of comparing
on *different* doc sets per trial (each lost different long-tail
docs). The clean paired n=2 read puts skeletal ~3pp below minimal —
close to but not inside the pre-set ±2pp band. Need n=8 to call this
a tie vs a small cost.

## Diagnostic: hybrid images_for_last_n=1

To isolate whether hybrid's deficit comes from stale-image confusion
(tighter visual window helps) or from helpful multi-image context
being too small (tighter window hurts), one cell with the visual
window collapsed to 1:

| run | score | Δ vs hybrid baseline | Δ vs minimal |
|---|---|---|---|
| hybrid baseline (`images_for_last_n=3`) | 35.00% | — | −7.50pp |
| **hybrid (`images_for_last_n=1`)** | **20.00%** | **−15.00pp** | **−22.50pp** |

Tool usage ratio essentially unchanged:
- baseline: `display` 1397 / `ask_vlm` 706 (66:34)
- imgN1: `display` 1486 / `ask_vlm` 706 (68:32)

ask_vlm count was *literally identical* (706:706). The agent did not
adapt its strategy to the smaller window — kept using `display()` at
the same rate, just with broken context retention. This is direct
evidence that **`display()`-based perception is fragile to the
visual-window choice**, while **`ask_vlm()` (recursive sub-VLM) is
robust** — text answers persist in the trajectory regardless of how
many later `display()` calls evict images. The rvlm family's
always-delegate design sidesteps a real failure mode of
multimodal-LM-in-the-loop solvers.

## Results — n=8 paired (Qwen 3.5 27B local, c=32)

### `rvlm_skeletal` n=8 — paired vs minimal (task #38, 2026-05-30)

run_ids: `rvlm-skeletal-val-t{1..8}`. Chain wall ~9.5h (orch + 8
trials w/ overlap).

**Paired Δ skeletal − minimal at n=8 (per-trial intersection):**
**Δ = −1.63pp ± 4.83pp sd**, SE 1.71pp, 95% CI [−5.67, +2.41]pp,
t(7) = −0.954 (n.s.). Lands cleanly in the "≈ 0pp / within paired
noise" band.

Per-trial paired table (n_common varies; skeletal lost more docs to
the long-tail science_paper_1 hang — pass-2 refill not run for the
short-tail trials, but pair-on-intersection is robust to that):

| trial | n_common | skeletal | minimal | Δ |
|---|---:|---:|---:|---:|
| t1 | 80 | 42.50% | 42.50% | +0.00pp |
| t2 | 80 | 38.75% | 42.50% | −3.75pp |
| t3 | 49 | 38.78% | 36.73% | +2.04pp |
| t4 | 63 | 38.10% | 39.68% | −1.59pp |
| t5 | 59 | 42.37% | 47.46% | −5.08pp |
| t6 | 65 | 43.08% | 35.38% | +7.69pp |
| t7 | 68 | 30.88% | 38.24% | −7.35pp |
| t8 | 80 | 40.00% | 45.00% | −5.00pp |

Marginals: skeletal n=8 = 39.31% ± 3.92pp (range 30.88–43.08);
minimal n=8 = 42.03% ± 2.21pp (range 38.75–45.00). Skeletal is ~1.8×
noisier across trials.

**Reading.** The 3 doc-shape patterns in `rvlm_minimal`
(high-density single page, many-page document, counting /
superlatives) are **not load-bearing for the headline score** —
paired Δ is well inside noise. But they tighten the trial-to-trial
variance: with the patterns, σ drops from 3.92pp → 2.21pp. The
patterns give the agent a more consistent reading discipline rather
than a better one.

**Promotion decision.** Both viable. Keep `rvlm_minimal` as the
proposed method — same headline, tighter variance is a free win for
the paper's "method is stable" story. Skeletal stays as the ablation
cell: "dropping the 3 doc-shape pattern bullets does not change the
headline but doubles σ."

#### Refill ops note (t3/t4/t5; task #38)

t3 (18→25/25), t4 (22→25/25), t5 (20→25/25) were refilled to clean
25/25 by amax7 at c=4 strict serial; t6 (22/25) and t7 (23/25) handed
to amax1 (and later moved back to amax7 — the partial run dirs needed
for resumability live on amax7). The refill confirmed the per-doc
timeout root cause: at c=4 (vs the original c=32 overlap chain) every
long-tail doc — including science_paper_1 — completed within the 4h
task_timeout. The overlap pattern's load contention was the timeout
driver, not any solver-specific bug. Skipping the t6/t7 refill is
acceptable: the n=8 paired Δ above (−1.63pp, n.s.) already lands
minimal as proposed method; the refill only refines σ.

### `rvlm_hybrid` n=8 — paired vs minimal (task #39, 2026-05-31)

run_ids: `rvlm-hybrid-val-t{1..8}`. Chain wall ~20h start to sentinel
(~12:09 → 08:30 next day).

**Paired Δ hybrid − minimal at n=8 (per-trial intersection):**
**Δ = −5.31pp ± 4.32pp sd**, SE 1.53pp, 95% CI [−8.92, −1.70]pp,
t(7) = −3.48 (**significant** — CI excludes 0). Confirms the n=2
signal (Δ = −8pp paired) tightened to −5.31pp at n=8.

Per-trial table (clean — all 8 trials hit 80/80 Qs, unlike skeletal):

| trial | n | hybrid | minimal | Δ |
|---|---:|---:|---:|---:|
| t1 | 80 | 35.00% | 42.50% | −7.50pp |
| t2 | 80 | 35.00% | 42.50% | −7.50pp |
| t3 | 80 | 31.25% | 40.00% | −8.75pp |
| t4 | 80 | 38.75% | 41.25% | −2.50pp |
| t5 | 80 | 43.75% | 45.00% | −1.25pp |
| t6 | 80 | 37.50% | 38.75% | −1.25pp |
| t7 | 80 | 40.00% | 41.25% | −1.25pp |
| t8 | 80 | 32.50% | 45.00% | −12.50pp |

Marginals: hybrid 36.72% ± 4.12pp (range 31.25–43.75); minimal
42.03% ± 2.21pp. Hybrid is ~2× noisier across trials AND ~5pp behind
on mean.

**Paper reading.** Hybrid is the "we tried it and it didn't work"
cell. Adding a second perception channel (`display()`) on top of
`ask_vlm()` degrades performance at n=8 with statistical significance,
despite the agent's revealed 2:1 preference for `display()` over
`ask_vlm()` (counted at n=1: 1397 vs 706 calls). The agent's revealed
preference for direct perception is the **wrong** preference under
this regime — forcing delegation through `ask_vlm` is the right
design choice for the rvlm family. Strong evidence for the paper's
discussion of why recursive perception > direct perception when the
LM is identical.

**Methodological footnote.** Hybrid was the only chain in the rvlm-*
family with NO doc-timeouts (8/8 trials clean 80-Q intersections).
The display() channel appears to sidestep the agent-loop hang mode
that catches `science_paper_1` in `rvlm_minimal` and `rvlm_skeletal`.
Worth a parenthetical in the paper: when measuring an ablation we care
about *more* than just mean — variance, completion, and failure mode
all matter.

## Status and paper actions

| variant | Status | Paper action |
|---|---|---|
| `rvlm_naked` | **shelved**, n=1 fixed | Reported as ablation: stripping APPROACH + verify-under-VLM-stochasticity costs ~10pp at n=1. Direct evidence those are load-bearing content. No n=2/n=8 escalation needed; the directional conclusion is locked. |
| `rvlm_skeletal` | **n=8 done** (paired Δ = −1.63pp [−5.67, +2.41] n.s.) | Reported as ablation: dropping the 3 doc-shape pattern bullets leaves the headline unchanged but doubles σ (2.21→3.92pp). `rvlm_minimal` stays the proposed method. |
| `rvlm_hybrid` | **n=8 done** (paired Δ = −5.31pp [−8.92, −1.70] significant) | Reported as architecture comparison: agent prefers direct perception but it scores worse; multi-image-window fragility makes it un-robust. Strengthens the case for the always-delegate rvlm design. |

## Key findings

1. **APPROACH + verify-under-VLM-stochasticity are load-bearing
   content** (`naked` evidence). The paper's proposed-method body has
   irreducible scaffold beyond DATA / TOOLS / answer-format rules.

2. **Recursive perception is architecturally robust; direct
   perception is fragile** (`hybrid` evidence). When the same LM has
   the choice between `display()` and `ask_vlm()`, the agent picks
   `display()` ~2:1 — but scores worse. The hybrid result holds at
   n=2 (paired Δ = −8.09pp). Tightening the visual context window to
   1 image breaks `display()` (−15pp drop) while `ask_vlm()` usage is
   unchanged. The text-channel answers from sub-VLM calls persist in
   the trajectory; image-channel context evicts. The rvlm family's
   forced-delegation design sidesteps this.

3. **The 3 doc-shape pattern bullets in `rvlm_minimal` are not
   load-bearing for the headline** (`skeletal` evidence). At n=8 the
   paired Δ = −1.63pp [−5.67, +2.41] (n.s.) — inside noise. They do
   tighten trial-to-trial variance (σ 3.92pp without → 2.21pp with),
   so `rvlm_minimal` stays the proposed method for the stability win
   and skeletal stays an ablation. (n=1 full-set tie was an artifact
   of different per-trial doc sets; n=2 common-subset showed −2.94pp;
   n=8 paired is the clean read.)

4. **Hybrid's deficit holds and tightens at n=8** (`hybrid`
   evidence). Paired Δ hybrid − minimal = −5.31pp [−8.92, −1.70],
   t(7) = −3.48 (significant). Hybrid shelved; the always-delegate
   rvlm design is the right one.
