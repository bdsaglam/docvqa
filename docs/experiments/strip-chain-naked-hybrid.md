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

## Status and paper actions

| variant | Status | Paper action |
|---|---|---|
| `rvlm_naked` | **shelved**, n=1 fixed | Reported as ablation: stripping APPROACH + verify-under-VLM-stochasticity costs ~10pp at n=1. Direct evidence those are load-bearing content. No n=2/n=8 escalation needed; the directional conclusion is locked. |
| `rvlm_skeletal` | n=8 chain in progress (post n=2 paired Δ = −2.94pp) | Either promote (if Δ → 0 at n=8) or report as honest small generality cost. |
| `rvlm_hybrid` | n=8 chain queued (post-skeletal) for parity | Reported as architecture comparison: agent prefers direct perception but it scores worse; multi-image-window fragility makes it un-robust. Strengthens the case for the always-delegate rvlm design. |

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

3. **The 3 doc-shape pattern bullets in `rvlm_minimal` are
   ~unaccountable** (`skeletal` evidence). Removing them at n=2 costs
   −2.94pp on common subset, n=1 full-set tied minimal exactly. n=8
   needed to call this a true tie vs a small cost.
