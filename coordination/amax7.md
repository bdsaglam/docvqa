# amax7 queue (adaptive host)

amax7 is the adaptive host — runs critical-path experiments where the
result might change the experiment plan. Tighter feedback loops; one
cell at a time; replan after each result.

**vllm:** Qwen 3.5 27B at `localhost:8927`.

## In progress

### `[→]` skeletal n=8 + hybrid n=8 chained — D-008 escalation — 2026-05-30

Two chains running on amax7:

- **Stage 1**: `scripts/skeletal_n8_orch.py` drives skeletal t2..t8
  with 22/25 overlap. Tmux: `skel-n8-orch` + per-trial `rvlm-skel-tN`.
  Sentinel: `/tmp/skeletal-n8.done`. Wall: ~5-6h.
- **Stage 2** (auto-chained): `scripts/hybrid_n8_post_orch.py` waits
  for stage-1 sentinel, then drives hybrid t3..t8 with same overlap.
  Hybrid t1+t2 already done. Wall: ~10-12h after stage 1.
  Sentinel: `/tmp/hybrid-n8.done`.

Naked **shelved** (n=1 −10.00pp, ~4.3σ outside minimal's noise band;
pre-set "−5pp or worse → load-bearing" rule triggered — no n=2/n=8
needed). Full writeup in
`docs/experiments/strip-chain-naked-hybrid.md`.

## Queued

### 1. `[ ]` rvlm_ocr n=1 val (task #14)

Locks the clean OCR-extension number. Current `rvlm_full` legacy data is
confounded with `look()` ergonomic wrapper. This is the clean cell.

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_ocr \
  data.split=val data.num_samples=null \
  max_concurrency=16 \
  run_id=rvlm-ocr-val-t1
```

- Expected wall: ~50min
- Compare to: `rvlm` val (OCR-free); expect approximately equal on
  DocVQA-2026 (moderate-length docs).

### 2. `[→moved]` direct_vlm n=1 val (task #19) — MOVED to amax1 (2026-05-29)

Moved to `coordination/amax1.md` to run paired with `direct_vlm_minimal`
on the throughput host, at **`max_iterations=40`** (the minimal cell
surfaced that cap=20 is binding — ~56/80 questions hit 20/20). Running
both direct-VLM-architecture cells at the same cap keeps the
prompt-stripping (TOOL_HINTS) comparison clean.

**Caveat for the paper claim (decision rules below):** `direct_vlm`
will now be cap=40, but the `rvlm` headline (n=8) was cap=20. The
`direct_vlm < rvlm by 5+pp` / `≈ rvlm` rules below assume equal
iteration budget — interpret the direct_vlm-vs-rvlm comparison only
after confirming the cap difference doesn't drive it (an rvlm cap=40
spot-check may be needed).

## Done

### `[✓]` rvlm_hybrid w/ images_for_last_n=1 n=1 val (task #37) — 2026-05-30

Tested the "visual window eviction" hypothesis for hybrid's −8pp deficit.
Overrode `solver.images_for_last_n=1` (vs yaml default 3). Hypothesis
was that hybrid's deficit might come from stale-image confusion across
multi-turn display() calls (tighter window helps) OR from helpful
multi-image context being too small (tighter window hurts).

**Result: imgN1 = 20.00% (16/80) — clean 25/25.**
- vs hybrid baseline (images_for_last_n=3): **−15.00pp**
- vs minimal: **−22.50pp**

The "multi-image context helps" branch, much more extreme than
predicted. The display() strategy heavily depends on accumulating
visual context across turns; collapsing the window to 1 cripples it.

Tool-usage ratio essentially unchanged:
- baseline hybrid (n=3): display 1397 / ask_vlm 706 (66:34)
- imgN1: display 1486 / ask_vlm 706 (68:32)
- ask_vlm count is *literally identical* (706:706)
- agent didn't adapt strategy to the smaller window — kept using
  display() at the same rate, just with broken context retention.

**Implication for the paper.** This is direct evidence that
display()-based perception is fragile to context-window choice in a
way that ask_vlm() (recursive sub-VLM) is not — sub-VLM text answers
persist in the trajectory regardless of how many later display()
calls evict images. The forced-delegation design of the rvlm family
sidesteps a real failure mode of multimodal-LM-in-the-loop solvers.

Open next-cell options if we want to keep digging:
- images_for_last_n=8 (or 12): does pushing the window up recover
  hybrid? If yes, locks the "display needs lots of context" story.
- per-question correlation: are hybrid's wrongs concentrated on docs
  where the agent did many display() calls (cumulative eviction)?

### `[✓]` rvlm_skeletal + rvlm_hybrid n=2 val (task #36) — 2026-05-29

n=2 follow-up to confirm n=1 reads. Skeletal-t2 lost 4 docs to
long-tail timeout (refill pending); hybrid-t2 was clean 25/25.

**Paired Δ on common 21-doc / 68-Q subset (clean 4-trial compare):**

| trial | score | Δ vs minimal-t* |
|---|---|---|
| minimal-t1 | 44.12% | — |
| minimal-t2 | 42.65% | — |
| skeletal-t1 | 41.18% | **−2.94pp** |
| skeletal-t2 | 39.71% | **−2.94pp** |
| hybrid-t1 | 35.29% | **−8.82pp** |
| hybrid-t2 | 35.29% | **−7.35pp** |

**n=2 paired mean Δ (common 68-Q):**
- **skeletal − minimal = −2.94pp** (eerily consistent across both trials)
- **hybrid − minimal = −8.09pp** (also tight: −8.82, −7.35)

**Updated reads (revising n=1):**
- **skeletal ≈ minimal but slightly below** (−2.94pp paired n=2 on
  common subset). The n=1 full-set "tie" was on different doc sets
  per trial. Clean paired comparison shows skeletal is consistently
  ~3pp below minimal — close to the pre-set ±2pp threshold but not
  cleanly inside it. n=8 needed to call this a tie vs a small cost.
- **hybrid ~8pp below minimal at n=2.** Both trials agree the drop
  is real (not n=1 noise). The "agent prefers display, scores
  lower" finding is confirmed. n=8 not needed to call hybrid worse;
  the question for the paper is *why*.

**Hybrid identical-score curiosity.** hybrid-t1 and hybrid-t2 both
scored 24/68 on the common subset. With temperature=1.0 across 4
parallel sub-VLMs, exact agreement is suspicious but plausible at
small denominators. Worth a per-question check if we publish this.

### `[✓]` rvlm_skeletal + rvlm_naked + rvlm_hybrid n=1 val (tasks #32 #33 #35) — 2026-05-29

Strip-chain n=1, Qwen 3.5 27B local, c=32. All three refilled to clean
25/25 docs / 80 questions. Reference: rvlm_minimal n=8 mean **42.03%
SD 2.21pp** (t1 = 42.50%).

**Locked full-set numbers (80 Q each):**

| Solver | n=1 score | vs minimal-t1 | vs minimal n=8 mean |
|---|---|---|---|
| minimal-t1 | 42.50% (34/80) | — | within mean |
| **skeletal-t1** | **42.50% (34/80)** | **+0.00pp** | within mean |
| naked-t1 | 32.50% (26/80) | **−10.00pp** | **−9.53pp (~4.3σ)** |
| hybrid-t1 | 35.00% (28/80) | **−7.50pp** | **−7.03pp (~3.2σ)** |

**Reads (n=1):**
1. **skeletal ≡ minimal at n=1** (literal 34/80 tie). The 3 doc-shape
   patterns (high-density, many-page, counting) don't carry the
   method. Strong promote-to-default candidate. Run n=2 → n=8 to
   confirm at the headline level.
2. **naked drops 10pp** — well outside minimal's σ (≈4.3σ at the
   point estimate). Removing the APPROACH steps + the verify-under-
   VLM-stochasticity principle costs real points. **APPROACH +
   verify are load-bearing.** Naked is a step too far; don't escalate.
3. **hybrid drops 7.5pp** — also outside minimal's σ. Adding a second
   perception channel (display) on top of ask_vlm hurts in n=1.
   Worth n=2 to confirm the magnitude.

**Tool-preference insight from hybrid (paper-worthy).** Counted
code-block tool calls across the full hybrid trial:
- `display(...)`: **1397 calls**
- `ask_vlm(...)`: **706 calls**

Agent strongly preferred `display()` over delegation (~2:1).
*Given the choice, the agent picked direct perception. And the score
dropped 7.5pp.* This is direct evidence that the rvlm family's
forced-delegation pattern is the right one — the agent's revealed
preference points toward seeing-itself, but seeing-itself produces
worse answers. A nice find for the paper's discussion of why
recursive perception > direct perception when the LM is identical.

**Operational note.** The new `rvlm_overlap_orch.py` pattern kept
the GPU warm across the n=1 chain (no idle on long-tails). Refill
needed for skeletal (4 docs) + naked (1) — sequential pass at full
14400s timeout. Hybrid was already 25/25 (no long-tail; possibly
because MultimodalRLM's inline-image channel sidesteps the
agent-loop hang mode that catches science_paper_1 in the rvlm
family).

### `[✓]` rvlm_minimal n=8 val — generality test (task #31) — 2026-05-29

run_ids: `rvlm-minimal-val-t{1..8}` · **n=8 mean 42.03% ± 2.21pp**
(range 38.75–45.00) on Qwen 3.5 27B local, c=32.

Paired vs `rvlm_unified` t1..t8: **Δ = +1.09pp [CI95: −3.14, +5.33]**,
paired t = 0.611, df = 7 (n.s.). Lands cleanly in the pre-set
"≈ 0pp / within paired noise" band → **`rvlm_minimal` is the proposed
method.** The 10.7 kB of hand-crafted DocVQA-2026 category tips and
the engineering-drawing-specific VLM sub-call signature are not
load-bearing; the recursive-perception mechanism is.

Variance note worth surfacing for the paper: minimal σ = 2.21pp,
unified σ = 4.05pp — almost 2× tighter trial-to-trial. Plausibly the
agent stops being yanked between competing per-category prescriptions
when the dispatch guess is wrong. Per-trial table + paired analysis:
`docs/experiments/rvlm-minimal-generality.md`.

Refill notes: 4/8 trials needed pass-2 refill — `science_paper_1`
systematically hits the 4h task_timeout (~50% per-attempt success
rate, agent-loop / vllm-hang specific to this doc). All cleared on
the sequential refill pass.

### `[✓]` unified-tips n=8 val (tasks #25 + #28) — 2026-05-28

run_ids: `rvlm-unified-val-t{1..8}` · **n=8 mean 40.94% ± 4.05pp**
(range 35.0–47.5) on Qwen 3.5 27B local, c=32.

Paired vs amax1's `rvlm` baseline at t1..t7: Δ mean = +0.71pp,
SE 1.72pp, 95% CI [−3.50, +4.93]pp — well inside noise. Per-trial
table + paired analysis: `docs/experiments/unified-category-tips-ablation.md`.

Variance asymmetry worth flagging for the promote-to-default
decision: unified σ=4.05pp vs rvlm σ=2.38pp (~1.7×). Final paired
analysis lands when amax1's t8 commits.

## Decision rules (set in advance)

- **unified-tips Δ ≈ 0pp** → promote unified to default; replace rvlm
  cells with rvlm_unified in subsequent cells.
- **rvlm_ocr ≈ rvlm on val** → OCR neutral on moderate-length docs;
  paper's §B doc-length-axis claim holds. Push to MMLongBench-Doc next.
- **direct_vlm < rvlm by 5+pp** → recursive sub-call is load-bearing;
  paper §C prediction 3 supported.
- **direct_vlm ≈ rvlm** → architecture-agnostic; reframing needed
  (the sub-call may not be the load-bearing piece — context-rationing
  is, regardless of architecture).
