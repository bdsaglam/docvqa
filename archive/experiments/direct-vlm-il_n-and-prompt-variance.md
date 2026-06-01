# direct_vlm — image-context (il_n) sweep, crash fix, and the prompt/variance investigation

**Status: parked 2026-05-31** (may revisit). Headline: `direct_vlm`
il_n=3 ≈ **35% ± ~5pp (non-comics, Qwen 3.5 27B val)**, high trial
variance. The OCR-free RLM (`rvlm_minimal`) remains the proposed method;
`direct_vlm` is alt-architecture evidence (paper §C prediction 3:
recursive sub-call vs direct in-context perception).

## What direct_vlm is

A single multimodal LM that `display()`s page crops **into its own
context** and reasons over them inline (vs the rvlm family, which
delegates perception to a VLM sub-call via `batch_look`/`ask_vlm` and
only sees the sub-call's *text* answer). The agent's images accumulate
in its trajectory across steps.

## 1. The `images_for_last_n` (il_n) sweep

`images_for_last_n=N` keeps **all step TEXT** but renders images only
for the last N steps (older steps keep the `[Image k]` placeholder, drop
the image data). Sweep at cap40, Qwen 27B val n=1:

| il_n | crash on comics? | accuracy |
|---|---|---|
| 1 | no | 23.7% (18/76) |
| 2 | no | 28.9% (22/76) |
| 3 | **YES** (>64 images → process killed) | 43.2% (32/74, comics excl.) |

- **Accuracy rises with retained visual context** (1<2<3). Accumulating
  images across steps is the mechanism direct_vlm relies on; collapsing
  the window to 1 costs ~14–20pp.
- **The best-accuracy config (il_n=3) is exactly the one that crashes.**

## 2. The >64-image crash and the fix

vllm is configured `--limit-mm-per-prompt {"image":64}`. Long comics
docs accumulate >64 displayed images → `BadRequestError: At most 64
images` → originally **killed the whole process** (lost the doc,
sometimes the run). The current build **catches it per-question**: logs
a WARNING, returns `Unknown` for that one question, continues. So at
il_n=3 the run now completes (image-heavy comics questions degrade to
`Unknown` instead of crashing). Crash safety is no longer the blocker;
the open question was purely accuracy.

## 3. Attempts that did NOT work (recorded so we don't repeat them)

- **Prompt-driven compaction.** Exposed a real `RESET_HISTORY(summary)`
  sandbox builtin + strong prompting ("compact OFTEN"). The agent
  (Qwen) **never called it** (0 calls, 2 independent confirmations).
  Compaction must be code-enforced if wanted, not prompted.
- **Strict "1–2 images/step" discipline.** Made the agent ~5× slower
  (nibbled to the iteration cap every question). Abandoned.
- **`max_messages` sliding window** (render only last N steps, text+
  images). Worse than il_n because it drops old **TEXT** too: mm8=33.3%,
  mm24=35.0%, mm40 too slow to finish — all below il_n=3. A
  non-compacting agent loses its accumulated notes. The window is the
  wrong knob.

## 4. The prompt-vs-build investigation (and the methodology lesson)

A new-prompt build (relaxed "look-then-note / sliding-window / compact"
language) scored il_n=3 = **27.6%** vs the old build's **43.2%** — a
~15pp apparent regression. The investigation whipsawed three times:

1. "It's the prompt." A diff showed the only solver-code difference was
   the prompt (rendering/params/model all identical). Plausible — the
   new prompt's "context is a sliding window / compact often" language
   is *false* for an unbounded-window config and coaches the agent away
   from il_n=3's accumulate-images strength.
2. "No, it's a tie." A byte-exact **legacy-prompt** rerun (`legacy_prompt`
   flag) recovered only ~+5pp, tied with new-prompt on early docs.
3. The resolution: **run it 3+ times.** Current-build il_n=3 n=3 (legacy
   prompt) = **34.3 / 31.4 / 40.0% non-comics** → wide ~9pp spread. Then
   a **faithful old-build rerun** (worktree @ f737190, old code + old
   deps) scored **35.7% non-comics — it did NOT reproduce its own
   43.2%.**

**Pooled across both builds, 6 trials, non-comics 70Q:
[28.6, 31.4, 34.3, 35.7, 40.0, 42.9] → 35.5% ± 5.3pp.** The old build's
own two trials span 42.9→35.7 (7pp). So **the 43.2% was the high tail of
a high-variance distribution, not a regression.** Old-build mean 39.3%
vs current 33.6% = 1.3σ, not significant.

**Lesson (the real takeaway):** `direct_vlm` il_n=3 has ~5pp trial SD —
*higher* than the project's usual ~3pp. Every verdict read off 1–2
trials ("prompt is the lever," "it's a regression," "it's luck") was an
artifact of that variance. This is exactly what the "run 3+ trials
before claiming a win" rule exists to prevent.

## Operational notes

- **`legacy_prompt: bool` flag** (`direct_vlm.yaml`, default false)
  selects a byte-exact pre-compaction prompt (no RESET_HISTORY / no
  sliding-window / no compact language). It finished crash-free 25/25.
  Either prompt is a fine ~35% default; the prompt is not a significant
  lever at this variance.
- **`images_for_last_n` knob** is wired through
  `VisualREPLHistory(max_messages, images_for_last_n, include_images)` —
  decouples the text window (`max_messages`) from the image window
  (`images_for_last_n`). Default `images_for_last_n=10_000` (= keep
  images for all rendered steps; unchanged behavior for other solvers).
- These direct_vlm runs occasionally **die abruptly** mid-run (no
  traceback, RAM/vllm fine) — relaunch resumable; runner picks up from
  completed docs.

## If revisited

- Get a clean **n≥5** mean for direct_vlm il_n=3 before any comparison.
- For the paper's §C (recursive vs direct), compare `direct_vlm` to
  `rvlm_minimal` at **equal cap and n≥5**, on matched docs — the ~5pp
  variance means small-n single-number comparisons are not trustworthy.
- A code-enforced **total-image cap keeping all text** (most-recent ~56
  across steps) remains the untested "best of both" — full reasoning
  context + hard crash safety — if direct_vlm accuracy ever needs to
  improve without the il_n image-truncation cost.

## Appendix — per-cell run log (migrated from coordination/amax1.md)

Operational detail for the cells behind the findings above. All
Qwen 3.5 27B, val (25 docs / 80 Q), n=1 each unless noted.

**Iteration-budget effect (cap20 vs cap40), `direct_vlm` minimal prompt:**
- `direct-vlm-minimal-val-t1` cap20 = 27.5% (22/80); **62/80 (78%) hit
  the 20/20 cap.**
- `direct-vlm-minimal-val-iter40-t1` cap40 = 21.6% (16/74; comics_2/4
  timed out); **58/74 still hit 40/40.**
- Verdict: doubling the cap did NOT help — the agent doesn't converge at
  either budget, it wanders to whatever ceiling it's given. The low
  score is solver behavior / task difficulty, not cap truncation.

**il_n sweep (`direct_vlm` cap40):** il_n=3 (`direct-vlm-val-iter40-t1`)
= 43.2% (32/74, crashes on comics >64 imgs); il_n=2
(`direct-vlm-iln2-val-iter40-t1`) = 28.9% (22/76); il_n=1
(`direct-vlm-iln1-val-iter40-t1`) = 23.7% (18/76). Accuracy monotonic in
retained images; il_n=1 collapses on maps/comics (the multi-region
categories that need several crops held at once). Clean R4/R5 same-76q
pair: il_n=2 beats il_n=1 by +5.2pp.

**max_messages window sweep (`direct_vlm` cap40, new-prompt build):**
mm8 = 33.3%, mm24 (`dvm-mm24-val-t1`) = 35.0% (finishes ~3h21m,
BadReq64=8 all caught per-Q), mm40 (`dvm-mm40-val-t1`) = DIED (too
slow — 2/25 docs in 63 min, ~4.5× slower than mm24; do not relaunch).
Going 8→24 buys only +1.7pp, still −8.2pp below il_n=3, at ~1.5–2× wall
cost. The window is the wrong knob — it drops old TEXT, not just images.

**prompt-vs-build (il_n=3, regression investigation):** 6 trials pooled
across both builds, non-comics 70Q = [28.6, 31.4, 34.3, 35.7, 40.0,
42.9] → 35.5% ± 5.3pp. Old-build rerun (`oldbuild-iln3-val-t1`, worktree
@ f737190, faithful old code + deps) = 35.7% nc — did NOT reproduce its
own 43.2%. Full reasoning in §4 above.
