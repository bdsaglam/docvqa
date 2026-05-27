# Flat Solo DA — multi-image VLM tool

**Status:** Shelved (single-trial regression; no clear category lift). Code retained for opt-in reuse.

**Hypothesis:** Sometimes it is easier to **show** the VLM something than to describe it in
English. A solver where `look()` accepts a list of images (legend + region, two crops to compare,
a few-shot example + target, a citation crop + its source figure, …) could let the agent solve
"matching" and "comparison" questions without translating visual properties into noisy English
prompts. Particularly hoped for: legend-symbol counting / road-type matching on **maps**, where
the existing single-image prompt's "crop X alongside the legend" pattern conflates two things
into one image and loses resolution.

**Setup:** New solver `flat_solo_da_mi` (sibling of `flat_solo_da`). VLM `dspy.Predict`
signature widened from `image: dspy.Image` to `images: list[dspy.Image]`. `look(images, query)`
and `batch_look(requests)` accept either a single image / page-index or a list — singletons
remain backward-compatible. Both cropping and page-only sandbox builders forked. Profile-driven
prompt formatting / per-category tips / per-question hint / scoring unchanged. No new ablation
knobs.

Implementation in `src/docvqa/solvers/flat_solo_da_mi_solver.py` (`a7864a8`, `1d19090`).
Hydra config in `configs/solver/flat_solo_da_mi.yaml`.

## Trials

All on Qwen 3.5 27B local vllm 8927, val/80q, `lm.enable_thinking=false`, `solver.rlm_type=lean`,
`max_concurrency=16`, `question_concurrency=4`. Compared against the same setup's flat_solo
baseline (5 prior trials: 32–35 correct, mean 33.8/80 = 42.3%, std ~1.4pp).

| Run | Prompt | Score | Multi-image / total `look()` |
|---|---|---|---|
| flat-solo-val-scrub-tN (5 trials) | flat_solo (no multi-image) | mean 42.3% (range 40.0–43.8) | — |
| **flat-solo-da-mi-val** (v1) | flat_solo_da_mi, neutral prompt | **35/80 = 43.8%** | 21 / 978 = 2.1% |
| **flat-solo-da-mi-val-v2** | v1 + SHOW-DON'T-DESCRIBE guideline + per-category overlays (maps: replacement; comics/infographics/science_paper: append) | **31/80 = 38.8%** | **114 / 1139 = 10.0%** |

```bash
# v1 (neutral prompt)
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=flat_solo_da_mi solver.rlm_type=lean \
  data.split=val data.num_samples=null max_concurrency=16 \
  run_id=flat-solo-da-mi-val

# v2 (after prompt push — commits 9058545, ec78cb3)
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=flat_solo_da_mi solver.rlm_type=lean \
  data.split=val data.num_samples=null max_concurrency=16 \
  run_id=flat-solo-da-mi-val-v2
```

## Per-category breakdown (v1 vs v2)

| Category | v1 acc | v2 acc | Δ | v2 multi% |
|---|---|---|---|---|
| business_report | 35% | 35% | 0 | 0% |
| comics | 27% | 27% | 0 | **35.8%** |
| engineering_drawing | 79% | 75% | −4 | 1.4% |
| infographics | 56% | 56% | 0 | 1.0% |
| maps | 7% | 7% | 0 | 2.8% |
| science_paper | 31% | 21% | **−10** | 1.6% |
| science_poster | 40% | 30% | **−10** | 2.8% |
| slide | 48% | 37% | **−11** | 0.8% |

## Findings

1. **Capability works end-to-end.** v1 already produced multi-image `look([…])` calls organically
   (e.g. `look([pages[0], pages[1], pages[2]], 'Describe the layout of each page in order.')`)
   in 5 of 25 docs. No infrastructure bugs.

2. **Prompt push (v2) moved adoption sharply, but only on comics.** Adding a SHOW-DON'T-DESCRIBE
   guideline + per-category worked examples lifted total multi-image usage 5.4× (21 → 114 calls,
   2.1% → 10.0% of all `look()` calls). The lift is almost entirely in **comics** (6.3% → 35.8%).
   The other six categories barely moved (0–3%).

3. **Maps adoption stayed flat in the full eval** despite the targeted replacement tip. The
   pre-eval smoke (`mi-adoption-smoke-v2`, 4 docs incl. maps_1/maps_2) showed maps multi-image
   2.75× higher, but on the full val (3 maps docs, 7 questions) usage was 6 / 218 calls = 2.8%
   (vs v1's 3.1%). The smoke happened to weight questions where the tip pattern-matched best;
   the full sample didn't.

4. **Headline accuracy regressed 5pp on a single trial (43.8% → 38.8%).** The 5pp drop is just
   outside the ~1.4pp std seen across baseline trials, but with n=1 each it is not conclusive.

5. **The regression is not caused by multi-image use.** The categories with the largest accuracy
   drops (slide −11, science_paper −10, science_poster −10) had **near-zero** multi-image
   adoption (0.8%, 1.6%, 2.8%). Their tool-choice mixes are essentially identical between v1 and
   v2. The most plausible mechanism for the regression in those categories is **prompt-length
   pressure**: v2's prompt is ~30% longer (longer `look()` doc + new guideline + per-category
   overlays), and the extra tokens may be a soft distractor even when they change no tool call.

6. **Even where adoption succeeded (comics), aggregate accuracy didn't move.** comics: 27% in
   both runs. Per-doc: comics_2 jumped 25% → 75% (with 3 → 34 multi-image calls), comics_4 fell
   50% → 0% (with 0 → 50 multi-image calls). Net zero — the new pattern wins on some questions
   and loses on others, with no aggregate signal.

## Decision

**Shelve, don't roll back.** The solver and config remain in the tree for opt-in reuse via
`solver=flat_solo_da_mi`, but it is not promoted as a default and not added to the headline
results table. v1 is the cleaner of the two prompts (neutral, no per-category overlay, the lift
from SHOW-DON'T-DESCRIBE is only present in comics and didn't change comics accuracy aggregate)
— if anyone resumes this line, start from the v1 prompt, not v2.

## Open questions for future work

- **Is the v2 regression real or noise?** Settle with 2 more v2 trials + 2 more baseline trials
  (≈6h on this hardware). Current n=1 vs n=1 is insufficient given trial std ≈ 1.4pp.
- **Can comics_4 / comics_2 swing be attributed to specific multi-image patterns?** comics_4
  went from 0 → 50 multi-image calls *and* 50% → 0% accuracy — worth a trajectory inspection
  before any further iteration.
- **Lean prompt variant** — keep multi-image capability and the SHOW-DON'T-DESCRIBE guideline
  but drop the per-category overlays (which seemed to add prompt-length pressure without
  changing tool mix in most categories). Cheapest information per token.
- **Is "show vs describe" the right framing for maps in this dataset?** The 3 val maps docs are
  heavy on path-following / spatial-reasoning questions ("walk from X, turn left, what's next")
  rather than clean legend-matching, so legend-crop + region-crop bites less often than expected.
  Maps from a more legend-heavy benchmark might give the tip a fairer chance.

## Files / runs

- Code: `src/docvqa/solvers/flat_solo_da_mi_solver.py`, `configs/solver/flat_solo_da_mi.yaml`.
- Spec: `docs/superpowers/specs/2026-05-21-flat-solo-da-multi-image-design.md`.
- Plan: `docs/superpowers/plans/2026-05-21-flat-solo-da-multi-image.md`.
- Runs: `output/runs/flat-solo-da-mi-val/`, `output/runs/flat-solo-da-mi-val-v2/`,
  smoke runs `mi-smoke-crop`, `mi-smoke-pageonly`, `mi-adoption-smoke`,
  `mi-adoption-smoke-v2`.
- Key commits: `1d19090` (solver), `a7864a8` (config), `9058545` (prompt push), `ec78cb3`
  (maps-tip replacement).
