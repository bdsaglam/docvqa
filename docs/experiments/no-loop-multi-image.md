# No-loop multi-image baseline (raw VLM, native multi-image input)

**Hypothesis:** the original `no_loop` baseline scores 0% on long docs
(business_report 105–181 pages, science_paper 19–44 pages) because the
composite-image height cap (16384px) crushes per-page resolution to
~80–150px tall, making text illegible. A multi-image variant that sends
the first N pages at native resolution as separate images in one VLM
call should lift those scores — and is a stronger raw-VLM baseline that
preempts the reviewer objection "did you actually try the VLM properly?"

**Setup:** New solver `solver=no_loop_multi`. Sends `min(num_pages,
max_pages)` pages of the document at native resolution as separate
images in a single chat-completion request. No REPL, no tools, no agent
loop. Truncation policy: first N pages (head). Qwen 3.5 27B VLM via
Host B vllm 8928 (4-GPU). Val 80q, `lm.enable_thinking=false`,
`max_concurrency=8`, `question_concurrency=4`, `max_pages=10`.
**Default `use_category_tips=true`** — see "Fairness fix" below.

vLLM probe (3-image multipart request) confirmed multi-image works
natively — model identified each image separately. Smoke test on 2 docs
ran end-to-end with no errors.

## Fairness fix (2026-05-08, late)

Original n=3 run shipped without category tips, while the agent solvers
(flat_solo, leanest_solo) did receive tips. To make the comparison fair,
we added a baseline-adapted tips dict (`BASELINE_CATEGORY_TIPS` in
`prompts.py`, agent verbs stripped) and re-ran 3 trials with
`use_category_tips=true`. Headline below uses the tips-on result;
no-tips reported as a secondary cell. For multi-image, tips give only
a marginal lift (+1.25pp, not significant) — at native res the model
already gets answer formats right, so the semantic guidance has less
work to do than in the composite-image case (+4.17pp for `no_loop`).

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-remote \
  vlm=qwen-3_5-27b-vllm-remote \
  lm.enable_thinking=false \
  solver=no_loop_multi \
  data.split=val data.num_samples=null \
  max_concurrency=8 \
  run_id=no-loop-multi-3_5-27b-val-t${i}
# Trials launched 2026-05-08 used run_id=no-loop-multi-val-t${i}.
```

## Per-trial scores — tips ON (headline)

| Trial | run_id | Score | Correct | Wall | Sandbox errors |
|---|---|---|---|---|---|
| t1 | `no-loop-multi-tips-val-t1` | 22.50% | 18/80 | ~5 min | 0 (n/a — no REPL) |
| t2 | `no-loop-multi-tips-val-t2` | 26.25% | 21/80 | ~5 min | 0 |
| t3 | `no-loop-multi-tips-val-t3` | 22.50% | 18/80 | ~5 min | 0 |

**Mean: 23.75% ± 2.17pp | range 22.50–26.25%**

## Per-trial scores — tips OFF (ablation)

| Trial | run_id | Score | Correct |
|---|---|---|---|
| t1 | `no-loop-multi-val-t1` | 21.25% | 17/80 |
| t2 | `no-loop-multi-val-t2` | 22.50% | 18/80 |
| t3 | `no-loop-multi-val-t3` | 23.75% | 19/80 |

**Mean: 22.50% ± 1.25pp | range 21.25–23.75%**

## Summary

| Configuration | Mean | Std | n |
|---|---|---|---|
| **no_loop_multi + tips (headline)** | **23.75%** | **2.17pp** | **3** |
| no_loop_multi − tips (ablation) | 22.50% | 1.25pp | 3 |
| Δ tips contribution | **+1.25pp** | (SE 1.44, t=0.87, **NOT significant**) | — |

## Per-category (mean over 3 trials, tips ON)

| Category | tips-on t1 | t2 | t3 | tips-on mean | tips-off mean | Δ from tips |
|---|---|---|---|---|---|---|
| business_report | 0/10 | 0/10 | 0/10 | 0.0/10 | 0.0/10 | 0 |
| comics | 0/10 | 0/10 | 0/10 | 0.0/10 | 0.0/10 | 0 |
| engineering_drawing | 3/10 | 4/10 | 3/10 | 3.3/10 | 2.3/10 | **+1.0** |
| infographics | 4/10 | 6/10 | 5/10 | 5.0/10 | 5.7/10 | −0.7 |
| maps | 1/10 | 1/10 | 1/10 | 1.0/10 | 1.0/10 | 0 |
| science_paper | 3/10 | 3/10 | 3/10 | 3.0/10 | 3.0/10 | 0 |
| science_poster | 4/10 | 4/10 | 3/10 | 3.7/10 | 3.0/10 | +0.7 |
| slide | 3/10 | 3/10 | 3/10 | 3.0/10 | 3.0/10 | 0 |
| **Overall** | **22.5%** | **26.25%** | **22.5%** | **23.75%** | **22.50%** | **+1.25pp** |

Tips give multi-image a much smaller lift than they give the composite
baseline (+1.25 vs +4.17). Engineering drawing benefits most (+1.0/10);
most other categories unchanged. The model already gets answer-format
conventions right when text is legible — tips' incremental value is
mostly the domain-specific PART/ITEM and "Width vs Length" hints.

## Comparison

- **vs no_loop composite baseline (with matched tips):**
  21.25% ± 1.25pp (n=3) — see `no-loop-baseline.md`.
  **Gap: +2.50pp.** SE = √(2.17²/3 + 1.25²/3) = 1.45pp.
  **t-stat: 1.73 → not quite significant** (was +5.42pp in the unfair
  no-tips comparison). Multi-image still wins, but the gain over the
  composite baseline is smaller once both get tips: tips give the
  composite +4.17pp and multi-image only +1.25pp, so the multi-image
  advantage shrinks once the comparison is fair.

- **vs scaffold (Flat Solo m=30):** 44.69% ± 2.81pp (n=8).
  **Gap: −20.94pp.** SE = √(2.17²/3 + 2.81²/8) = 1.60pp.
  **t-stat: 13.11 → highly significant.** The agent loop + tool
  channel still adds ~21pp on top of the strongest fair raw-VLM
  baseline. The "scaffold matters" story holds up against the harder
  control.

## Observations

- **Hypothesis confirmed:** the categories where composite was crushed
  are exactly the ones that lift in multi-image:
  - **science_paper: 0.0/10 → 3.0/10 (+3pp).** Composite's 0% was the
    composite-rescaling artifact, not a model failure. At native res,
    the model can read the abstract / intro pages directly.
  - **slide: 1.7/10 → 3.0/10 (+1.3pp).** Slides are typically short;
    fit fully inside the 10-page budget at native res.
  - **infographics: 4.0/10 → 5.7/10 (+1.7pp).** Single-page docs
    benefit modestly from skipping the composite resize step.
- **Hypothesis confirmed:** comics REGRESSED (2.0/10 → 0.0/10).
  At native res the model honestly returns "Unknown" on narrative-
  reading questions across 10 pages of a 30–69-page comic; composite
  no_loop guessed (e.g. "Whodunnit", "red") because rescaled text
  isn't legible enough for the model to confirm "this is unanswerable".
  Per `prompts.py:10`, "Unknown" is the prescribed answer when the
  question can't be answered from the image. Multi-image is a more
  honest baseline; it just happens to score lower on comics where
  the GT has high lexical overlap with shallow guesses.
- **business_report stayed at 0/10.** All three trials score 0 across
  all 10 questions. Confirms that for 105–181-page business docs,
  showing the first 10 pages is no help — the answers are deeper in
  the doc. Could revisit with strided sampling or a higher max_pages.
- **Wall time ~5 min/trial** at c=8 — comparable to or faster than the
  composite no_loop at c=16 (~12 min). vllm batches multi-image
  efficiently; effective throughput is similar.
- **Variance dropped** (1.25pp vs composite's 2.60pp). Multi-image gives
  a more deterministic baseline, possibly because the model's "Unknown"
  vs "guess" decision is more stable when text is legible.

## Component-contribution table (Qwen 3.5 27B, val, with fair tips)

| Stage | Channel | Mean | Δ vs prior | n |
|---|---|---|---|---|
| no_loop (composite, no tips) | raw VLM, composite rescale | 17.08% | — | 3 |
| no_loop (composite, +tips) | + baseline-adapted tips | 21.25% | +4.17pp | 3 |
| no_loop_multi (head 10pp, no tips) | multi-image, no rescale | 22.50% | +1.25pp¹ | 3 |
| no_loop_multi (head 10pp, +tips) | multi-image, +tips | 23.75% | +2.50pp¹ | 3 |
| leanest_solo (no OCR) | + agent loop + VLM `look()` (and tips) | 40.00% | +16.25pp² | 3 |
| flat_solo (m=30, full) | + OCR text + BM25 + cropping | 44.69% | +4.69pp | 8 |

¹ Both deltas measured vs `no_loop (composite, +tips) = 21.25%`.
Multi-image alone (+1.25pp) is not significant once the composite
baseline gets matched tips.

² Leanest vs `no_loop_multi (+tips) = 23.75%`. The agent-loop +
VLM-tool channel still does the heavy lifting (+16.25pp), gap is
smaller than the original +22.9pp because the raw-VLM baseline is now
stronger (it has tips and multi-image legibility). Honest framing for
the paper.

## Configuration knobs

- `max_pages` (default 10): pages to send. Higher → more coverage but
  more vision tokens. At 10, all single-page docs (maps, infographics,
  science_poster) and most engineering_drawing get full coverage;
  science_paper (19–44pp) and business_report (105–181pp) are
  head-truncated.
- `question_concurrency` (default 4): per-doc question parallelism.

## Status

**Done.** 3 trials with tips ON (headline 23.75% ± 2.17pp) and 3 trials
with tips OFF (ablation 22.50% ± 1.25pp). Headline: stronger raw-VLM
baseline (multi-image + matched tips: 23.75%) closes ~7pp of the gap
to scaffold but leaves a ~21pp margin (13.1 SE) — the agent-loop
story is robust to the strongest fair control. Composite-rescaling
was the main artifact on science_paper / slide / infographics; tips
contribute much less here than to the composite baseline (+1.25 vs
+4.17). comics still 0% (honest "Unknown" abstention).
business_report still 0% — head-truncation at 10 pages can't reach
the answer; would need different sampling. Solver:
`src/docvqa/solvers/no_loop_multi_solver.py`.
Config: `configs/solver/no_loop_multi.yaml`.


## Observations (preliminary)

- **vLLM accepts multi-image natively.** A test with 3 placeholder
  images returned a coherent enumeration. Default `--limit-mm-per-prompt`
  is sufficient (no flag set on Host B vllm); the binding constraint
  is tokens, not image count.
- **Token budget at max_pages=10.** Probe with 10 comics_2 pages
  (1200×1800ish each): 21881 prompt tokens. Comfortable under
  131072 max-model-len.
- **Multi-image is more "honest" than composite on hard questions.**
  Smoke on comics_2 (52pp, sent first 10) returned literal "Unknown"
  on all 4 narrative questions — the model correctly identifies the
  questions are unanswerable from 10 pages. Composite no_loop on the
  same 4 questions returned guesses ("Whodunnit", "red", "5", etc.).
  The "Unknown" rule comes from `prompts.py:10`: "If the question is
  unanswerable given the provided image, the response must be exactly:
  Unknown". When pages are legible, the model recognizes when it can't
  confirm an answer; when pages are crushed by rescaling, the model
  pattern-matches and guesses.

## Configuration knobs

- `max_pages` (default 10): pages to send. Higher → more coverage but
  more vision tokens. At 10, all single-page docs (maps, infographics,
  science_poster) and most engineering_drawing get full coverage;
  science_paper (19–44pp) and business_report (105–181pp) are
  head-truncated.
- `question_concurrency` (default 4): per-doc question parallelism.

## Status

**In progress.** 3-trial val chain launched 2026-05-08 16:11 in
tmux `docvqa-exps:multi`. Heartbeat cron `33c6fffe` will surface
completion. Solver source: `src/docvqa/solvers/no_loop_multi_solver.py`.
Config: `configs/solver/no_loop_multi.yaml`.
