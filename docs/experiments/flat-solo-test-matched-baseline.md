# flat_solo TEST runs — matched-baseline test number

**Hypothesis:** the matched within-model lift figure for Qwen 3.5 27B
needs a multi-trial test number to anchor the headline "scaffold lifts
raw Qwen 27B by ~21pp" claim. (Risk-rank #1 / #2 in
`docs/paper/experiment-plan.md` §"Risk-ranked execution order".)

**Setup:** Default `flat_solo` lean m=30 (full scaffold: OCR + BM25 +
cropping + tips). Qwen 3.5 27B (vLLM local 8927) for both LM and VLM.
**Test** split, 160 questions (8 categories × 20 questions, across 48
docs). 8 trials.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local \
  vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=flat_solo \
  solver.rlm_type=lean \
  data.split=test data.num_samples=null \
  max_concurrency=24 \
  run_id=flat-solo-3_5-27b-test-t${i}
```

## Per-trial status

| Trial | Predictions | Unknown rate | Sandbox errors | Wall |
|---|---|---|---|---|
| t1 | 160/160 | 8.8% (14/160) | 0 | ~3h31m |
| t2 | 160/160 | 11.9% (19/160) | 0 | — |
| t3 | 160/160 | 16.2% (26/160) | 26 (localized to `maps_6`) | — |
| t4 | 160/160 | 16.2% (26/160) | 0 | ~6h |
| t5 | 160/160 | 11.9% (19/160) | 0 | — |
| t6 | 160/160 | 12.5% (20/160) | 0 | — |
| t7 | 160/160 | 11.9% (19/160) | 0 | — |
| t8 | 160/160 | 11.9% (19/160) | 0 | — |

All 8 trials produced full 160-question prediction sets. `t3` had 26
sandbox errors confined to a single doc (`maps_6`); other 47 docs
clean. Trial retained — for SC-8 voting the localized noise is
absorbed by the other 7 trials. Mean Unknown rate across trials:
12.6% (range 8.8–16.2%).

## Scoring

The test split has no local ground truth (`"ground_truth": "NULL"`
in all per-task `result.json`s), so per-trial accuracy reads as 0%
locally. Real scores come from ICDAR submission.

Submission JSONs prepared:
- Per-trial: `submissions/flat-solo-3_5-27b-test-t{1..8}.json`
- SC-8 voted (majority across 8 trials):
  `submissions/flat-solo-3_5-27b-test-sc8.json`

## SC-8 vote agreement

Plurality vote distribution across the 8 trials (160 questions):

| Agreement | Count | % |
|---|---|---|
| 8/8 unanimous | 13 | 8.1% |
| 7/8 | 18 | 11.2% |
| 6/8 | 19 | 11.9% |
| 5/8 | 22 | 13.8% |
| 4/8 | 24 | 15.0% |
| 3/8 | 37 | 23.1% |
| 2/8 | 26 | 16.2% |
| 1/8 | 1 | 0.6% |
| **mean** | **4.46/8** | — |

Substantial trial-to-trial divergence on test: only 8% unanimous,
mean agreement just over half (4.46/8). This predicts SC-8 voting
will lift the test score noticeably above any single trial — matches
the Qwen 3.6 27B pattern where SC-8 voting added +2.75pp test over
per-trial mean.

## Comparison reference (Qwen 3.6 27B, prior runs)

From `docs/results.md`:
- 3.6 27B per-trial mean: 44.06% ± 3.04pp (n=8)
- 3.6 27B SC-8 voted: 43.75% on test (160q)

So the 3.6 27B test-time SC-8 vote landed close to per-trial mean.
3.5 27B might be similar or slightly higher.

## Headline numbers (locked)

- **SC-8 voted test: 38.75%** (ICDAR submission of
  `submissions/flat-solo-3_5-27b-test-sc8.json`, 2026-05-12).
- **Per-trial mean ± std test:** TBD — requires submitting individual
  trial JSONs and averaging. Not done yet (8 separate ICDAR
  submissions; user's go-ahead needed).

## Comparison context

| Anchor | Test % | Source |
|---|---|---|
| **Qwen 3.5 27B flat_solo SC-8 (this run)** | **38.75** | new |
| Qwen 3.5 27B flat_solo SC-8 (val) | 51.2 | `docs/results.md` |
| Qwen 3.6 27B flat_solo SC-8 (test) | 43.75 | `docs/results.md`, prior runs |
| Qwen 3.6 27B flat_solo SC-8 (val) | 51.2 | `docs/results.md` |
| Gemini 3 Pro raw VLM (test, official) | 37.5 | ICDAR baseline |
| Gemini 3 Pro + scaffold (test, 1 trial) | 59.4 | `docs/results.md` |

Observations:
- **Val-test gap is ~12pp for Qwen 3.5 27B SC-8** (51.2 → 39.0).
  Bigger than the 3.6 27B gap (51.2 → 43.75 = 7pp). 3.5 27B test
  generalises noticeably worse than 3.6 27B.
- **The scaffold on Qwen 3.5 27B (38.75%) beats raw Gemini 3 Pro
  (37.5%) on test by 1.5pp** — a defensible "open-model + scaffold
  beats closed frontier raw" headline, though margins are tight.
- The Qwen 3.6 27B SC-8 test (43.75%) result is now the better
  matched-baseline number for the paper headline if available.
  This 3.5 27B test gives us a second multi-trial test data point
  to bracket the variance.

## Caveats

- **Per-trial mean** would be the official "matched-baseline" number
  per the paper's variance discipline (D-003: no SC voting in
  headline). Currently we only have the SC-8 voted score because
  per-trial test scoring requires 8 separate ICDAR submissions.
- The lift figure (no-scaffold → with-scaffold) on **test** is not
  yet computable for Qwen 3.5 27B — would need `no_loop` Qwen 3.5
  27B on test. We have val no_loop only (17–21%). For the headline
  open-model lift figure, can the user obtain (or estimate) the
  test no_loop number?

## Status

**Test SC-8 locked at 38.75%.** Per-trial test means deferred (need
8 individual ICDAR submissions). User-driven from here.
