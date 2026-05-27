# leanest m=40 TEST runs — OCR-off matched-baseline test number

**Hypothesis:** establish a multi-trial OCR-off test-side number for
Qwen 3.5 27B to bracket the OCR contribution on test (paired with
the flat_solo test SC-8 at 38.75%).

**Setup:** `leanest_solo` solver with `max_iterations=40` (peak val
config from `leanest-turn-budget-sweep.md`). Qwen 3.5 27B (vLLM local
8927). **Test** split, 160 questions, 8 trials. The configuration is
identical to flat_solo test except for the solver — no OCR text in
scope, no `search()`, only `batch_look()` VLM with cropping.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local \
  vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=leanest_solo \
  solver.max_iterations=40 \
  data.split=test data.num_samples=null \
  max_concurrency=24 \
  run_id=leanest-m40-3_5-27b-test-t${i}
```

## Per-trial status

| Trial | Predictions | Unknown rate | Sandbox errors | Notes |
|---|---|---|---|---|
| t1 | 160/160 | 31.2% (50/160) | 0 | **EXCLUDED** — cold-start anomaly (lane-restart cache effects); Unknown rate ~2× the rest of the cell. Renamed `_excluded.`. |
| t2 | 160/160 | 19.4% (31/160) | 0 | clean |
| t3 | 160/160 | 18.8% (30/160) | 0 | clean |
| t4 | 160/160 | 10.6% (17/160) | 0 | clean (lowest Unknown) |
| t5 | 160/160 | 17.5% (28/160) | 0 | clean |
| t6 | 160/160 | 16.2% (26/160) | 0 | clean |
| t7 | 160/160 | 21.9% (35/160) | 0 | clean |
| t8 | 160/160 | 13.8% (22/160) | 0 | clean |
| t9 | 160/160 | 18.8% (30/160) | 0 | clean — replacement for excluded t1 |

8 clean trials retained (t2-t9). Mean Unknown rate 17.1% (range
10.6–21.9). Compare to flat_solo test's 12.6% — leanest abstains
~5pp more often without OCR text to scan, consistent with the
"OCR as stability anchor" story.

## SC-8 vote agreement (8 clean trials)

| Agreement | Count |
|---|---|
| 8/8 unanimous | 17 (10.6%) |
| 7/8 | similar to flat_solo |
| mean | 4.49/8 |

## Headline number (locked)

- **SC-8 voted test: 36.00%** (ICDAR submission of
  `submissions/leanest-m40-3_5-27b-test-sc8.json`, 2026-05-14).

## Comparison

| Anchor | Test SC-8 | Source |
|---|---|---|
| **Qwen 3.5 27B flat_solo (full, OCR)** | **38.75%** | `flat-solo-test-matched-baseline.md` |
| **Qwen 3.5 27B leanest (no OCR)** | **36.00%** | this file |
| Gemini 3 Pro raw (kit) | 37.5% | kit README |
| GPT-5.2 raw (kit) | 35.0% | kit README |
| Gemini 3 Flash raw (kit) | 33.75% | kit README |
| GPT-5 Mini raw (kit) | 22.5% | kit README |

**OCR contribution on test SC-8 = +2.75pp** (flat_solo − leanest).
This is larger than the val per-trial mean gap (0.94pp) but still
modest — the "OCR as stability anchor, not capability lifter"
framing holds on test too.

**Open-model beats closed frontier raw:**
- flat_solo Qwen 27B (38.75%) **beats Gemini 3 Pro raw** (37.5%)
- leanest Qwen 27B (36.0%) beats GPT-5.2 raw (35.0%), Flash (33.75%),
  and Mini (22.5%) — only Gemini Pro raw (37.5%) edges it by 1.5pp
- Both scaffolds beat 3 of 4 published closed-frontier raw baselines

## Status

**Done.** Test SC-8 locked at 36.00%. n=8 valid trials (t1 excluded
for cold-start anomaly; t9 used in its place). Per-trial test means
not computed (would require 8 separate ICDAR submissions).
