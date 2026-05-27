# Split-calibration: no_loop_multi on val and test (Qwen 3.5 27B local)

Follow-up to `docs/experiments/scrub-audit.md` open question: is the
8–12pp val→test gap on scaffolded solvers a **generalization gap**
(prompts/scaffold overfit val) or a **split-difficulty gap** (test is
intrinsically harder)?

`no_loop_multi` is the cleanest baseline that runs on both splits
(unlike `official_baseline`, which OOM'd on test). If its val SC-8
and test SC-8 land within ~2pp, splits are calibrated.

## Setup

- **Solver:** `no_loop_multi`, `max_pages=10`, `use_category_tips=true`
  (default), `question_concurrency=4`.
- **VLM/LM:** `qwen-3_5-27b-vllm-local` on port 8927, `enable_thinking=false`.
- **vllm container:** restarted 2026-05-19 16:41 with explicit
  `--limit-mm-per-prompt '{"image":10}'` and `--tool-call-parser qwen3_coder`
  (see vllm setup notes below).
- **Concurrency:** `max_concurrency=16`.
- **Trial count:** n=8 each (SC-8 standard for this project).
- **Val anchor (prior, remote 8928, n=3):** 23.75% ± 2.17pp from
  `no-loop-multi-image.md`.

```bash
# Per-trial (val and test each ×8)
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=no_loop_multi solver.max_pages=10 \
  data.split=$split data.num_samples=null \
  max_concurrency=16 \
  run_id=no-loop-multi-3_5-27b-$split-t$i
```

## Per-trial scores

### Val (locally ANLS-scored, 80 questions / 25 docs)

| Trial | run_id | Score | Correct |
|---|---|---|---|
| t1 | `no-loop-multi-3_5-27b-val-t1` | 21.2% | 17/80 |
| t2 | `no-loop-multi-3_5-27b-val-t2` | 21.2% | 17/80 |
| t3 | `no-loop-multi-3_5-27b-val-t3` | 18.8% | 15/80 |
| t4 | `no-loop-multi-3_5-27b-val-t4` | 25.0% | 20/80 |
| t5 | `no-loop-multi-3_5-27b-val-t5` | 21.2% | 17/80 |
| t6 | `no-loop-multi-3_5-27b-val-t6` | 20.0% | 16/80 |
| t7 | `no-loop-multi-3_5-27b-val-t7` | 21.2% | 17/80 |
| t8 | `no-loop-multi-3_5-27b-val-t8` | 20.0% | 16/80 |

**Per-trial mean: 21.07% ± 1.81pp** (range 18.8–25.0%)

**Val SC-8 vote: 20.0%** (16/80) — `submissions/no-loop-multi-3_5-27b-val-sc8.json`

### Test (predictions saved; ICDAR scoring required, 160 questions / 48 docs)

| Trial | run_id | Wall |
|---|---|---|
| t1 | `no-loop-multi-3_5-27b-test-t1` | 24m |
| t2 | `no-loop-multi-3_5-27b-test-t2` | 23m |
| t3 | `no-loop-multi-3_5-27b-test-t3` | 22m |
| t4 | `no-loop-multi-3_5-27b-test-t4` | 22m |
| t5 | `no-loop-multi-3_5-27b-test-t5` | 28m |
| t6 | `no-loop-multi-3_5-27b-test-t6` | 23m |
| t7 | `no-loop-multi-3_5-27b-test-t7` | 24m |
| t8 | `no-loop-multi-3_5-27b-test-t8` | 24m |

**Test SC-8 submission: `submissions/no-loop-multi-3_5-27b-test-sc8.json`**
(160 questions, ~61KB).

**Test SC-8 ICDAR result (2026-05-19): 11.0%** (0.11 ANLS).

## Per-category — val SC-8

| Category | val SC-8 |
|---|---|
| business_report | 0/10 |
| comics | 0/10 |
| engineering_drawing | 3/10 |
| infographics | 3/10 |
| maps | 1/10 |
| science_paper | 3/10 |
| science_poster | 3/10 |
| slide | 3/10 |
| **Overall** | **16/80 = 20.0%** |

business_report and comics 0% as expected — both have many-page docs
where the first 10-page truncation can't reach the answer.

## Observation: SC-8 < per-trial mean on val

Val SC-8 (20.0%) is **1.07pp below** the per-trial mean (21.07%).
This is unusual — voting normally helps. Likely cause: "Unknown"
answers form a stable consensus across trials, overriding the
occasional lucky guess that happens to match GT in a single trial.
For `no_loop_multi`, "Unknown" abstentions are the right move when
the answer isn't on the first 10 pages, so voting reinforces the
honest-abstention path at the cost of agg score.

This matters for the audit's gap-narrowing claim: if the same pattern
holds on test (test SC-8 < test per-trial mean), the val→test
comparison stays internally consistent. If only val shrinks under
voting (and test gains), the calibration argument weakens slightly.

## Val→test gap — verdict

| Config | Val SC-8 | Test SC-8 | Gap | Notes |
|---|---|---|---|---|
| **`no_loop_multi`** (this experiment) | **20.0%** | **11.0%** | **9.0pp** | raw-VLM baseline, no scaffold |
| flat_solo SC-8 (pre-scrub) | 51.2% | 39.0% | 12.2pp | full scaffold |
| flat_solo SC-8 (v1 scrub) | 45.0% | 37.0% | 8.0pp | scrubbed scaffold |
| leanest_solo SC-8 (v1 scrub) | 48.8% | 39.0% | 9.8pp | scrubbed scaffold |

**Reading rules** (set in advance):
- ≤ 2pp → splits calibrated, audit gap-narrowing claim stands
- > 5pp → test intrinsically harder, audit claim weakens
- 2–5pp → mixed

**Result: 9.0pp >> 5pp → test is intrinsically harder than val** for
Qwen 3.5 27B at this baseline level. The audit's "4.2pp gap narrowing
on flat_solo" claim **needs requalification**.

### Revised interpretation of the scrub-audit numbers

A baseline-level split-difficulty gap of ~9pp means a significant
chunk of every scaffolded solver's val→test gap is "test is harder",
not "prompts overfit val." Decomposing:

| Solver config | Total gap | Split-difficulty floor¹ | TRUE generalization gap |
|---|---|---|---|
| flat_solo pre-scrub | 12.2pp | ~9pp | ~3pp |
| flat_solo v1 scrub | 8.0pp | ~9pp | ~−1pp |
| leanest_solo v1 scrub | 9.8pp | ~9pp | ~+1pp |

¹ Floor from `no_loop_multi` baseline. Treats baseline split-diff as
a lower bound on scaffolded solvers' split-diff (scaffolded models
may be even more split-asymmetric, e.g. if BM25 indices benefit val
more, in which case the true split-diff floor for the scaffold is
*higher* than 9pp and the true generalization gap is even smaller).

**Implications:**

1. The pre→post scrub gap narrowing (12.2 → 8.0pp, **−4.2pp**) on
   flat_solo is preserved as a measurable effect — but it's now
   better described as **"prompt scrub closed almost all of the
   remaining true generalization gap on flat_solo"** rather than
   "narrowed a generic 12pp val→test gap by 4pp."
2. Post-scrub flat_solo and leanest are effectively at the
   split-difficulty floor — there's little to no measurable
   *prompt-overfit* signal left in their gap.
3. The audit's qualitative conclusion (val-leak scrub helped) holds.
   The quantitative framing ("12 → 8pp gap") should now reference
   the baseline's 9pp floor to avoid implying that a 0pp gap was
   ever a reasonable target.

### Caveat — SC-8 reduced score on val

Val per-trial mean 21.07% vs val SC-8 20.0% (SC-8 lowered the score).
"Unknown" consensus on long-doc questions overrides occasional lucky
guesses. If the same effect holds on test, the 11.0% SC-8 already
includes that voting penalty; if test per-trial mean is higher than
test SC-8, the true split-difficulty gap is even larger than 9pp
(making the requalification stronger). Cannot verify without local
test GT.

## vllm setup notes

- Original 8927 container (running since May 7) wedged its API-server
  thread pool on a cascade of `PIL.Image.DecompressionBombError` 500s
  during the first val-t1 attempt. Root cause: leaked queued requests
  from earlier `official-test` window targeting `maps_5` page 0
  (246M pixels) — that page belongs to the test split, not val.
  val pages max at 14M pixels.
- Restarted with (the user's previously-working command):
  ```bash
  docker run --rm --runtime nvidia --gpus all --name qwen35-27b \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env HF_TOKEN=... \
    -p 8927:8927 --ipc=host \
    vllm/vllm-openai \
    --port 8927 --model Qwen/Qwen3.5-27B \
    --data-parallel-size 3 --async-scheduling \
    --gpu-memory-utilization 0.85 --dtype bfloat16 \
    --max-model-len 131072 \
    --limit-mm-per-prompt '{"image":10}' \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
  ```
- Smoke tests post-restart: text-only OK; 3-page multi-image OK
  (correctly identified 3 distinct pages).
- Local 8927 per-trial mean (21.07% ± 1.81pp) is ~3pp below the prior
  remote-8928 anchor (23.75% ± 2.17pp). Likely vllm version /
  prefix-caching / processor differences; internally consistent for
  val-vs-test calibration done here.

## Status

**Done.** Val SC-8 = 20.0% (local). Test SC-8 = 11.0% (ICDAR,
2026-05-19). Val−test gap = 9.0pp → splits are NOT calibrated; test
is intrinsically harder than val for the raw-VLM baseline. Audit's
gap-narrowing claim revised above.

Wall time: ~3h18m for all 16 trials (test ~22-28 min each, val ~4 min each).
