# RVLM baseline — direct image manipulation, no agent sub-loop

## Paper angle

The other Qwen 3.5 27B configs we run sit at two ends of a spectrum:

- **No-call baselines** (`no_loop`, `no_loop_multi`): one VLM call,
  no tools, no scaffold. Val SC-8 20.0% / test SC-8 11.0% on
  `no_loop_multi` (see `docs/experiments/split-calibration-no-loop-multi.md`).
- **Agent-loop solvers** (`flat_solo`, `leanest_solo`): a planner LM
  that delegates to a separate VLM sub-agent via `look()` / `batch_look()`
  tool calls. Val SC-8 45–48.8% / test SC-8 37–39% post-scrub
  (`docs/experiments/scrub-audit.md`).

**RVLM** is a third, distinct angle. The model is itself multimodal
and manipulates page images directly inside a Python REPL — `display(pages[i])`,
`display(pages[i].crop((l,t,r,b)))` — and sees the resulting images
inline in the next step. **No VLM tool call, no sub-agent.** The single
model handles both the reasoning and the perception, with image
operations as cheap Python primitives.

The paper question this isolates: **does direct image manipulation by
a single multimodal model close the gap to the agent-loop solvers, or
is the planner/VLM split itself doing most of the work?**

Anchors (Qwen 3.5 27B, local 8927, lean+nothink):

| Config | Val SC-8 | Test SC-8 | Notes |
|---|---|---|---|
| `no_loop_multi` (no tools) | 20.0% | 11.0% | first-N-page truncation baseline |
| `flat_solo` (planner + VLM sub-agent + BM25) | 45.0% (v1) | 37.0% (v1) | post-scrub |
| `leanest_solo` (planner + `batch_look` only) | 48.8% (v1) | 39.0% (v1) | post-scrub |
| **`rvlm` (single multimodal model + REPL)** | **TBD SC-8** | **TBD SC-8** | this experiment |

The split-difficulty floor is **~9pp** val→test
(`docs/experiments/split-calibration-no-loop-multi.md`); any RVLM
val→test gap must be read against that floor.

## Setup

- **Solver:** `rvlm` (`configs/solver/rvlm.yaml`).
  - `max_iterations=20`
  - `images_for_last_n=3` (last 3 displayed images carried in context)
  - `max_image_pixels=8_000_000` (per displayed image cap)
  - `use_category_tips=true` — uses the val-leak-scrubbed
    `RVLM_CATEGORY_TIPS` in `src/docvqa/solvers/rvlm_solver.py`
    (scrubbed per `docs/experiments/scrub-audit.md` v1 standard:
    no verbatim val-question phrasings, no val-doc entity names,
    no example letters/numbers tied to val items).
  - `question_concurrency=4` (intra-doc question parallelism).
- **LM/VLM:** `qwen-3_5-27b-vllm-local` (Host A, port 8927, 3×A100),
  `enable_thinking=false`. RVLM uses the same model for reasoning and
  perception — the `vlm` slot in the config is unused.
- **Concurrency:** `max_concurrency=16` (across docs).
- **Trial count:** n=8, SC-8 vote standard for this project.
- **Val:** 80 questions / 25 docs, locally ANLS-scored.
- **Test:** 160 questions / 48 docs, ICDAR-scored.

## Per-trial scores

### Val (n=8, locally scored)

| Trial | run_id | Score | Correct | Wall |
|---|---|---|---|---|
| t1 | `rvlm-val-t1` | TBD | /80 | TBD |
| t2 | `rvlm-val-t2` | TBD | /80 | TBD |
| t3 | `rvlm-val-t3` | TBD | /80 | TBD |
| t4 | `rvlm-val-t4` | TBD | /80 | TBD |
| t5 | `rvlm-val-t5` | TBD | /80 | TBD |
| t6 | `rvlm-val-t6` | TBD | /80 | TBD |
| t7 | `rvlm-val-t7` | TBD | /80 | TBD |
| t8 | `rvlm-val-t8` | TBD | /80 | TBD |

**Per-trial mean: TBD ± TBD pp**
**Val SC-8 vote: TBD** — `submissions/rvlm-val-sc8.json`

### Test (n=8, ICDAR-scored)

| Trial | run_id | Wall |
|---|---|---|
| t1 | `rvlm-test-t1` | TBD |
| t2 | `rvlm-test-t2` | TBD |
| t3 | `rvlm-test-t3` | TBD |
| t4 | `rvlm-test-t4` | TBD |
| t5 | `rvlm-test-t5` | TBD |
| t6 | `rvlm-test-t6` | TBD |
| t7 | `rvlm-test-t7` | TBD |
| t8 | `rvlm-test-t8` | TBD |

**Test SC-8 submission: `submissions/rvlm-test-sc8.json`**
**Test SC-8 ICDAR result: TBD**

## Per-category — val SC-8

| Category | val SC-8 |
|---|---|
| business_report | /10 |
| comics | /10 |
| engineering_drawing | /10 |
| infographics | /10 |
| maps | /10 |
| science_paper | /10 |
| science_poster | /10 |
| slide | /10 |
| **Overall** | **/80** |

## SC-8 vote plan

After each per-trial run produces `output/runs/rvlm-<split>-t$i/predictions.json`,
vote with `scripts/vote_submissions.py`:

```bash
uv run python scripts/vote_submissions.py \
  --runs $(for i in 1 2 3 4 5 6 7 8; do echo output/runs/rvlm-val-t$i; done) \
  --output submissions/rvlm-val-sc8.json

uv run python scripts/vote_submissions.py \
  --runs $(for i in 1 2 3 4 5 6 7 8; do echo output/runs/rvlm-test-t$i; done) \
  --output submissions/rvlm-test-sc8.json
```

Per-trial test submissions are saved automatically by the runner
(`submissions/rvlm-test-t{1..8}.json`) for inspection if needed.

## Chain command (DO NOT LAUNCH until v2 flat_solo chain completes)

Host A's local Qwen 27B at port 8927 is currently running the v2
flat_solo chain at `max_concurrency=24`. Launching RVLM SC-8 against
the same server would contend and tank both. Wait until:

```bash
pgrep -f scrubv2  # returns empty
tmux ls | grep -i scrubv2  # no docvqa-evals:scrubv2-val window
```

Once the v2 chain has produced both `submissions/flat-solo-val-scrubv2-sc8.json`
and `submissions/flat-solo-test-scrubv2-sc8.json`, launch:

```bash
# Per-trial (val and test each ×8). Run in a fresh tmux session.
tmux new-session -d -s rvlm-chain -n chain
mkdir -p logs/rvlm-chain

for split in val test; do
  for i in 1 2 3 4 5 6 7 8; do
    uv run python evals.py \
      lm=qwen-3_5-27b-vllm-local \
      vlm=qwen-3_5-27b-vllm-local \
      lm.enable_thinking=false \
      solver=rvlm \
      data.split=$split data.num_samples=null \
      max_concurrency=16 \
      run_id=rvlm-$split-t$i 2>&1 \
      | tee logs/rvlm-chain/rvlm-$split-t$i.log
  done
done

# Vote both splits
uv run python scripts/vote_submissions.py \
  --runs $(for i in 1 2 3 4 5 6 7 8; do echo output/runs/rvlm-val-t$i; done) \
  --output submissions/rvlm-val-sc8.json

uv run python scripts/vote_submissions.py \
  --runs $(for i in 1 2 3 4 5 6 7 8; do echo output/runs/rvlm-test-t$i; done) \
  --output submissions/rvlm-test-sc8.json

# Submit submissions/rvlm-test-sc8.json to ICDAR.
```

### Wall-time estimate

RVLM has a REPL with image-crop / image-zoom operations but **no
external VLM call** (the model sees images natively). Expected wall
time per trial sits between the two reference points:

- `no_loop_multi` val trial: ~4 min (one big multi-image VLM call).
- `flat_solo` val trial: ~50 min (planner LM + many VLM sub-agent calls).

RVLM should be closer to the lean end — no VLM round-trip per image
display, just multimodal-LM context expansion. **Rough estimate:
val ~15–30 min/trial, test ~60–120 min/trial.** Total chain wall:
~3–4 h val + ~10–16 h test = ~13–20 h end-to-end. Notification cadence:
phase boundaries (val done → test done).

## Smoke test (sanity, not lift)

Before the SC-8 chain, a 2-doc smoke confirms the scrubbed solver
runs end-to-end:

```bash
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false solver=rvlm \
  data.split=val data.num_samples=2 max_concurrency=2 \
  run_id=rvlm-smoke
```

Smoke result (2026-05-19/20, against Host A 8927 while v2 flat_solo
chain was warming up):

- **Pass:** solver ran end-to-end, no crashes, no Python exceptions.
- **Docs sampled:** `comics_1` (36p, 1 question) and `comics_2`
  (52p, 4 questions) — both `comics`, the known-worst-case category.
- **Accuracy:** 0/5 = 0.0%. All 5 predictions: `Unknown`.
- **Reading:** Consistent with `no_loop_multi` raw-VLM baseline,
  which also scored 0/10 on `comics` val SC-8
  (`docs/experiments/split-calibration-no-loop-multi.md`). The smoke
  hit the worst possible 2-doc draw (long-doc anthology counting
  questions); a non-zero verdict requires the SC-8 chain across all
  25 val docs, not a 2-doc smoke.
- **Wall:** ~5–7 min per doc on Host A while v2 flat_solo
  `max_concurrency=24` was contending for vllm. Standalone wall will
  be lower.
- Per-question logs at `output/runs/rvlm-smoke/tasks/comics_{1,2}/summary.md`.

## Decision rules (set in advance)

| RVLM val SC-8 vs anchors | Reading |
|---|---|
| < no_loop_multi (20.0%) | Negative result — direct image ops without scaffold worse than truncated single-call. Worth writing up. |
| 20–35% | Direct manipulation provides some lift but the planner/VLM split is the dominant scaffold contributor. |
| 35–45% | Direct manipulation captures most of the agent-loop lift without a sub-agent. Strong paper angle. |
| > 45% (matches/beats leanest) | Single-model multimodal REPL is competitive with planner+VLM. Headline finding. |

Test SC-8 will sit ~9pp below val SC-8 by the split-difficulty floor.

## Status

**Tips scrubbed (v1 standard).** Solver smoke test pending.
SC-8 chain **NOT launched** — waiting for v2 flat_solo chain on Host A
8927 to complete first.

## Open questions

- Does `images_for_last_n=3` (carry 3 most recent displayed images) hit
  context limits on long docs (slides, business reports)? Per-trial
  wall time on long-doc trials will indicate this.
- Does RVLM's per-category mix differ from `flat_solo`? E.g., direct
  image manipulation may help `maps` (spatial reasoning over crops)
  more than `science_paper` (text-heavy, OCR-friendly).
- Compute cost comparison vs `flat_solo`: RVLM has no VLM round-trip
  but multimodal context tokens grow with each `display()` call. Net
  cost-per-question is a separate but reportable metric.
