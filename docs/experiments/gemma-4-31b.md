# Gemma-4 31B (homog) — cross-family model point (val, n=8)

Homogeneous **Gemma-4 31B** (`google/gemma-4-31B-it`; lm = vlm = 31B) across
the three harnesses — the cross-family large-model point against Qwen 3.5 27B.
`enable_thinking=false`, val 25 docs / 80 Q, per-question micro-average.
Cross-family synthesis: [`harness-axis-summary.md`](harness-axis-summary.md).

## Results

| Harness | Val (n=8) |
|---|---|
| RLM (`rvlm`) | **32.50% ± 4.48** |
| ReAct | **18.44% ± 3.58** |
| CodeAct | _n=8 running (n=1 pilot 37.50 — inside trial noise, do not cite as final)_ |
| `raw_vlm_multi_baseline` | _running_ |
| `official_baseline` | _running_ |

vs Qwen 3.5 27B homog: RLM 39.4 / ReAct 25.2 / CodeAct 37.0 — Gemma 31B is in
the same regime, the headline ordering preserved.

## Reads

- **rvlm ≫ react, the headline cross-family result.** RLM 32.50 ≫ ReAct 18.44
  — a **+14.1pp** gap, ≫ the combined std. The recursive VLM sub-call (`rvlm`)
  is **load-bearing**; the REPL-only ReAct harness collapses to the
  no-recursion tier. This mirrors Qwen 27B (RLM 39.4 ≫ ReAct 25.2) almost
  exactly, so the ordering **recursive-perception ≫ tool-only ReAct is robust
  across model families** — and is *sharp* at 31B while *vanishing into noise*
  at 4B (see `gemma-4-e4b.md`). Capacity gates whether the scaffold can be
  driven.
- Numbers moved modestly from the n=1/n=2 pilots (rvlm 30.00→32.50,
  react 20.00→18.44) — the pilots were inside the ~3–4pp trial noise.

## Setup / serving

- **No special tool/reasoning parser** — `rvlm`/`codeact`/`react` parse Python
  from text, not native `tool_calls`.
- **Serving:** `vllm/vllm-openai:gemma4` + `--reasoning-parser gemma4`. 31B runs
  **TP=2 @8931** (`--enforce-eager --shm-size=16g`), **one trial at a time** —
  an inherent shm-broadcast crash recurs under bursty multi-image load (server
  Exits(0) with a psm/resource_tracker warning → API "Connection refused");
  handled by restart-container + relaunch-`run_id` (resumes losslessly — failed
  docs aren't persisted). On 1 GPU it was glacial (~4 Q/h); TP=2 → ~80 Q/h.
- **31B ReAct slow-doc note.** ReAct on Gemma-31B is expensive on image-heavy
  docs (`engineering_drawing_1`, `science_poster_2`) — it can burn its full
  `max_iters` (~34) without emitting an answer (a single doc ~30–60min). Trials
  that stalled >10min on such a doc past 40min total runtime were accepted as
  **timeout = failure** (that doc scored 0/N via an empty-prediction
  placeholder, keeping a consistent /80 denominator). 3 of 8 ReAct trials used
  this; the other 5 completed all docs naturally and landed in the same range
  (13.75–22.50), so the placeholder does not bias the cell. rvlm/codeact do not
  hit this grind.

```bash
# 31B: CUDA_VISIBLE_DEVICES=1,2 --tensor-parallel-size 2 --enforce-eager --shm-size=16g
#      --port 8931 --model google/gemma-4-31B-it (vllm/vllm-openai:gemma4 --reasoning-parser gemma4)
uv run python evals.py lm=gemma-4-31b-vllm-local vlm=gemma-4-31b-vllm-local \
  lm.enable_thinking=false solver=rvlm data.split=val data.num_samples=null \
  max_concurrency=6 run_id=gemma-31b-rvlm-val-tN   # solver=react_baseline|codeact
```

Configs: `configs/{lm,vlm}/gemma-4-31b-vllm-local.yaml` (@8931).

## Status

`done` (rvlm + react, n=8) / `in progress` (codeact n=8, both baselines). The
31B codeact cell and `raw_vlm_multi_baseline` / `official_baseline` on both
Gemma sizes are running to complete the per-model **harness-lift table** —
update the rows above and `harness-axis-summary.md` when they land.
Coordination: `coordination/amax1.md`.
