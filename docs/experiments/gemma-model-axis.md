# Gemma-4 model-axis — rvlm + codeact (val)

## Question

Extend the harness model-size axis (Qwen 3.5 4B/9B/27B homog) with a
**cross-family** point: homogeneous **Gemma-4** (lm=vlm=Gemma) running the
`rvlm_minimal` and `codeact` harnesses. Does the active-perception scaffold
behave the same on a different model family, and how do the Gemma sizes land
vs the Qwen curve?

## Setup

- Models (both cached on amax1): **Gemma-4 E4B** (`google/gemma-4-E4B-it`,
  ~4B-effective) and **Gemma-4 31B** (`google/gemma-4-31B-it`).
- Homogeneous: lm = vlm = the same Gemma. `enable_thinking=false`.
- Profile: DocVQA-2026 (val, 25 docs / 80 Q). **n=8** (escalated from the
  n=1/n=2 pilots on 2026-06-09); + 2 baselines for the harness-lift table.
- **Parser:** Gemma needs **no special tool/reasoning parser** — `rvlm`/
  `codeact` parse Python code from the model's text response, not native
  vllm tool_calls. Verified by a smoke (batch_look fires, the model
  self-corrects from real sandbox errors → not the silent-tool-fail
  "fake-low score" failure mode).
- **GPU serving (lesson learned):** serve ONE Gemma on the full GPU set and
  finish it, then switch — don't shard small+large 1-GPU-each. Use **DP**
  for a small model that fits (throughput), **TP** for a large one that's
  tight on a single GPU (memory + per-call latency). E4B ran fine on 1 GPU;
  31B was glacial on 1 GPU (~4 Q/h) and was moved to **TP=2** (~80 Q/h).

## Results (n=8; escalated 2026-06-09)

Harnesses (rvlm / react / codeact) are reported at **n=8**. **Baselines**
(`raw_vlm_multi_baseline` = multi-image single VLM call, `official_baseline`
= kit baseline) were added 2026-06-09 to build a per-model **harness-LIFT
table** (scaffold vs no-scaffold); they and the 31B codeact cell are **still
running** — pending cells marked below.

| Model | rvlm | ReAct | codeact | rawvlm (base) | official (base) |
|---|---|---|---|---|---|
| **Gemma-4 E4B** | **7.34% ± 3.30** | **6.09% ± 2.36** | **7.66% ± 1.94** | _running_ | _running_ |
| **Gemma-4 31B** | **32.50% ± 4.48** | **18.44% ± 3.58** | _running_ (n=1 pilot 37.50) | _running_ | _running_ |

vs Qwen homog: Qwen 4B rvlm 21.1 / react 11.9 / codeact 12.2 (E4B far lower);
Qwen 27B rvlm 39.4 / react 25.2 / codeact 37.0 (Gemma 31B same regime).

**Read (E4B) — all harnesses tied ~6–8%:** Gemma-4 E4B lands rvlm 7.34 /
react 6.09 / codeact 7.66 — **statistically indistinguishable** (within ~1
std; per-cell stds 1.9–3.3pp). A 4B model is **too weak to exploit any
scaffold**: it burns its iteration budget on coding mistakes and has weak
homogeneous vision, so harness type doesn't separate. Real scores (genuine
predictions), not a config artifact. This is the clean **negative control**
for the lift hypothesis — lift requires a capable-enough base model, and a 4B
isn't it.

**Read (31B) — rvlm ≫ react, the headline:** Gemma-4 31B lands **rvlm 32.50
≫ react 18.44, a +14.1pp gap ≫ the combined std.** The recursive VLM sub-call
(rvlm) is **load-bearing**; the REPL-only ReAct harness collapses to the
no-recursion tier. This mirrors Qwen 27B (rvlm 39.4 ≫ react 25.2) almost
exactly, so the ordering **recursive-perception ≫ tool-only ReAct is robust
across model families** — and is *sharp* at 31B while *vanishing into noise*
at 4B (capacity gates whether the scaffold can be driven). codeact at n=8 is
pending (the n=1 pilot 37.50 was inside trial noise; don't cite it as final).

**Read (ReAct):** ReAct is the **weakest harness on both Gemma sizes**, same
as Qwen. E4B ReAct 6.09 sits at the floor with rvlm/codeact (4B can't sustain
any trajectory). 31B ReAct 18.44 ≪ rvlm 32.50 — lacking a recursive-perception
sub-call costs it ~14pp. **31B ReAct is also expensive:** on image-heavy docs
(`engineering_drawing_1`, `science_poster_2`) it can burn its full `max_iters`
(~34) without answering (one doc ~30–60min); trials stalled >10min on such a
doc past 40min total were accepted as **timeout=failure** (doc → 0/N via
empty-prediction placeholder, consistent /80 denominator). 3/8 react trials
used this; the 5 that completed naturally landed in the same 13.75–22.50 range,
so it doesn't bias the cell. rvlm/codeact don't hit this grind.

**Serving note:** the canonical Gemma image is **`vllm/vllm-openai:gemma4`**
with **`--reasoning-parser gemma4`** (per `docs/scratchpad.md`) — NOT the
generic `:latest`. The image ENTRYPOINT is already `["vllm","serve"]`, so the
container command must start with `--port` (passing a leading `serve` →
`unrecognized arguments: serve` → instant exit).

## How to run

```bash
# bring up the Gemma server (example: E4B on a GPU, DP for small / TP for 31B)
docker run -d --name gemma-e4b --runtime nvidia --gpus all -e CUDA_VISIBLE_DEVICES=1 \
  --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=<tok> \
  -p 8904:8904 vllm/vllm-openai --port 8904 --model google/gemma-4-E4B-it \
  --data-parallel-size 1 --gpu-memory-utilization 0.85 --dtype bfloat16 \
  --max-model-len 131072 --limit-mm-per-prompt '{"image":64}'
# (31B: CUDA_VISIBLE_DEVICES=1,2 --tensor-parallel-size 2 ... --port 8931 --model google/gemma-4-31B-it)

# eval (homogeneous Gemma, rvlm or codeact)
uv run python evals.py lm=gemma-4-e4b-vllm-local vlm=gemma-4-e4b-vllm-local \
  lm.enable_thinking=false solver=rvlm_minimal data.split=val data.num_samples=null \
  max_concurrency=6 run_id=gemma-e4b-rvlm-val-tN
# (solver=codeact for codeact; lm=vlm=gemma-4-31b-vllm-local for 31B)
```

Configs: `configs/{lm,vlm}/gemma-4-{e4b,31b}-vllm-local.yaml` (E4B @8904, 31B @8931).

## Status

**IN PROGRESS — n=8 escalation + baselines (started 2026-06-09).**
- **Harnesses DONE (n=8):** E4B rvlm 7.34±3.30 / react 6.09±2.36 /
  codeact 7.66±1.94; 31B rvlm 32.50±4.48 / react 18.44±3.58.
- **Running:** 31B codeact (n=8), and both baselines
  (`raw_vlm_multi_baseline`, `official_baseline`) on **both** models — added
  to quantify per-model **harness lift** (scaffold vs no-scaffold).
- **Reprioritized 2026-06-09 (user request):** finish the in-flight 31B
  codeact-t1, then **detour to the E4B baselines** (16 trials) before the
  remaining 31B cells, so the 4B model's lift table completes first.
- **Headline so far:** 31B rvlm 32.50 ≫ react 18.44 (+14.1pp; recursion
  load-bearing, robust vs Qwen). E4B all harnesses tied 6–8% (4B too weak
  to exploit any scaffold — clean negative control).
- Rolled into `harness-types-vlm-axis.md` (v1 homog table) + `results.md` +
  `experiment-status.md`. Per-host coordination: `coordination/amax1.md`.

**Operating notes (amax1):** 31B = TP=2 on `vllm/vllm-openai:gemma4`
(`--enforce-eager --shm-size=16g`), **one trial at a time** — an inherent
shm-broadcast crash recurs under bursty multi-image load (server Exits(0) with
a psm/resource_tracker warning → API "Connection refused"); handled by
restart-container + relaunch-run_id (resumes losslessly — failed docs aren't
persisted). E4B = DP=2 @8904 (stable, no shm crash). Always keep the 27B up
@8927 on GPU0.
