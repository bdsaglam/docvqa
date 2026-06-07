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
- Profile: DocVQA-2026 (val, 25 docs / 80 Q). n=2 pilots (D-008 budget).
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

## Results (n=2 pilots)

| Model | rvlm | ReAct | codeact | vs Qwen homog |
|---|---|---|---|---|
| **Gemma-4 E4B** | **6.88%** (6.25, 7.50; n=2) | **4.38%** (2.50, 6.25; n=2) | **7.50%** (8.75, 6.25; n=2) | Qwen 4B: rvlm 21.1 / react 11.9 / codeact 12.2 — **E4B far lower** |
| **Gemma-4 31B** | **30.00%** (n=1) | **20.00%** (n=1) | **37.50%** (n=1) | Qwen 27B: rvlm 39.4 / react 25.2 / codeact 37.0 — same ballpark |

**Read (E4B):** Gemma-4 E4B scores ~7% on both harnesses — well below Qwen
3.5 4B (21% rvlm). The tiny elastic model burns its iteration budget on
coding mistakes and has weak homogeneous vision, so it can barely drive the
recursive code+perception loop. Real scores (genuine predictions), not a
config artifact.

**Read (31B):** Gemma-4 31B (n=1) lands at **rvlm 30.00 / codeact 37.50** —
below Qwen 3.5 27B (rvlm 39.4 / codeact 37.0) but in the same regime, so the
E4B collapse is a **scale/capacity** effect ("small models can't drive the
scaffold"), not a Gemma-family weakness: at 31B Gemma runs both harnesses
competently. Note codeact (37.5) > rvlm (30.0) here — the **reverse** of the
Qwen 27B ordering (rvlm 39.4 > codeact 37.0). At n=1 per cell this is inside
trial noise (recall Qwen rvlm/codeact sit ~2pp apart with ±1.5-5pp std), so
read it as "the two harnesses are close on Gemma 31B," not a robust
cross-family inversion — n>2 would be needed to claim one.

**Read (ReAct, added 2026-06-07):** ReAct is the **weakest harness on both
Gemma sizes**, same as Qwen. E4B ReAct **4.38** (n=2) is below its own rvlm
6.9 / codeact 7.5 — near the floor; a 4B model can't sustain the tool-only
ReAct trajectory. 31B ReAct **20.00** (n=1) ≪ its rvlm 30.0 / codeact 37.5,
exactly mirroring Qwen 27B (react 25.2 ≪ rvlm 39.4 / codeact 37.0). The
harness ordering — REPL-bearing rvlm/codeact > tool-only ReAct — is therefore
**robust across model families**: lacking a Python REPL costs ReAct the most,
regardless of model. Predictions verified real (e.g. map place-names read off
the page, not all-Unknown).

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

**DONE (2026-06-07).** Full harness sweep:
- **E4B** (n=2): rvlm 6.88 / react 4.38 / codeact 7.50.
- **31B** (n=1): rvlm 30.00 / react 20.00 / codeact 37.50.
ReAct weakest on both (REPL > tool-only, robust across families). n>2
escalation left to the user. Rolled into `harness-types-vlm-axis.md` (v1
homog table) + `experiment-status.md`.
