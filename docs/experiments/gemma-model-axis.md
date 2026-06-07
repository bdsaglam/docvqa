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

| Model | rvlm | codeact | vs Qwen 3.5 4B homog |
|---|---|---|---|
| **Gemma-4 E4B** | **6.88%** (6.25, 7.50) | **7.50%** (8.75, 6.25) | rvlm 21.1 / codeact 12.2 — **Gemma E4B far lower** |
| **Gemma-4 31B** | _in progress (TP=2)_ | _pending_ | (Qwen 27B homog: rvlm 39.4 / codeact 37.0) |

**Read (E4B):** Gemma-4 E4B scores ~7% on both harnesses — well below Qwen
3.5 4B (21% rvlm). The tiny elastic model burns its iteration budget on
coding mistakes and has weak homogeneous vision, so it can barely drive the
recursive code+perception loop. Real scores (genuine predictions), not a
config artifact. The 31B point (running) tells us whether this is "Gemma
weak at this task" or "small models can't drive the scaffold."

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

E4B **done** (n=2): rvlm 6.88%, codeact 7.50%. 31B **in progress** (TP=2):
rvlm-t1 running, codeact-t1 pending (n=1 cost-limited). n>2 escalation left
to the user. Rolled into `harness-types-vlm-axis.md` (v1 homog table).
