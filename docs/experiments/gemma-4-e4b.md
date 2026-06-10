# Gemma-4 E4B (homog) — cross-family model point (val, n=8)

Homogeneous **Gemma-4 E4B** (`google/gemma-4-E4B-it`, ~4B-effective;
lm = vlm = E4B) across all three harnesses — the cross-family 4B point against
the Qwen 3.5 4B homog cells. Does the active-perception scaffold behave the
same on a different model family? `enable_thinking=false`, val 25 docs / 80 Q,
per-question micro-average. Cross-family synthesis:
[`harness-axis-summary.md`](harness-axis-summary.md).

## Results

| Harness / baseline | Val (n=8) | Lift vs no-scaffold |
|---|---|---|
| RLM (`rvlm`) | 7.34% ± 3.30 | +1.1pp (n.s.) |
| CodeAct | 7.66% ± 1.94 | +1.4pp (n.s.) |
| ReAct | 6.09% ± 2.36 | −0.2pp (n.s.) |
| `raw_vlm_multi_baseline` | 3.75% ± 0.00 | — |
| `official_baseline` | 6.25% ± 1.16 | — |

**No-scaffold baseline = max(rawvlm 3.75, official 6.25) = 6.25.** All three
harnesses sit **within noise of the `official_baseline`** — no scaffold lift at
4B. This is the clean **negative control**: lift requires a capable-enough base.
vs Qwen 3.5 4B homog: RLM 21.1 / ReAct 11.9 / CodeAct 12.2 — E4B lands far
lower (weak homogeneous vision + weak reasoner).

## Reads

- **All three harnesses tied ~6–8%.** RLM 7.34 / ReAct 6.09 / CodeAct 7.66 are
  statistically indistinguishable (within ~1 std; per-cell stds 1.9–3.3pp). A
  4B model is **too weak to exploit any scaffold** — it burns its iteration
  budget on coding mistakes and has weak homogeneous vision, so harness type
  doesn't separate. These are genuine predictions (not a silent-tool-fail
  artifact — verified `batch_look` fires and the model self-corrects from real
  sandbox errors). This is the clean **negative control** for the harness-lift
  hypothesis on a second model family: lift requires a capable-enough base, and
  a 4B isn't it.
- Numbers moved modestly from the earlier n=1/n=2 pilots (rvlm 6.88→7.34,
  react 4.38→6.09) — the pilots were inside the ~3–4pp trial noise, as expected.

## Setup / serving

- Models cached on amax1. Homogeneous: lm = vlm = E4B. `enable_thinking=false`.
- **No special tool/reasoning parser:** `rvlm`/`codeact`/`react` parse Python
  code from the model's text response, not native vllm `tool_calls`.
- **Serving:** canonical image is **`vllm/vllm-openai:gemma4`** +
  **`--reasoning-parser gemma4`** (per `docs/scratchpad.md`), NOT generic
  `:latest`. The image ENTRYPOINT is already `["vllm","serve"]`, so the
  container command must start with `--port` (a leading `serve` →
  `unrecognized arguments: serve` → instant exit). E4B = **DP=2 @8904**
  (stable, no shm crash); ran fine on 1 GPU.

```bash
docker run -d --name gemma-e4b --runtime nvidia --gpus all -e CUDA_VISIBLE_DEVICES=1 \
  --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=<tok> \
  -p 8904:8904 vllm/vllm-openai:gemma4 --port 8904 --model google/gemma-4-E4B-it \
  --reasoning-parser gemma4 --data-parallel-size 2 --gpu-memory-utilization 0.85 \
  --dtype bfloat16 --max-model-len 131072 --limit-mm-per-prompt '{"image":64}'

uv run python evals.py lm=gemma-4-e4b-vllm-local vlm=gemma-4-e4b-vllm-local \
  lm.enable_thinking=false solver=rvlm data.split=val data.num_samples=null \
  max_concurrency=6 run_id=gemma-e4b-rvlm-val-tN   # solver=react_baseline|codeact
```

Configs: `configs/{lm,vlm}/gemma-4-e4b-vllm-local.yaml` (@8904).

## Status

`done` — full per-model harness-lift table complete: all three harnesses (n=8)
+ both baselines (n=8). At 4B no harness clears the `official_baseline` (clean
negative control). See `harness-axis-summary.md` for the cross-family synthesis.
Coordination: `coordination/amax1.md`.
