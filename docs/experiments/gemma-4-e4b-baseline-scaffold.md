# Gemma 4 E4B-it — small-model baseline + scaffold (val)

**Hypothesis:** extends the §A model-axis to a non-Qwen small model.
Gemma E4B is a Google "effective ~4B" instruction-tuned multimodal
model — pairs with Qwen 3.5 9B as a second ≤8B point and tests whether
the scaffold lift generalizes across model families, not just within
Qwen.

**Setup:** Gemma 4 E4B-it served via local docker vllm 0.20.1 on Host
B, port 8904, GPU 0 (1× A100 80GB, prefix caching ON, async scheduling,
`--gpu-memory-utilization 0.85`, `--max-model-len 65536`). Val 80q.
`lm.enable_thinking=false`, `max_concurrency=8`, `question_concurrency=4`.
Both LM and VLM = same Gemma E4B endpoint.

## Commands

```bash
docker run --rm --name vllm-gemma-e4b --ipc=host --gpus '"device=0"' -p 8904:8904 \
  -v /home/baris/.cache/huggingface:/root/.cache/huggingface vllm/vllm-openai \
  --port 8904 --model google/gemma-4-E4B-it --async-scheduling --enable-prefix-caching \
  --gpu-memory-utilization 0.85 --dtype bfloat16 --max-model-len 65536

bash scripts/run_gemma_chain.sh gemma-4-e4b-vllm-local 4-e4b
# inside: 3 trials no_loop_multi+tips, then 3 trials flat_solo
# run_id pattern: no-loop-multi-tips-4-e4b-val-t{1,2,3}, flat-solo-4-e4b-val-t{1,2,3}
```

## Per-trial scores

### Baseline — no_loop_multi + tips

| Trial | run_id | Score | Correct | Sandbox errors |
|---|---|---|---|---|
| t1 | `no-loop-multi-tips-4-e4b-val-t1` | 3.75% | 3/80 | 0 (n/a) |
| t2 | `no-loop-multi-tips-4-e4b-val-t2` | 3.75% | 3/80 | 0 |
| t3 | `no-loop-multi-tips-4-e4b-val-t3` | 3.75% | 3/80 | 0 |

**Mean: 3.75% ± 0.00pp | range 3.75–3.75%**

Identical 3/80 each trial *by coincidence on the aggregate*: at the
question level, **46.2% of predictions differ across trials** (only
43/80 are exactly equal across t1=t2=t3). The model has real sampling
variance — it just happens to land on the same number correct each
time. Per-category: infographics 2/10, slide 1/10, all other
categories 0/10 across all 3 trials. This is a model-capability floor:
the E4B can only reliably answer a tiny set of trivial layout-grouping
questions (e.g. infographic single-fact lookups).

### Scaffold — flat_solo lean m=30

| Trial | run_id | Score | Correct | Sandbox errors |
|---|---|---|---|---|
| t1 | `flat-solo-4-e4b-val-t1` | 11.25% | 9/80 | 0 |
| t2 | `flat-solo-4-e4b-val-t2` | 8.75% | 7/80 | 0 |
| t3 | `flat-solo-4-e4b-val-t3` | 8.75% | 7/80 | 0 |

**Mean: 9.58% ± 1.44pp | range 8.75–11.25%**

## Summary

| Configuration | Mean | Std | n |
|---|---|---|---|
| no_loop_multi + tips (baseline) | 3.75% | 0.00pp | 3 |
| **flat_solo lean m=30 (scaffold)** | **9.58%** | **1.44pp** | **3** |
| Δ scaffold lift | **+5.83pp** | (SE 0.83, t=7.0, **highly significant**) | — |

Relative lift: 2.55× the baseline (9.58 / 3.75). The scaffold more
than doubles the absolute score on the smallest model in the axis.

## Per-category (mean over 3 trials)

| Category | Baseline t1/t2/t3 | Scaffold t1/t2/t3 | Baseline mean | Scaffold mean | Δ |
|---|---|---|---|---|---|
| business_report | 0/0/0 | 3/1/2 | 0.0/10 | 2.0/10 | **+2.0** |
| comics | 0/0/0 | 0/0/0 | 0.0/10 | 0.0/10 | 0 |
| engineering_drawing | 0/0/0 | 0/0/0 | 0.0/10 | 0.0/10 | 0 |
| infographics | 2/2/2 | 1/2/1 | 2.0/10 | 1.3/10 | −0.7 |
| maps | 0/0/0 | 0/0/0 | 0.0/10 | 0.0/10 | 0 |
| science_paper | 0/0/0 | 2/1/1 | 0.0/10 | 1.3/10 | **+1.3** |
| science_poster | 0/0/0 | 0/0/0 | 0.0/10 | 0.0/10 | 0 |
| slide | 1/1/1 | 3/3/3 | 1.0/10 | 3.0/10 | **+2.0** |
| **Overall** | 3/3/3 | 9/7/7 | **3.75%** | **9.58%** | **+5.83pp** |

## Comparison vs Qwen 9B and 27B (same axis)

| Model (size) | Baseline | Scaffold | Lift |
|---|---|---|---|
| Gemma 4 E4B-it (~4B effective) | 3.75% ± 0.00pp | 9.58% ± 1.44pp | **+5.83pp** |
| Qwen 3.5 9B | 15.00% ± 1.25pp | 21.25% ± 2.50pp | **+6.25pp** |
| Qwen 3.5 27B | 23.75% ± 2.17pp | 44.69% ± 2.81pp | +20.94pp |

Both small models (E4B and 9B) get a roughly comparable absolute lift
(+5.83 / +6.25pp), much smaller than the 27B (+20.94pp). The pattern
is consistent with the user's interpretation: **scaffold absolute gain
scales with model code-writing capability**, which scales with model
size. See memory entry `feedback_scaffold_lift_scales_with_model_size`.

## Observations

- **The scaffold opens up new categories the E4B baseline cannot
  touch.** business_report 0→2.0/10, slide 1.0→3.0/10, science_paper
  0→1.3/10. These are the categories where retrieval / cropping /
  multi-turn unlock answers that a single forward pass can't reach.
- **Comics, engineering_drawing, maps, science_poster all 0/10 in both
  baseline and scaffold.** The E4B floor is structural: even with the
  agent loop and tools, the model can't read engineering drawings,
  navigate comics, or trace maps. A 4B model is below the capability
  threshold for these categories.
- **Infographics regresses (2.0→1.3/10).** Same pattern as Qwen 9B:
  on single-page text-heavy layouts, the agent loop introduces
  off-task tool calls that walk the model away from an answer it
  could have produced in one shot.
- **No sandbox subprocess errors across all 6 trials** (clean infra).
- **Wall time:** baseline ~7 min/trial, scaffold ~50–60 min/trial.
  Total Gemma E4B chain wall: ~3.3h.
- **Variance.** Baseline std exactly 0pp on the aggregate but 46.2%
  question-level disagreement — the equality is a coincidence, not
  a deterministic-sampling artifact. Scaffold std 1.44pp.

## Status

**Done.** 3 baseline + 3 scaffold trials, all clean (0 sandbox errors).
Headline: scaffold lifts E4B by **+5.83pp** (3.75% → 9.58%, t=7.0,
highly significant despite the tiny base). Adds a non-Qwen ≤8B
data point — the §A "lifts every model class" claim survives onto
a smaller and a different family. Same lift-vs-size pattern as Qwen
9B vs 27B.
