# Qwen 3.5 (9B + 4B) — rvlm_minimal, VLM-quality axis

> ## ⚠ LOCKED n=8 RESULTS (captured 2026-06-01) — supersede the n=1 numbers below
>
> Per-question micro-average over 80 Qs/trial, val 25 docs, new code
> (whole-agent retry removed; `num_retries=5` per-call). Both arms run
> with `lm.enable_thinking=false` for parity.
>
> **Qwen 3.5 9B reasoner (LLM fixed at 9B, swap VLM):**
> | Arm | VLM | Mean | SD | n |
> |---|---|---|---|---|
> | v1 homog | 9B | **16.67%** | 3.40 | 8 |
> | v2 mixed | 27B | **24.54%** | 5.30 | 8 |
> | **Δ (v2−v1)** | | **+7.87pp** | Welch t=3.54, 95%CI [+3.4,+12.3] | **significant** |
>
> **Qwen 3.5 4B reasoner (LLM fixed at 4B, swap VLM):**
> | Arm | VLM | Mean | SD | n |
> |---|---|---|---|---|
> | v1 homog | 4B | **12.49%** | 3.74 | 8 |
> | v2 mixed | 27B | **21.09%** | 3.16 | 8 |
> | **Δ (v2−v1)** | | **+8.60pp** | Welch t=4.96, 95%CI [+5.20,+11.99] | **significant** |
>
> 4B v1 per-trial: 10.77 / 11.34 / 12.87 / 14.48 / 17.54 / 5.34 / 16.00
> / 11.60. 4B v2 per-trial: 25.00 / 20.00 / 21.25 / 23.75 / 23.75 /
> 20.00 / 15.00 / 20.00.
>
> **Qwen3-8B reasoner — TEXT-ONLY, different family (v2 only):**
> | Arm | LLM / VLM | Mean | SD | n |
> |---|---|---|---|---|
> | v2 mixed | Qwen3-8B (text-only) / 27B | **11.73%** | 2.96 | 8 |
>
> 8B v2 per-trial: 18.18 / 12.84 / 10.90 / 9.71 / 10.18 / 10.34 / 12.88
> / 8.84. (No v1: Qwen3-8B is text-only, can't serve as the VLM.)
>
> Headline (9B + 4B, locked): swapping **only** the VLM →27B with the
> reasoner fixed lifts ~8pp at **both** sizes (9B +7.87pp, 4B +8.60pp) →
> the scaffold is **perception-budget-bound** for mid/small reasoners,
> supporting D-006. The lift is consistent across reasoner size, which
> is the signature of a perception (not orchestration) bottleneck. (NB
> this reverses the stale n=1 "0.0pp" read below, a single high-variance
> draw on the 9B v1 arm.)
>
> **The 8B point — an older-generation reasoner, NOT a modality
> confound.** In v2 the reasoner delegates *all* perception to the 27B
> VLM via `batch_look`; it never sees pixels in its own context
> (`rvlm_solver.py` loads `pages` only inside the sandbox). So text-only
> vs multimodal is **irrelevant in v2** — every reasoner is a text
> orchestrator on the same VLM. The only variable vs 9B/4B is
> **generation**: Qwen3-8B is the older Qwen3 family, the 9B/4B are
> Qwen3.5. The older 8B is a **weaker orchestrator** — it thrashed ~18
> RLM iterations/question (force-submitting wrong) and scored 11.73%,
> *below* even the newer 4B (21.09%). Bug ruled out: `enable_thinking=
> false` is correctly applied (`types.py:66` → `chat_template_kwargs`),
> tool/parse errors are negligible, `batch_look` returns real content.
> Reading: even with a strong fixed VLM, a weaker reasoner can't drive
> the scaffold — a clean reasoner-*quality* signal. It sits off the
> Qwen3.5 9B↔4B size curve because it's a different generation, so keep
> it as a separate quality point rather than a size step.

## Hypothesis / question

For a small (9B) reasoner driving the `rvlm_minimal` scaffold, is the
score limited by **perception** (the VLM it calls to read pages) or by
the **orchestrator/LLM** (the 9B writing the code and reasoning over
returned reads)?

Test by holding the reasoner fixed at Qwen 3.5 9B and swapping only the
VLM tool backend:

- **Variant 1 (homogeneous):** LLM = VLM = Qwen 3.5 9B.
- **Variant 2 (mixed):** LLM = Qwen 3.5 9B, VLM = Qwen 3.5 **27B**.

This is the model-size axis (D-006 prediction 1) viewed from the
perception side: if perception is the binding constraint, a 3× larger
VLM should lift the headline. If the 9B orchestrator is the constraint,
upgrading the VLM buys nothing.

## Setup

- Solver: `rvlm_minimal` (`rlm_type=lean`, `max_iterations=25`,
  `page_factor=1.5`, `question_concurrency=4`).
- LLM: Qwen 3.5 9B vllm @ `localhost:8909` (DP=4, vision enabled,
  `max-model-len=65536`), `enable_thinking=false`.
- VLM: variant 1 → same 9B @ :8909; variant 2 → Qwen 3.5 27B @
  `localhost:8928`.
- Data: `val`, all 25 docs / 80 questions. `max_concurrency=16`.
- Metric: exact/threshold correctness, micro-averaged over the 80
  questions (as reported by `scripts/report.py`).
- **n = 8 per variant** (per-user direction 2026-05-31, escalating
  past the D-008 n=1→n=2 ladder). Trials `t1`..`t8`; `t1` is the
  original exploratory run. **n=1 numbers below; n=8 table filled on
  completion.**

## Command

```bash
# Variant 1 — homogeneous 9B (LLM = VLM = 9B)
uv run python evals.py \
  lm=qwen-3_5-9b-vllm-local vlm=qwen-3_5-9b-vllm-local \
  lm.enable_thinking=false solver=rvlm_minimal \
  data.split=val data.num_samples=null max_concurrency=16 \
  run_id=rvlm-minimal-3_5-9b-val

# Variant 2 — mixed: 9B LLM + 27B VLM
# NB: the 27B was serving at :8928, not the config default :8927.
uv run python evals.py \
  lm=qwen-3_5-9b-vllm-local \
  vlm=qwen-3_5-27b-vllm-local vlm.api_base=http://localhost:8928/v1 \
  lm.enable_thinking=false solver=rvlm_minimal \
  data.split=val data.num_samples=null max_concurrency=16 \
  run_id=rvlm-minimal-9b-llm-27b-vlm-val
```

## Per-trial table

n=8 in progress (trials `t1`..`t8`, run_ids
`rvlm-minimal-3_5-9b-val-tN` / `rvlm-minimal-9b-llm-27b-vlm-val-tN`).
n=1 (t1) below; full n=8 mean ± std filled on completion.

| run_id | LLM | VLM | score | correct/total | wall* |
|---|---|---|---|---|---|
| `rvlm-minimal-3_5-9b-val-t1` | Qwen3.5-9B | Qwen3.5-9B | **21.2%** | 17/80 | ~48 min |
| `rvlm-minimal-9b-llm-27b-vlm-val-t1` | Qwen3.5-9B | Qwen3.5-27B | **21.2%** | 17/80 | ~19 min (excl. stall) |

\* Wall is approximate. Variant 2's clock excludes a ~1h50m stall on a
single hung question (see Observations); the productive runtime was
shorter than variant 1 because vision was offloaded to the separate
27B server, leaving the 9B free for orchestration.

### Per-category (question-level, 10 Q/category)

| Category | V1 (9B/9B) | V2 (9B/27B) | Δ |
|---|---|---|---|
| business_report | 2/10 | 2/10 | 0 |
| comics | 2/10 | 1/10 | −1 |
| engineering_drawing | 3/10 | 3/10 | 0 |
| infographics | 3/10 | 4/10 | +1 |
| maps | 1/10 | 1/10 | 0 |
| science_paper | 2/10 | 2/10 | 0 |
| science_poster | 2/10 | 1/10 | −1 |
| slide | 2/10 | 3/10 | +1 |
| **TOTAL** | **17/80** | **17/80** | **0** |

## Summary

- Both variants: **21.2% (17/80)**, n=1.
- Upgrading the VLM 9B → 27B moved the headline by **0.0pp**.
- The tie is at the *aggregate* level only. Per-category, the 27B VLM
  redistributes outcomes: +1 each on text-dense **slide** and
  **infographics**, −1 each on **comics** and **science_poster**, net
  zero. So the stronger VLM does change *which* questions are answered
  — it just doesn't change *how many*.

## Comparison

- vs the 27B-reasoner reference (`rvlm_minimal`, Qwen 27B/27B,
  ~42% val mean over n=8 per `coordination/amax1.md` handoff): the 9B
  reasoner lands ~21pp lower. That gap is **not** closed by giving the
  9B a 27B VLM — pointing at the reasoner, not perception, as the 9B's
  ceiling.
- Speaks to **D-006 prediction 1 (model size)** from the perception
  side: the visual-context-budget mechanism only pays off when the
  reasoner is strong enough to exploit better reads. At 9B the
  orchestration/code-writing capability is the bottleneck, so perception
  quality is slack. Consistent with the standing note that scaffold and
  perception lift scale with model size — small-model lift is expected
  to be small and should not be flagged as a surprise.

## Observations / caveats

- **n=1, ~3–4% trial σ.** Read the 0.0pp as "no detectable lift," not
  literal equality. Same correct *count* (17), not the same 17
  questions (see category shifts). Locking this as a paper claim would
  need n≥3 on both arms.
- **Infra: a single hung question.** Variant 2 stalled at 24/25 for
  ~1h50m on `maps_2_q5` — the request was dispatched (18:47) but never
  logged even iteration 1, i.e. an I/O stall on a non-returning model
  call, not a reasoning loop (which would have hit `max_iterations=25`
  and submitted). Killed the process (SIGINT didn't break the blocked
  asyncio call; needed SIGKILL) and relaunched the same command;
  resume re-ran only `maps_2`, and q5 then returned CORRECT. The
  per-doc resumability (`run_id`) made this a clean recovery.
- Servers: 9B brought up with vision (DP=4) at :8909; 27B reachable at
  :8928 (the config default :8927 was not up — hence the
  `vlm.api_base` override).

## Status

`in progress` — n=1 (t1) done and reported above; n=8 launched
2026-05-31 (per-user direction) to lock the "VLM quality is slack for
a small reasoner" point with a real σ. Heartbeat-driven on amax7, one
trial per variant at a time. This section updates to `done` with the
n=8 mean ± std and a paired per-trial Δ once t8 lands.
