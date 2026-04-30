# Experiment Plan: Temperature & Sampling Parameter Sweep

## Goal
Find optimal LLM and VLM sampling parameters for Qwen3.5-27B on DocVQA 2026 val set.
Reduce trial-to-trial variance while maintaining or improving accuracy.

## Baseline
- **Best Qwen/Qwen**: 42.5% (flat_batch lean, b6-pq4+pf1.5, t=1.0/1.0) — but high variance (35-42.5%)
- **Best overall**: 55.0% (Pro LLM + Flash VLM)

## Qwen3.5-27B Recommended Settings (from HuggingFace)
- **Non-thinking general**: `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`
- **Precise tasks**: `temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0`

## Common Config (all experiments)
- Solver: `flat_batch` (lean)
- Budget: `b6-pq4+pf1.5` (base=6, pq=4, page_factor=1.5, max=40)
- Server: remote Qwen (port 8928)
- Concurrency: 4
- Data: full val set (25 docs, 80 questions)

---

## Phase 1: VLM Temperature Sweep (isolate VLM effect)

Fix LLM at t=1.0 (our baseline), sweep VLM temperature.

| Exp | LLM Config | VLM Config | Command Suffix |
|-----|-----------|-----------|----------------|
| 1a | t=1.0 | t=1.0 | `lm.temperature=1.0 vlm.temperature=1.0` |
| 1b | t=1.0 | t=0.6 | `lm.temperature=1.0 vlm.temperature=0.6` |
| 1c | t=1.0 | t=0.3 | `lm.temperature=1.0 vlm.temperature=0.3` |

**Already have partial data:**
- 1a: 35.0-42.5% (2 trials, high variance)
- 1b: running (flat-lm10-vlm06), partial
- 1c: running (flat-lm10-vlm03), partial

## Phase 2: LLM Temperature Sweep (isolate LLM effect)

Fix VLM at best from Phase 1, sweep LLM temperature.

| Exp | LLM Config | VLM Config | Command Suffix |
|-----|-----------|-----------|----------------|
| 2a | t=0.3 | t=best_from_P1 | `lm.temperature=0.3 vlm.temperature=X` |
| 2b | t=0.6 | t=best_from_P1 | `lm.temperature=0.6 vlm.temperature=X` |
| 2c | t=0.7 | t=best_from_P1 | `lm.temperature=0.7 vlm.temperature=X` |
| 2d | t=1.0 | t=best_from_P1 | `lm.temperature=1.0 vlm.temperature=X` |

**Already have:**
- LLM t=0.6, VLM t=1.0: 42.5%

## Phase 3: Qwen Recommended Sampling Parameters

Test top_k, top_p, presence_penalty from Qwen docs.
Requires adding these params to LMConfig and solver factories.

| Exp | LLM Config | VLM Config | Notes |
|-----|-----------|-----------|-------|
| 3a | t=0.7, top_p=0.8, top_k=20, pp=1.5 | t=0.6, top_p=0.95, top_k=20, pp=0.0 | Qwen recommended (non-thinking LLM + precise VLM) |
| 3b | t=0.7, top_p=0.8, top_k=20, pp=1.5 | t=best_from_P1 | Qwen LLM + best VLM temp |
| 3c | t=best_from_P2, top_p=0.8, top_k=20, pp=1.5 | t=best_from_P1, top_p=0.95, top_k=20 | Best temps + Qwen sampling |

**Prereq:** Add `top_p`, `top_k`, `presence_penalty` support to:
- `LMConfig.to_dspy_lm()` in `types.py`
- VLM config passthrough in all solver factories
- Hydra YAML configs

## Phase 4: Variance Measurement

Run the best config from Phases 1-3 three times to measure variance.

| Exp | Config | Notes |
|-----|--------|-------|
| 4a | best config, trial 1 | |
| 4b | best config, trial 2 | |
| 4c | best config, trial 3 | |

Expected: if lower temperature works, std should drop from ~7pp to ~3pp.

## Phase 5: Pro LLM (when quota available)

Test best VLM settings with Pro LLM.

| Exp | LLM Config | VLM Config | Notes |
|-----|-----------|-----------|-------|
| 5a | Pro 3.1, t=1.0 | Qwen, t=best | Pro reasoning + stable VLM |
| 5b | Pro 3.1, t=1.0 | Flash, t=default | Previous best (55%), re-test with new prompts |

---

## Execution Order

1. **Phase 1** first — cheapest, isolates VLM effect (3 runs)
2. **Phase 2** — uses Phase 1 winner (3-4 runs)
3. **Phase 3** — requires code changes, then 2-3 runs
4. **Phase 4** — 3 runs of best config
5. **Phase 5** — Pro quota dependent

## Launch Template

```bash
# Remote Qwen, c4
tmux new-window -t docvqa-evals -n EXP_NAME \
  "stdbuf -oL uv run python evals.py \
    lm=qwen27b-remote vlm=qwen27b-remote \
    lm.temperature=LLM_T vlm.temperature=VLM_T \
    solver=flat_batch data.num_samples=null \
    max_concurrency=4 run_id=RUN_ID \
    2>&1 | tee /tmp/RUN_ID.log; exec bash"
```

## Results Tracking

Update `docs/eval-results.md` as runs complete. Key columns to add: LLM temp, VLM temp.
