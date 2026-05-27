# amax1 queue (throughput host)

amax1 is the throughput host — runs side-track experiments where the
direction is already known and we just need to lock numbers. No adaptive
iteration; if a cell shows an unexpected direction, **halt and append a
`## NOTE FOR AMAX7`** section at the bottom of this file.

**vllm:** brings up its own per-model containers as needed. Set
`MODEL_TAG` per cell.

## In progress

### 1. `[→]` rvlm n=8 val — paired-comparison anchor for unified-tips (task #28)

Claimed 2026-05-27T23:11Z. tmux session `rvlm-paired`, c=24 (user
override of default c=32; concurrency affects wall-time/load, not
per-question accuracy, so the paired comparison stays valid). run_ids
`rvlm-val-t1`..`rvlm-val-t8`.

**Priority over model-axis cells below.** amax7 is running `rvlm_unified`
t2..t8 to escalate the unified-tips ablation to n=8 (t1 = 45.0%). We have
no current-prompt n=8 baseline for `rvlm` itself at c=32 — the legacy
per-trial mean of 42.8% comes from pre-rename `leanest_solo` n=8 on older
prompts. Without a matched-conditions `rvlm` baseline, the within-trial
comparison for the unified-tips decision (promote vs keep) is weaker.

This cell locks the matched-conditions baseline so the unified-vs-rvlm
comparison is clean (same model, same prompts, same c, paired t1..t8
trials).

```bash
# Bring up Qwen 3.5 27B vllm on amax1 (matching amax7's localhost:8927).
# Then:
tmux new-session -d -s rvlm-paired bash scripts/run_rvlm_paired_baseline.sh
# Resumes-on-crash: bash scripts/run_rvlm_paired_baseline.sh (it skips
# completed run_ids).
```

- Expected wall: ~7h (8 × ~50min)
- Compare to: `rvlm_unified` t1..t8 from amax7 (paired by trial number)
- All trials use `solver=rvlm`, `data.split=val`, `data.num_samples=null`,
  `max_concurrency=32`, `lm.enable_thinking=false`. Identical to amax7's
  unified-tips chain modulo `solver`.
- run_ids: `rvlm-val-t1` through `rvlm-val-t8`.

## Queued

### 2. `[ ]` Gemma 4 E4B baseline + scaffold n=1 val (task #8 part 1)

Re-run on clean prompts (D-009). Original 2026-05-09 cells used
pre-scrub prompts. Direction is robust (+5.83pp lift in original n=3);
just locking the magnitude under clean prompts.

```bash
# Bring up vllm for Gemma 4 E4B (see existing setup notes in
# docs/experiments/gemma-4-e4b-baseline-scaffold.md)
TRIALS=1 SOLVER=rvlm bash scripts/run_gemma_chain.sh gemma-4-e4b-vllm-local 4-e4b
```

- Expected wall: ~2-3h (baseline + scaffold)
- Expected direction: lift sign preserved (~+5pp baseline → scaffold)

### 3. `[ ]` Qwen 3.5 9B baseline + scaffold n=1 val (task #8 part 2)

```bash
TRIALS=1 SOLVER=rvlm bash scripts/run_gemma_chain.sh qwen-3_5-9b-vllm-local 3_5-9b
```

- Expected wall: ~2-3h
- Expected direction: lift sign preserved (~+6pp from original n=3)

### 4. `[ ]` Gemma 4 31B baseline + scaffold n=1 val (task #8 part 3)

```bash
# Per docs/experiments/gemma-4-31b-baseline-scaffold.md: needs
# vllm --tensor-parallel-size 4 --enforce-eager to survive scaffold load
TRIALS=1 SOLVER=rvlm bash scripts/run_gemma_chain.sh gemma-4-31b-vllm-local 4-31b
```

- Expected wall: ~3-4h
- Expected direction: lift sign preserved (~+25pp from original n=3)

## Done

(none yet under new naming)

## Decision rules (set in advance)

- **All three cells: lift sign preserved** → model-axis prediction 1 is
  robust under clean prompts. Escalate to n=2 (then n=8 only after the
  paper headline locks).
- **Any cell: lift sign reverses** → halt this queue and `## NOTE FOR
  AMAX7`. Pre-scrub prompts were inflating the lift differentially
  somehow; need to investigate before drawing model-axis conclusions.
- **Magnitude shifts by 5+pp** → expected (per-trial std ~3-4pp on
  Qwen 27B; smaller models can be noisier). Note in the result line;
  still escalate to n=2.
