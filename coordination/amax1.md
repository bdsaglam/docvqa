# amax1 queue (throughput host)

amax1 is the throughput host — runs side-track experiments where the
direction is already known and we just need to lock numbers. No adaptive
iteration; if a cell shows an unexpected direction, **halt and append a
`## NOTE FOR AMAX7`** section at the bottom of this file.

**vllm:** brings up its own per-model containers as needed. Set
`MODEL_TAG` per cell.

## In progress

### A. `[→]` ReAct baseline n=8 val — REPL-vs-no-REPL ablation

Claimed 2026-05-28T14:47Z. tmux `react-chain` running
`scripts/run_react_chain.sh` (c=24, run_ids `react-val-t1..t8`).
Tests whether the code-REPL in LeanRLM is doing real work — vs plain
`dspy.ReAct` with the same VLM tool surface (look + look_many) but no
Python execution. Paired comparison vs `rvlm` (per-category) and
`rvlm_unified` (decided default). Smoke test on `comics_1` passed
(1/1 = 100%, 3min wall). Solver: `src/docvqa/solvers/react_solver.py`;
config: `configs/solver/react.yaml` (commit `96246ca`).

User away; autonomous execution authorized. Heartbeat cron `3e93a103`
+ watcher `b8ba1tn0o` will do final pull/commit/push when done.

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

### 1. `[✓]` rvlm n=8 val — paired-comparison anchor for unified-tips (task #28)

Finished 2026-05-28T16:45 (c=24, run_ids `rvlm-val-t1`..`rvlm-val-t8`).
**n=8 mean = 40.94%, std 2.29pp, range 36.25%–43.75%.** Paired vs amax7
`rvlm_unified` n=8: **Δ = 0.00pp** (both arms 262/640 total correct),
95% CI [−3.91, +3.91]pp — lands cleanly in "promote unified to default"
per the pre-set decision table. Operational note: t6→t7 transition hit
a same-`run_id` contamination incident (chain's auto-t7 launched in
parallel with a standalone t7); recovered via clean restart on
`science_paper_1`. Runner's silent-timeout fallback was changed to
record-error-and-retry mid-chain (commit `8309710`). Full per-trial
tables, paired Δ stats, contamination story, and silent-failure audit
in [docs/experiments/unified-category-tips-ablation.md](../docs/experiments/unified-category-tips-ablation.md).

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
