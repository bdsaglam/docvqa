# Experiments

One file per experiment / ablation cell. Each file is the canonical
record for that cell — per-trial data + summary + caveats — so future
sessions don't have to re-derive numbers from `output/runs/`.

## File layout

Each experiment file has these sections:

1. **Hypothesis / question** — what we're testing, in one sentence.
2. **Command** — the exact CLI, copy-pasteable.
3. **Per-trial table** — `run_id`, score, correct/total, sandbox errors,
   wall time when relevant. One row per trial.
4. **Summary** — mean ± std, n, range. Note any excluded trials.
5. **Comparison** — gap vs baseline + standard error. Be explicit about
   what the baseline is.
6. **Observations / caveats** — what was surprising, what to be careful
   about, infra issues, links to memory entries.
7. **Status** — `in progress`, `done`, or `done — needs replication`.

## Conventions

- Run IDs follow `<solver-tag>-<model>-<split>-t<trial>`. Example:
  `flat-solo-no-tips-3_5-27b-val-t3`.
- 8 trials is the standard cell size; 3 is the floor for any headline
  number.
- Sandbox-error count is computed via `grep -ho "Subprocess is not
  running" output/runs/<run_id>/tasks/*/result.json | wc -l`. Any trial
  with >100 errors is contaminated and **excluded** (see
  `sandbox_subprocess_errors` memory).
- When a sweep has multiple cells (e.g. turn budget points), put all
  cells in one file with one section per cell — they're a single
  experiment.

## Index

| File | Status |
|---|---|
| [flat-solo-turn-budget-sweep.md](flat-solo-turn-budget-sweep.md) | done (4 cells × 8 trials) |
| [flat-solo-category-tips-off.md](flat-solo-category-tips-off.md) | done (8 clean trials) |
| [flat-solo-vlm-cropping-off.md](flat-solo-vlm-cropping-off.md) | done (8 trials) |
| [leanest-turn-budget-sweep.md](leanest-turn-budget-sweep.md) | in progress (m=30, m=50) |
