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
| [flat-solo-turn-budget-sweep.md](flat-solo-turn-budget-sweep.md) | done (5 cells; m=10/20/30/40 × 8 trials, m=5 × 3 trials) |
| [flat-solo-category-tips-off.md](flat-solo-category-tips-off.md) | done (8 clean trials) |
| [flat-solo-vlm-cropping-off.md](flat-solo-vlm-cropping-off.md) | done (8 trials) |
| [leanest-turn-budget-sweep.md](leanest-turn-budget-sweep.md) | done (4 cells × 8 trials; peak m=40) |
| [flat-solo-search-off.md](flat-solo-search-off.md) | done (8 trials; n.s. effect) |
| [no-loop-baseline.md](no-loop-baseline.md) | done (3 trials tips-on + 3 trials tips-off) |
| [leanest-ocr-off.md](leanest-ocr-off.md) | done (3 clean trials, comparisons updated for fair baselines) |
| [no-loop-multi-image.md](no-loop-multi-image.md) | done (3 trials tips-on + 3 trials tips-off) |
| [efficiency-summary.md](efficiency-summary.md) | cross-cell turns-per-question summary (12 local cells, pooled 8×80q each) |
| [per-doc-flat-vs-leanest.md](per-doc-flat-vs-leanest.md) | per-doc comparison of best flat_solo vs best leanest configs (OCR's role by doc length & category) |
| [flat-solo-test-matched-baseline.md](flat-solo-test-matched-baseline.md) | done (8 trials); SC-8 voted test = 38.75% (ICDAR) |
| [qwen-9b-baseline-scaffold.md](qwen-9b-baseline-scaffold.md) | done (3+3 val trials, lift +6.25pp) |
| [gemma-4-e4b-baseline-scaffold.md](gemma-4-e4b-baseline-scaffold.md) | done (3+3 val trials, lift +5.83pp) |
| [gemma-4-31b-baseline-scaffold.md](gemma-4-31b-baseline-scaffold.md) | done (3+3 val trials, lift +25.00pp; vllm triage required) |
| [mp-docvqa-qwen27b.md](mp-docvqa-qwen27b.md) | done (9 legacy + 9 DA val trials. After DA: all 3 solvers ~73-74% ANLS; scaffold delta = 0. Legacy "regression" was prompt mismatch.) |
| [mmlongbench-doc-qwen27b.md](mmlongbench-doc-qwen27b.md) | done (9 legacy + 3 ceiling + 9 DA val trials. Legacy lift +26.43pp; fair lift +16.84pp judge. About half the headline was baseline crippling.) |
