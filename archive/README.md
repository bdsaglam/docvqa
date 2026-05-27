# Archive

Code, configs, scripts, and docs that are NOT part of the current paper
or the active codebase. Kept here for historical record and possible
reactivation; **not imported by anything in `src/docvqa/`**.

Reasons code lands here:

1. **Shelved approaches** that didn't pan out and are out of scope per
   D-006 (research-mode reframing). Examples: GEPA prompt optimization,
   multi-image VLM extension, pydantic-ai-rlm port.
2. **Pre-D-010 / pre-D-009 architecture** that has been superseded by
   the merged DA-by-default solver pattern.
3. **One-off exploratory work** kept for git-archeology purposes.

## Layout

```
archive/
├── README.md              # this file
├── configs/
│   └── solver/            # shelved solver Hydra configs
├── docs/                  # old design docs, exploratory writeups
│   └── superpowers/specs/ # shelved feature specs
├── scripts/               # old experiment scripts (chains, optimizers,
│                          # analysis utilities)
└── src/
    ├── solvers/           # shelved solver source files
    └── tools/             # old utility modules
```

The structure mirrors the live project (`src/`, `configs/`, `scripts/`,
`docs/`) so a moved file is easy to locate.

## Shelved solvers (D-006 reframing, 2026-05-27)

In `archive/src/solvers/` and `archive/configs/solver/`:

- `flat_batch_solver.py` / `flat_batch.yaml` — batch baseline; replaced
  by direct VLM calls + agent loop.
- `flat_solo_gepa_solver.py` / `flat_solo_gepa.yaml` — GEPA prompt
  optimization. Out of scope per D-006 (competition tactic, not paper
  contribution).
- `flat_solo_da_mi_solver.py` / `flat_solo_da_mi.yaml` — multi-image
  VLM extension. Single-trial regression; shelved per D-006.
- `lean_solo_solver.py` / `lean_solo.yaml` — intermediate solver with
  search + OCR but using `look()`. Redundant with `rvlm_ocr` post-D-010.
- `pyai_leanest_solo_da_solver.py` / `pyai_leanest_solo_da.yaml` —
  pydantic-ai port; 35.0% val vs dspy 43.8%. Shelved per D-006.
- `routing_solver.py` / `routing.yaml` — per-category routing dispatcher.
  Out of scope under the unified-tips ablation direction.

Plus older `archive/src/solvers/` entries (pre-D-006): `flat_batch_ocr`,
`flat_parallel`, `lean`, `orchestrator`, `parallel_rvlm`, `parallel_vlm`,
`perceive`, `structured_vlm`.

## Shelved scripts

In `archive/scripts/`: legacy run-chain helpers (voting, sweep, analysis,
optimize_flat_batch, run_pyai_evals, etc.).

## Reactivation

If you ever need to reactivate an archived solver:

1. Move the source file back to `src/docvqa/solvers/`.
2. Move the config back to `configs/solver/`.
3. Audit imports — archived files may still reference `docvqa.solvers.*`
   targets that have been renamed under D-010 (e.g.,
   `flat_solo_gepa_solver` imports from `rvlm_full_solver`).
4. Run smoke imports + a 2-doc eval.
5. Add an entry to `coordination/` for the experiment.
6. Update `docs/paper/decisions.md` if the reactivation reverses a
   prior decision.
