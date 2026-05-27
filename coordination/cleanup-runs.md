# Cleaning up `output/runs/` (run-dir disk cleanup)

`output/runs/` accumulates fast — each Qwen 27B full-val run is ~289MB,
each smoke ~52MB, and exploratory date-stamped runs add up. Periodic
cleanup keeps the host disk usable without losing paper-relevant data.

On amax7 we went from **46GB to 7GB (39GB freed)** by deleting the
categories below. amax1 can do the same.

## Safe-delete categories

Each of these can be deleted with high confidence — the
paper-relevant data is preserved elsewhere (submission JSONs in
`submissions/`, experiment writeups in `docs/experiments/`, decision
log in `docs/paper/decisions.md`).

| Category | Pattern | Reason |
|---|---|---|
| Date-stamped exploratory | `2026-*` | Hydra's default run-dir when no `run_id=` is set. Almost always failed-early or transient. |
| Smoke runs | `*smoke*` | 2-4 doc exploratory runs; never paper-cited. |
| Shelved-experiment runs | `pyai-*`, `flat-solo-da-mi*`, `mi-*` | Shelved per D-006; conclusions documented in `archive/docs/experiments/`. |
| Prompt-scrub runs | `*scrub*` | Scrub-audit archived; per-trial scores in `archive/docs/experiments/scrub-audit.md`; SC-8 votes preserved in `submissions/*-scrub-sc8.json`. |
| Model-axis pre-rerun cells | `*4-e4b*`, `*3_5-9b*`, `*4-31b*` | Used pre-D-009 prompts; re-runs scheduled per coordination/amax1.md (task #8). Direction documented in `docs/experiments/{gemma-4-e4b,qwen-9b,gemma-4-31b}-baseline-scaffold.md`. |
| Legacy cross-benchmark cells (non-DA) | `flat-solo-mmlb-remote-*`, `flat-solo-mpdv-local-*`, `leanest-solo-mmlb-remote-*`, `leanest-solo-mpdv-local-*`, `no-loop-multi-mmlb-remote-*`, `no-loop-multi-mpdv-local-*`, `no-loop-multi-pages80-mmlb-local-*` | Used DocVQA-2026 prompts mis-applied; the clean numbers come from the DA cells. Legacy headline was inflated by prompt-mismatch — documented. |

## Keep (do NOT delete)

- **Currently running cells.** Cross-check with `coordination/<host>.md`
  for any `[→]` cells. The run dir is the only place the eval is
  writing to.
- **Paper-cell anchors** (current evidence in `docs/results.md`):
  - `rvlm-*-val-t{1..8}` / `rvlm-*-test-t{1..8}` (new naming)
  - `*-val-t{1..8}` and `*-test-t{1..8}` referenced in
    `docs/experiments/{leanest,flat-solo}-test-matched-baseline.md`
    (old naming retained per D-010)
- **DA cross-benchmark cells**: `*-da-mmlb-*`, `*-da-mpdv-*` — the
  clean (DA-profile) cells. These are prediction-2 evidence.
- **Raw-VLM baselines**: `no-loop-multi-val-t{1..3}`,
  `no-loop-val-t{1..3}`, `no-loop-multi-tips-val-t{1..3}`,
  `no-loop-tips-val-t{1..3}`, and `no-loop-multi-3_5-27b-{val,test}-t{1..8}`
  (split-calibration anchor — predictions in `submissions/` if dirs
  are gone).
- **Turn-budget points**: `flat-solo-m5-val-t{1..3}` (and any other
  m-budget cells referenced in `flat-solo-turn-budget-sweep.md` /
  `leanest-turn-budget-sweep.md`).

## Procedure

```bash
cd /home/baris/repos/docvqa/output/runs

# 1. Take a "before" disk-usage snapshot.
before=$(du -sm . | cut -f1)

# 2. Sanity-check: confirm no in-progress cells from coordination/<host>.md
#    are about to be deleted. List your host's [→] cells:
grep -A1 '`\[→\]`' ../coordination/<host>.md | head -20

# 3. Delete safe categories (skip patterns that match your in-progress cells).
rm -rf 2026-*/ *smoke*/ pyai-*/ flat-solo-da-mi*/ mi-*/
rm -rf *scrub*/
rm -rf *4-e4b*/ *3_5-9b*/ *4-31b*/
rm -rf flat-solo-mmlb-remote-*/ flat-solo-mpdv-local-*/ \
       leanest-solo-mmlb-remote-*/ leanest-solo-mpdv-local-*/ \
       no-loop-multi-mmlb-remote-*/ no-loop-multi-mpdv-local-*/ \
       no-loop-multi-pages80-mmlb-local-*/

# 4. Report what was freed.
after=$(du -sm . | cut -f1)
echo "freed: $((before-after))MB ($(((before-after)/1024))GB)"
echo "remaining: $(ls | wc -l) dirs"
```

## What to do with unfamiliar dirs

If you see a dir whose name you don't recognize:

1. Check `ls <dir>` — does it have `results.json` and a non-empty
   `tasks/` subdir? If neither, it's almost certainly a failed-early
   run; safe to delete.
2. Grep `docs/experiments/` and `submissions/` for the run-id stem.
   If nothing references it, it's not paper-cited.
3. If in doubt, leave it. Disk is cheap; re-running is expensive.

## When NOT to clean up

- **During an active multi-cell chain.** If `coordination/<host>.md`
  has multiple `[→]` cells or back-to-back queued cells, the next cell
  might be about to start; pattern-matched `rm -rf` could race.
- **Just before pulling.** Pull first so you know what new run dirs the
  other host may have committed metadata for.

## Reactivation if you deleted something needed

Every paper-cell number is preserved in at least one of:

- `submissions/*.json` (committed predictions)
- `docs/experiments/*.md` (per-trial scores in markdown tables)
- `docs/results.md` (headline numbers)
- `archive/docs/experiments/*.md` (archived experiments)

If you need the raw trajectories or per-question logs back, re-run the
cell. With clean prompts and fixed seeds the re-run matches within
trial-noise (~3–4pp at Qwen 27B).
