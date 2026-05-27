# Host coordination

Two hosts share this repo: **amax7** (the host running you/Claude on the
adaptive side) and **amax1** (the secondary host). To avoid stepping on
each other's work, each host owns one queue file. The agent or operator
on each host picks an experiment from their file, runs it, marks it
done, commits + pushes, then the other host pulls before its next pick.

## Files

- [`amax7.md`](amax7.md) — **amax7** = adaptive host. Runs critical-path
  experiments where the result might change the experiment plan. Tighter
  feedback loops; one cell at a time; replan after each result.
- [`amax1.md`](amax1.md) — **amax1** = throughput host. Runs side-track
  experiments where the direction is already known and we just need to
  lock numbers. No adaptive iteration; if a cell shows an unexpected
  direction, **halt and write a note here** for amax7 to triage.
- [`cleanup-runs.md`](cleanup-runs.md) — procedure for deleting obsolete
  `output/runs/` dirs to free disk. amax7 ran this 2026-05-28 and freed
  39GB. amax1 should do the same when disk pressure builds.

## Workflow (per unit of work)

A unit of work = one experiment cell (one solver × one split × n=1).

1. **Pull latest:** `git pull --rebase`
2. **Open your host file.** Pick the first cell marked `[ ]` (queued).
3. **Mark it `[→]`** with an ISO timestamp and (optionally) PID/tmux session.
4. **Commit + push** the host file. This advertises the lock so the other
   host doesn't accidentally duplicate.
5. **Run the experiment** (foreground or tmux background).
6. **When the cell completes:** mark it `[✓]` with run_id and a one-line
   result (or `[✗]` with the failure mode if it crashed).
7. **Commit + push** the updated host file + any new run artifacts you
   want shared (run dirs in `output/runs/` are gitignored — share via
   the result line; submission JSONs and experiment docs are committed
   if applicable).
8. **Loop to step 1.**

If your queue is empty: pull, check the other host's queue for stuck
items, otherwise stop and write a `# QUEUE EMPTY` note at the bottom of
your host file.

## Status legend

- `[ ]` queued — not started
- `[→]` in progress — has start timestamp; one cell at a time per host
- `[✓]` done — has run_id and one-line result
- `[✗]` failed — has failure mode and timestamp
- `[~]` deferred — explicitly punted to later (e.g., waiting on upstream)

## Conventions

- **One in-progress cell per host at a time.** Don't pre-lock multiple
  cells; that creates merge conflicts when both hosts edit the same file.
- **Trial-budget escalation per D-008.** Cells start at n=1. If the n=1
  result matches expectations, the *queue owner* adds n=2 as a follow-up
  cell when they file the result; n=8 only after the paper headline
  locks.
- **Side-tracks for amax1, critical path for amax7.** If amax1 hits an
  unexpected result on a side-track, halt and append `## NOTE FOR AMAX7`
  to its file with the surprise. amax7 reads on its next pull and
  decides whether to redirect.
- **Commit messages:** `coord: <host> <action> <cell-name>` e.g.,
  `coord: amax1 done gemma-e4b-baseline-val-t1`.
- **Run IDs** in the new naming scheme use the new solver names. E.g.,
  `rvlm-unified-val-t1`, `rvlm-ocr-val-t1`, `raw-vlm-multi-gemma-e4b-val-t1`.
  Historical run IDs (`leanest-*`, `flat-solo-*`, etc.) stay as they
  were; new runs use new names.

## When to update vs replace this file

- This README is the protocol — change it if the workflow itself changes
  (rare).
- `amax7.md` and `amax1.md` are turnover; edit each unit of work.
