# Session registry (multi-session coordination)

Active work claims on this checkout. Update/remove your entry when done.

- **2026-07-02 02:27 (cron session, hourly 3x3-matrix watch):** claimed the
  `rvlm-27b-llm-9b-vlm-val` cell after its runner died (t2 stalled at 23/25
  since 23:51, no evals.py alive, GPUs idle). Running now: tmux
  `matrix-27b9b-t2` (resume, fills engineering_drawing_1) and
  `matrix-27b9b-t3` (fresh trial). t4 to follow. Do not double-launch these
  run_ids; if you take the queue back, kill these tmux sessions first and
  update this entry.
