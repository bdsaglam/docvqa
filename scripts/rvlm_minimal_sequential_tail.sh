#!/usr/bin/env bash
# Sequential tail for the rvlm_minimal n=8 chain after the overlap
# experiment caused per-doc timeouts in t2/t3/t4.
#
# Plan (refills prioritized to get full-denominator scores sooner):
#  1. Wait for t6 to finish (it's running solo; should be 80/80 clean).
#  2. Refill t2/t3/t4 missing docs FIRST. The runner is per-doc
#     resumable; deleting the stale results.json forces the runner to
#     regenerate it over the full 25-doc set after running the missing
#     2-3 docs. Once these land we have 6 complete trials (t1..t6) for
#     a paired-vs-unified read-out before the chain fully closes.
#  3. Launch t7 sequentially (no vllm contention).
#  4. Launch t8 sequentially.
#  5. Touch /tmp/rvlm-minimal-chain.done so the pre-existing watcher
#     fires and notifies.
#
# Use exec uv run ... >> LOG 2>&1 (no pipe-to-tee) so this script is
# safe to kill from outside without SIGPIPE'ing the python eval.

set -uo pipefail
cd /home/baris/repos/docvqa

LOG=/home/baris/repos/docvqa/output/runs/rvlm-minimal-chain.log

note() { echo "[seq-tail $(date +'%H:%M:%S')] $*" >> "$LOG"; }

run_solo() {
  local rid=$1
  note "launching ${rid} (task_timeout_seconds=1800)"
  uv run python evals.py \
    lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
    lm.enable_thinking=false \
    solver=rvlm_minimal \
    data.split=val data.num_samples=null \
    max_concurrency=32 \
    ++task_timeout_seconds=1800 \
    run_id="${rid}" >> "$LOG" 2>&1
  note "${rid} finished"
}

# --- 1. Refill incomplete trials with bumped per-doc timeout ---
# Trials t2/t3/t4 missed 2-3 docs each from the overlap-induced
# contention; t6 missed business_report_3 even running solo, suggesting
# the 600s timeout is too tight for some long docs regardless of
# contention. Refill all four with task_timeout_seconds=1800 (30 min).
#
# No need to delete the trial-level results.json: the runner's
# _load_completed() in src/docvqa/runner.py reads existing
# tasks/<doc>/result.json files and _compute_summary() aggregates over
# the FULL set (loaded + new), so re-running with the same run_id
# regenerates results.json over all docs it can see on disk.
for t in 2 3 4 6; do
  run_solo "rvlm-minimal-val-t${t}"
done

# --- 2. Sequential t7 + t8 ---
run_solo rvlm-minimal-val-t7
run_solo rvlm-minimal-val-t8

# --- 5. Sentinel ---
touch /tmp/rvlm-minimal-chain.done
note "DONE — wrote /tmp/rvlm-minimal-chain.done"
