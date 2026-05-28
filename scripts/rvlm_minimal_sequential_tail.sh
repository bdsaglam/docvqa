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
  note "launching ${rid}"
  uv run python evals.py \
    lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
    lm.enable_thinking=false \
    solver=rvlm_minimal \
    data.split=val data.num_samples=null \
    max_concurrency=32 \
    run_id="${rid}" >> "$LOG" 2>&1
  note "${rid} finished"
}

# --- 1. Wait for t6 to finish ---
note "waiting for t6 to finalize results.json"
while [ ! -f output/runs/rvlm-minimal-val-t6/results.json ]; do
  sleep 60
done
note "t6 done"

# --- 2. Refill t2/t3/t4 missing docs (prioritized for fast partial readout) ---
# Delete the trial-level results.json so the runner regenerates it over
# the FULL doc set when it re-runs the missing per-doc results.
for t in 2 3 4; do
  rid=rvlm-minimal-val-t${t}
  d=output/runs/${rid}
  if [ -f "${d}/results.json" ]; then
    note "removing stale ${d}/results.json to allow refill regeneration"
    rm "${d}/results.json"
  fi
  run_solo "${rid}"
done

# --- 3-4. Sequential t7 + t8 ---
run_solo rvlm-minimal-val-t7
run_solo rvlm-minimal-val-t8

# --- 5. Sentinel ---
touch /tmp/rvlm-minimal-chain.done
note "DONE — wrote /tmp/rvlm-minimal-chain.done"
