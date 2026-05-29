#!/usr/bin/env bash
# Refill skeletal (4 missing) + naked (1 missing) to round out full-set
# n=1 scores after the strip-chain. Hybrid was already 25/25.
#
# Sequential — single eval at a time so vllm has full headroom and we
# can give the long-tail (science_paper_1) the full 14400s timeout.

set -uo pipefail
cd /home/baris/repos/docvqa

LOG=/home/baris/repos/docvqa/output/runs/rvlm-strip-chain.log

note() { echo "[refill $(date +'%H:%M:%S')] $*" >> "$LOG"; }

run_solo() {
  local solver=$1
  local rid=$2
  note "launching ${rid} (solver=${solver})"
  uv run python evals.py \
    lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
    lm.enable_thinking=false \
    solver=${solver} \
    data.split=val data.num_samples=null \
    max_concurrency=32 \
    run_id="${rid}" >> "$LOG" 2>&1
  note "${rid} finished"
}

rm -f /tmp/rvlm-strip-refill.done

# Skeletal has 4 missing docs (including the hard science_paper_1).
# Naked has 1 missing doc.
run_solo rvlm_skeletal rvlm-skeletal-val-t1
run_solo rvlm_naked    rvlm-naked-val-t1

touch /tmp/rvlm-strip-refill.done
note "DONE — wrote /tmp/rvlm-strip-refill.done"
