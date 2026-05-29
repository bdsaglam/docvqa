#!/usr/bin/env bash
# rvlm strip chain: skeletal n=1 → naked n=1.
#
# Tests how far the rvlm prompt can be stripped before the score
# drops. Skeletal first (drops the 3 doc-shape patterns); if that
# holds, naked next (drops everything except DATA + TOOLS +
# faithfulness + OUTPUT FORMAT).
#
# Sequential — one cell at a time per D-008 / amax7 coord convention.

set -uo pipefail
cd /home/baris/repos/docvqa

LOG=/home/baris/repos/docvqa/output/runs/rvlm-strip-chain.log

note() { echo "[strip-chain $(date +'%H:%M:%S')] $*" >> "$LOG"; }

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

rm -f /tmp/rvlm-strip-chain.done

run_solo rvlm_skeletal rvlm-skeletal-val-t1
run_solo rvlm_naked    rvlm-naked-val-t1

touch /tmp/rvlm-strip-chain.done
note "DONE — wrote /tmp/rvlm-strip-chain.done"
