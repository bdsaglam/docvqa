#!/usr/bin/env bash
# rvlm_minimal refill pass 2.
#
# After tail-v2 finished, t2/t4/t8 are missing science_paper_1 and t7 is
# missing both science_paper_1 and slide_2. science_paper_1 has a 4/8
# per-attempt completion rate (the other half hits the 14400s task
# timeout on what looks like an agent loop / vllm hang).
#
# Strategy: serial refills (so all four attempts of science_paper_1 don't
# compete for vllm), default 14400s timeout, then re-touch the sentinel.

set -uo pipefail
cd /home/baris/repos/docvqa

LOG=/home/baris/repos/docvqa/output/runs/rvlm-minimal-chain.log

note() { echo "[refill-p2 $(date +'%H:%M:%S')] $*" >> "$LOG"; }

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

# Removing the previous sentinel so the watcher signals on this pass.
rm -f /tmp/rvlm-minimal-chain.done

note "refill pass 2: t2 t4 t7 t8 (sequential)"
run_solo rvlm-minimal-val-t2
run_solo rvlm-minimal-val-t4
run_solo rvlm-minimal-val-t7
run_solo rvlm-minimal-val-t8

touch /tmp/rvlm-minimal-chain.done
note "DONE — wrote /tmp/rvlm-minimal-chain.done"
