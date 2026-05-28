#!/usr/bin/env bash
# rvlm_minimal tail v2: parallel refills, then sequential t7/t8.
#
# Refills (t2/t3/t4/t6) each only need to process 1-2 missing docs, so
# running them sequentially leaves the GPU mostly idle. Run them all in
# parallel — peak concurrent VLM requests is bounded by
# (#trials × question_concurrency) ≈ 4 × 4 = 16, well below the c=32 cap
# we know vllm handles fine for a single full trial.
#
# After all refills land, run t7 + t8 sequentially (they're full c=32
# trials and the contention story bites us at 2 of those in flight).
#
# Per-doc timeout: leaves the default (14400s = 4h, from configs/config.yaml)
# in place. t1's data showed long docs taking up to 85 min (infographics_1
# at 5097s, business_report_4 at 4468s, etc.), so a 30-min override
# we tried earlier was way too tight and killed every science_paper_1
# attempt.

set -uo pipefail
cd /home/baris/repos/docvqa

LOG=/home/baris/repos/docvqa/output/runs/rvlm-minimal-chain.log

note() { echo "[tail-v2 $(date +'%H:%M:%S')] $*" >> "$LOG"; }

launch_bg() {
  local rid=$1
  note "launching ${rid} (bg)"
  ( uv run python evals.py \
      lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
      lm.enable_thinking=false \
      solver=rvlm_minimal \
      data.split=val data.num_samples=null \
      max_concurrency=32 \
        run_id="${rid}" >> "$LOG" 2>&1
    note "${rid} finished" ) &
}

run_solo() {
  local rid=$1
  note "launching ${rid} (fg)"
  uv run python evals.py \
    lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
    lm.enable_thinking=false \
    solver=rvlm_minimal \
    data.split=val data.num_samples=null \
    max_concurrency=32 \
    run_id="${rid}" >> "$LOG" 2>&1
  note "${rid} finished"
}

# --- 1. Parallel refills t2/t3/t4/t6 ---
note "starting parallel refill phase: t2 t3 t4 t6"
launch_bg rvlm-minimal-val-t2
launch_bg rvlm-minimal-val-t3
launch_bg rvlm-minimal-val-t4
launch_bg rvlm-minimal-val-t6
wait
note "parallel refill phase done"

# --- 2. Sequential t7 + t8 (full trials; avoid contention) ---
run_solo rvlm-minimal-val-t7
run_solo rvlm-minimal-val-t8

# --- 3. Sentinel ---
touch /tmp/rvlm-minimal-chain.done
note "DONE — wrote /tmp/rvlm-minimal-chain.done"
