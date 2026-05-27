#!/usr/bin/env bash
set -uo pipefail
cd /home/baris/repos/docvqa
LOG_DIR=/home/baris/repos/docvqa/logs
mkdir -p "$LOG_DIR"

# Disable logfire telemetry export so the runner doesn't hang on flush.
export LOGFIRE_SEND_TO_LOGFIRE=false
export LOGFIRE_IGNORE_NO_CONFIG=1

for TRIAL in 1 2 3; do
  RUN_ID="pyai-leanest-val-t${TRIAL}"
  LOG="${LOG_DIR}/${RUN_ID}.log"
  echo "=== Trial ${TRIAL} starting at $(date -u +%H:%M:%S) ===" | tee -a "$LOG"
  uv run python evals.py \
    lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
    lm.enable_thinking=false \
    solver=pyai_leanest_solo_da \
    data.split=val data.num_samples=null \
    max_concurrency=4 \
    run_id="$RUN_ID" 2>&1 | tee -a "$LOG"
  echo "=== Trial ${TRIAL} EXIT_$? at $(date -u +%H:%M:%S) ===" | tee -a "$LOG"
done
echo "=== ALL 3 TRIALS DONE at $(date -u +%H:%M:%S) ==="
