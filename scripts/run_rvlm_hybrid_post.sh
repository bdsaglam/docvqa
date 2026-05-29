#!/usr/bin/env bash
# Post-chain launcher: waits for the strip-chain (skeletal + naked) sentinel
# to land, then runs rvlm_hybrid n=1 immediately so the GPU doesn't idle.
#
# Uses default 14400s task_timeout (4h) per the rvlm_minimal lesson.

set -uo pipefail
cd /home/baris/repos/docvqa

LOG=/home/baris/repos/docvqa/output/runs/rvlm-strip-chain.log

note() { echo "[hybrid-post $(date +'%H:%M:%S')] $*" >> "$LOG"; }

note "waiting for strip-chain sentinel"
until [ -f /tmp/rvlm-strip-chain.done ]; do sleep 30; done
note "strip-chain done; launching rvlm-hybrid-val-t1"

rm -f /tmp/rvlm-hybrid.done

uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_hybrid \
  data.split=val data.num_samples=null \
  max_concurrency=32 \
  run_id=rvlm-hybrid-val-t1 >> "$LOG" 2>&1

note "rvlm-hybrid-val-t1 finished"
touch /tmp/rvlm-hybrid.done
note "DONE — wrote /tmp/rvlm-hybrid.done"
