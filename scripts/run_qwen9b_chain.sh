#!/usr/bin/env bash
set -uo pipefail
cd /home/baris/repos/docvqa

MODEL_TAG=qwen-3_5-9b-vllm-local
SHORT=3_5-9b

echo "=== BASELINE: no_loop_multi + tips, 3 trials ==="
for i in 1 2 3; do
  echo "--- baseline t$i ---"
  uv run python evals.py \
    lm=$MODEL_TAG \
    vlm=$MODEL_TAG \
    lm.enable_thinking=false \
    solver=no_loop_multi \
    data.split=val data.num_samples=null \
    max_concurrency=8 \
    run_id=no-loop-multi-tips-${SHORT}-val-t$i || { echo "BASELINE t$i FAILED"; exit 1; }
done
echo "=== BASELINE CHAIN DONE ==="

echo "=== SCAFFOLD: flat_solo lean m=30, 3 trials ==="
for i in 1 2 3; do
  echo "--- scaffold t$i ---"
  uv run python evals.py \
    lm=$MODEL_TAG \
    vlm=$MODEL_TAG \
    lm.enable_thinking=false \
    solver=flat_solo \
    data.split=val data.num_samples=null \
    max_concurrency=8 \
    run_id=flat-solo-${SHORT}-val-t$i || { echo "SCAFFOLD t$i FAILED"; exit 1; }
done
echo "=== SCAFFOLD CHAIN DONE ==="
echo "=== ALL QWEN 9B TRIALS DONE ==="
