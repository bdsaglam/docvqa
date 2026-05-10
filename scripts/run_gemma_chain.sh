#!/usr/bin/env bash
# Args: $1 = model tag (config name without configs/lm/ prefix and .yaml), $2 = short id (for run_id)
# Example: bash run_gemma_chain.sh gemma-4-e4b-vllm-local 4-e4b
set -uo pipefail
cd /home/baris/repos/docvqa

MODEL_TAG=${1:?need model tag}
SHORT=${2:?need short id}

echo "=== BASELINE: no_loop_multi + tips, 3 trials, model=$MODEL_TAG ==="
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

echo "=== SCAFFOLD: flat_solo lean m=30, 3 trials, model=$MODEL_TAG ==="
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
echo "=== ALL ${SHORT} TRIALS DONE ==="
