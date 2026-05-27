#!/usr/bin/env bash
# Model-axis chain template (D-008 escalation: defaults to TRIALS=1).
#
# Runs baseline (raw_vlm_multi) and proposed method (rvlm) on val for a
# given model. Use this template for the model-axis cells per D-006
# prediction 1.
#
# Args:
#   $1 = model tag (config name without configs/lm/ prefix and .yaml)
#   $2 = short id (for run_id)
# Env:
#   TRIALS = number of trials (default 1; escalate to 2 then 8 per D-008)
#   SOLVER = scaffold solver (default rvlm; rvlm_full is the kitchen-sink
#            legacy if you want to compare with historical flat_solo data)
#
# Examples:
#   bash run_gemma_chain.sh gemma-4-e4b-vllm-local 4-e4b
#   TRIALS=2 SOLVER=rvlm_ocr bash run_gemma_chain.sh qwen-3_5-9b-vllm-local 3_5-9b

set -uo pipefail
cd /home/baris/repos/docvqa

MODEL_TAG=${1:?need model tag}
SHORT=${2:?need short id}
TRIALS=${TRIALS:-1}
SOLVER=${SOLVER:-rvlm}

echo "=== BASELINE: raw_vlm_multi, ${TRIALS} trial(s), model=${MODEL_TAG} ==="
for i in $(seq 1 "${TRIALS}"); do
  echo "--- baseline t$i ---"
  uv run python evals.py \
    lm=$MODEL_TAG \
    vlm=$MODEL_TAG \
    lm.enable_thinking=false \
    solver=raw_vlm_multi \
    data.split=val data.num_samples=null \
    max_concurrency=8 \
    run_id=raw-vlm-multi-${SHORT}-val-t$i || { echo "BASELINE t$i FAILED"; exit 1; }
done
echo "=== BASELINE CHAIN DONE ==="

echo "=== SCAFFOLD: ${SOLVER}, ${TRIALS} trial(s), model=${MODEL_TAG} ==="
for i in $(seq 1 "${TRIALS}"); do
  echo "--- scaffold t$i ---"
  uv run python evals.py \
    lm=$MODEL_TAG \
    vlm=$MODEL_TAG \
    lm.enable_thinking=false \
    solver=${SOLVER} \
    data.split=val data.num_samples=null \
    max_concurrency=8 \
    run_id=${SOLVER}-${SHORT}-val-t$i || { echo "SCAFFOLD t$i FAILED"; exit 1; }
done
echo "=== SCAFFOLD CHAIN DONE ==="
echo "=== ALL ${SHORT} TRIALS DONE ==="
