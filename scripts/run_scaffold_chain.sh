#!/usr/bin/env bash
# Scaffold-only (flat_solo) chain. Args same as run_gemma_chain.sh.
set -uo pipefail
cd /home/baris/repos/docvqa
MODEL_TAG=${1:?need model tag}
SHORT=${2:?need short id}

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
echo "=== ALL ${SHORT} SCAFFOLD TRIALS DONE ==="
