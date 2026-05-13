#!/usr/bin/env bash
# Run MP-DocVQA evals: no_loop_multi × 3 trials, then leanest_solo × 3 trials.
# Default endpoint = local (8927). Override with ENDPOINT=remote for 8928.
#
# Usage:
#   scripts/run_mp_docvqa_chain.sh            # local (8927)
#   ENDPOINT=remote scripts/run_mp_docvqa_chain.sh   # remote (8928)
#
# Notification target: $NTFY_TOPIC must be set in the environment.

set -uo pipefail
cd /home/baris/repos/docvqa

ENDPOINT=${ENDPOINT:-local}
if [ "$ENDPOINT" = "remote" ]; then
  MODEL_TAG=qwen-3_5-27b-vllm-remote
else
  MODEL_TAG=qwen-3_5-27b-vllm-local
fi

DATASET=lmms-lab/MP-DocVQA
DOC_IDS_FILE=data/mp-docvqa/val/sample_200q_doc_ids.txt
TAG=mpdv

notify() {
  local title=$1
  local body=$2
  if [ -n "${NTFY_TOPIC:-}" ] && [ -x "$HOME/dotfiles/tools/notify.sh" ]; then
    "$HOME/dotfiles/tools/notify.sh" "$NTFY_TOPIC" "$title" "$body" || true
  fi
}

run_one() {
  local solver=$1
  local trial=$2
  local run_id=$3
  echo "=== [$ENDPOINT] $solver trial $trial → $run_id ==="
  uv run python evals.py \
    data.dataset=$DATASET \
    data.split=val \
    data.num_samples=null \
    data.doc_ids_file=$DOC_IDS_FILE \
    lm=$MODEL_TAG \
    vlm=$MODEL_TAG \
    lm.enable_thinking=false \
    solver=$solver \
    max_concurrency=8 \
    run_id=$run_id 2>&1 | tail -50
  local ec=$?
  if [ $ec -ne 0 ]; then
    notify "DocVQA chain FAILED" "$run_id exit=$ec on $ENDPOINT"
    echo "FAILED: $run_id"
  fi
  return $ec
}

summarize() {
  local run_id=$1
  if [ ! -f "output/runs/$run_id/results.json" ]; then
    echo "$run_id | NO RESULTS FILE"
    return
  fi
  uv run python -c "
import json
d = json.load(open('output/runs/$run_id/results.json'))
s = d['summary']
acc = s.get('overall_accuracy')
n = s.get('scored_questions') or 0
c = s.get('correct') or 0
if acc is None:
    print(f'$run_id | acc=None | n={n}')
else:
    print(f'$run_id | acc={acc:.4f} | correct={c}/{n}')
"
}

echo "=== MP-DocVQA chain on $ENDPOINT ($MODEL_TAG) start ==="
date

echo "--- BASELINE no_loop_multi ---"
for i in 1 2 3; do
  RUN_ID=no-loop-multi-${TAG}-${ENDPOINT}-t$i
  run_one no_loop_multi $i $RUN_ID
  out=$(summarize $RUN_ID)
  echo "$out"
  if [ -f "output/runs/$RUN_ID/results.json" ]; then
    notify "MP-DocVQA cell done" "$out"
  fi
done

echo "--- SCAFFOLD leanest_solo ---"
for i in 1 2 3; do
  RUN_ID=leanest-solo-${TAG}-${ENDPOINT}-t$i
  run_one leanest_solo $i $RUN_ID
  out=$(summarize $RUN_ID)
  echo "$out"
  if [ -f "output/runs/$RUN_ID/results.json" ]; then
    notify "MP-DocVQA cell done" "$out"
  fi
done

echo "=== MP-DocVQA chain on $ENDPOINT DONE ==="
date
notify "MP-DocVQA chain DONE" "endpoint=$ENDPOINT — all 6 cells finished"
