#!/usr/bin/env bash
# Run MP-DocVQA flat_solo (lean) × 3 trials on the local endpoint (8927).
# Requires OCR data at data/mp-docvqa/val/ocr/<doc_id>/page_*.md.

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
    echo "$body" | "$HOME/dotfiles/tools/notify.sh" "$NTFY_TOPIC" "$title" || true
  fi
}

run_one() {
  local trial=$1
  local run_id=$2
  echo "=== [$ENDPOINT] flat_solo trial $trial → $run_id ==="
  uv run python evals.py \
    data.dataset=$DATASET \
    data.split=val \
    data.num_samples=null \
    data.doc_ids_file=$DOC_IDS_FILE \
    lm=$MODEL_TAG \
    vlm=$MODEL_TAG \
    lm.enable_thinking=false \
    solver=flat_solo \
    solver.rlm_type=lean \
    max_concurrency=8 \
    run_id=$run_id 2>&1 | tail -50
  local ec=$?
  if [ $ec -ne 0 ]; then
    notify "MP-DocVQA flat_solo FAILED" "$run_id exit=$ec on $ENDPOINT"
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

echo "=== MP-DocVQA flat_solo chain on $ENDPOINT ($MODEL_TAG) start ==="
date

for i in 1 2 3; do
  RUN_ID=flat-solo-${TAG}-${ENDPOINT}-t$i
  run_one $i $RUN_ID
  out=$(summarize $RUN_ID)
  echo "$out"
  if [ -f "output/runs/$RUN_ID/results.json" ]; then
    notify "MP-DocVQA flat_solo cell done" "$out"
  fi
done

echo "=== MP-DocVQA flat_solo chain on $ENDPOINT DONE ==="
date
notify "MP-DocVQA flat_solo chain DONE" "endpoint=$ENDPOINT — 3 cells finished"
