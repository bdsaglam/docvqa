#!/usr/bin/env bash
# Run MMLongBench-Doc flat_solo (lean) × 3 trials on the remote endpoint (8928).
# Each cell is auto-judged with qwen_judge (judge endpoint: local 8927).
# Requires OCR data at data/mmlongbench-doc/val/ocr/<doc_id>/page_*.md.

set -uo pipefail
cd /home/baris/repos/docvqa

ENDPOINT=${ENDPOINT:-remote}
if [ "$ENDPOINT" = "remote" ]; then
  MODEL_TAG=qwen-3_5-27b-vllm-remote
  JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://localhost:8927/v1}
else
  MODEL_TAG=qwen-3_5-27b-vllm-local
  JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://localhost:8928/v1}
fi
export QWEN_JUDGE_BASE_URL=$JUDGE_BASE_URL

DATASET=yubo2333/MMLongBench-Doc
DOC_IDS_FILE=data/mmlongbench-doc/val/sample_200q_doc_ids.txt
TAG=mmlb

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
    max_concurrency=4 \
    run_id=$run_id 2>&1 | tail -60
  local ec=$?
  if [ $ec -ne 0 ]; then
    notify "MMLongBench flat_solo FAILED" "$run_id exit=$ec on $ENDPOINT"
    echo "FAILED: $run_id"
  fi

  echo "--- judging $run_id ---"
  uv run python scripts/judge_mmlongbench_run.py --run-id $run_id --concurrency 8 2>&1 | tail -30
  return $ec
}

summarize() {
  local run_id=$1
  if [ ! -f "output/runs/$run_id/results.json" ]; then
    echo "$run_id | NO RESULTS FILE"
    return
  fi
  uv run python -c "
import json, os
def read(p):
    return json.load(open(p)) if os.path.exists(p) else None
d = read('output/runs/$run_id/results.json') or {}
j = read('output/runs/$run_id/results-judged.json') or {}
s = d.get('summary') or {}
js = j.get('summary') or {}
acc_a = s.get('overall_accuracy')
acc_j = js.get('overall_accuracy_judge')
n = s.get('scored_questions') or 0
a_str = f'{acc_a:.4f}' if acc_a is not None else 'None'
j_str = f'{acc_j:.4f}' if acc_j is not None else 'None'
print(f'$run_id | anls={a_str} | judge={j_str} | n={n}')
"
}

echo "=== MMLongBench-Doc flat_solo chain on $ENDPOINT ($MODEL_TAG) start ==="
echo "Judge endpoint: $JUDGE_BASE_URL"
date

for i in 1 2 3; do
  RUN_ID=flat-solo-${TAG}-${ENDPOINT}-t$i
  run_one $i $RUN_ID
  out=$(summarize $RUN_ID)
  echo "$out"
  if [ -f "output/runs/$RUN_ID/results.json" ]; then
    notify "MMLongBench flat_solo cell done" "$out"
  fi
done

echo "=== MMLongBench-Doc flat_solo chain on $ENDPOINT DONE ==="
date
notify "MMLongBench-Doc flat_solo chain DONE" "endpoint=$ENDPOINT — 3 cells finished"
