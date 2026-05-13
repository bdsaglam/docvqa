#!/usr/bin/env bash
# Ceiling-pass baseline: no_loop_multi with max_pages=80 (matches the
# MMLongBench loader's DEFAULT_MAX_PAGES) so the raw-VLM baseline sees
# every rendered page. Tests how much of the +26pp scaffold lift is
# real vs an artifact of the standard max_pages=10 truncation.

set -uo pipefail
cd /home/baris/repos/docvqa

ENDPOINT=${ENDPOINT:-local}
if [ "$ENDPOINT" = "remote" ]; then
  MODEL_TAG=qwen-3_5-27b-vllm-remote
else
  MODEL_TAG=qwen-3_5-27b-vllm-local
fi

DATASET=yubo2333/MMLongBench-Doc
DOC_IDS_FILE=data/mmlongbench-doc/val/sample_200q_doc_ids.txt
TAG=mmlb

JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://localhost:8928/v1}
export QWEN_JUDGE_BASE_URL=$JUDGE_BASE_URL

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
  echo "=== [$ENDPOINT] no_loop_multi max_pages=80 trial $trial → $run_id ==="
  uv run python evals.py \
    data.dataset=$DATASET \
    data.split=val \
    data.num_samples=null \
    data.doc_ids_file=$DOC_IDS_FILE \
    lm=$MODEL_TAG \
    vlm=$MODEL_TAG \
    lm.enable_thinking=false \
    solver=no_loop_multi \
    solver.max_pages=80 \
    max_concurrency=4 \
    run_id=$run_id 2>&1 | tail -50
  echo "--- judging $run_id ---"
  uv run python scripts/judge_mmlongbench_run.py --run-id $run_id --concurrency 8 2>&1 | tail -20
}

summarize() {
  local run_id=$1
  if [ ! -f "output/runs/$run_id/results.json" ]; then
    echo "$run_id | NO RESULTS FILE"
    return
  fi
  uv run python -c "
import json, os
def read(p): return json.load(open(p)) if os.path.exists(p) else None
d = read('output/runs/$run_id/results.json') or {}
j = read('output/runs/$run_id/results-judged.json') or {}
s = d.get('summary') or {}
js = j.get('summary') or {}
acc_a = s.get('overall_accuracy'); acc_j = js.get('overall_accuracy_judge'); n = s.get('scored_questions') or 0
a = f'{acc_a:.4f}' if acc_a is not None else 'None'
jstr = f'{acc_j:.4f}' if acc_j is not None else 'None'
print(f'$run_id | anls={a} | judge={jstr} | n={n}')
"
}

echo "=== MMLongBench-Doc no_loop_multi max_pages=80 ceiling chain on $ENDPOINT start ==="
date
for i in 1 2 3; do
  RUN_ID=no-loop-multi-pages80-${TAG}-${ENDPOINT}-t$i
  run_one $i $RUN_ID
  out=$(summarize $RUN_ID)
  echo "$out"
  if [ -f "output/runs/$RUN_ID/results.json" ]; then
    notify "MMLB no_loop pages80 cell" "$out"
  fi
done
echo "=== Done ==="
date
notify "MMLB no_loop pages80 chain DONE" "3 cells finished"
