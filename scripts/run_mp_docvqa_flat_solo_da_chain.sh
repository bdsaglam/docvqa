#!/usr/bin/env bash
# Re-run MP-DocVQA flat_solo with the dataset-aware profile + new
# scoring path. Compares against the existing flat-solo-mpdv-local-t*
# cells (which used the DocVQA-2026 default answer-formatting rules
# and category tips).
#
# Default endpoint = remote (8928) so the chain runs in parallel with
# anything on local. Override with ENDPOINT=local.

set -uo pipefail
cd /home/baris/repos/docvqa

ENDPOINT=${ENDPOINT:-remote}
if [ "$ENDPOINT" = "remote" ]; then
  MODEL_TAG=qwen-3_5-27b-vllm-remote
else
  MODEL_TAG=qwen-3_5-27b-vllm-local
fi

DATASET=lmms-lab/MP-DocVQA
DOC_IDS_FILE=data/mp-docvqa/val/sample_200q_doc_ids.txt
TAG=mpdv

notify() {
  local title=$1; local body=$2
  if [ -n "${NTFY_TOPIC:-}" ] && [ -x "$HOME/dotfiles/tools/notify.sh" ]; then
    echo "$body" | "$HOME/dotfiles/tools/notify.sh" "$NTFY_TOPIC" "$title" || true
  fi
}

run_one() {
  local trial=$1; local run_id=$2
  echo "=== [$ENDPOINT] flat_solo_da trial $trial → $run_id ==="
  uv run python evals.py \
    data.dataset=$DATASET \
    data.split=val \
    data.num_samples=null \
    data.doc_ids_file=$DOC_IDS_FILE \
    data.use_profile_scoring=true \
    lm=$MODEL_TAG \
    vlm=$MODEL_TAG \
    lm.enable_thinking=false \
    solver=flat_solo_da \
    max_concurrency=8 \
    run_id=$run_id 2>&1 | tail -50
}

summarize() {
  local run_id=$1
  if [ ! -f "output/runs/$run_id/results.json" ]; then
    echo "$run_id | NO RESULTS FILE"; return
  fi
  uv run python -c "
import json
d = json.load(open('output/runs/$run_id/results.json'))
s = d['summary']
acc = s.get('overall_accuracy'); n = s.get('scored_questions') or 0; c = s.get('correct') or 0
print(f'$run_id | acc={(acc or 0):.4f} | correct={c}/{n}')
"
}

echo "=== MP-DocVQA flat_solo_da chain on $ENDPOINT ($MODEL_TAG) start ==="
date
for i in 1 2 3; do
  RUN_ID=flat-solo-da-${TAG}-${ENDPOINT}-t$i
  run_one $i $RUN_ID
  out=$(summarize $RUN_ID)
  echo "$out"
  if [ -f "output/runs/$RUN_ID/results.json" ]; then
    notify "MP-DocVQA flat_solo_da cell done" "$out"
  fi
done
echo "=== chain DONE ==="
date
notify "MP-DocVQA flat_solo_da chain DONE" "endpoint=$ENDPOINT — 3 cells finished"
