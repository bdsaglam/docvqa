#!/usr/bin/env bash
# Re-run MMLongBench-Doc flat_solo with the dataset-aware profile.
# The profile injects per-question answer_format hints into the agent
# prompt and uses the Qwen judge for in-loop scoring (so results.json
# already contains judge accuracy — no post-hoc judge pass needed).
#
# Default endpoint = remote (8928). Set JUDGE_BASE_URL if the judge
# endpoint differs from QWEN_JUDGE_BASE_URL defaults (8928).

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
  local title=$1; local body=$2
  if [ -n "${NTFY_TOPIC:-}" ] && [ -x "$HOME/dotfiles/tools/notify.sh" ]; then
    echo "$body" | "$HOME/dotfiles/tools/notify.sh" "$NTFY_TOPIC" "$title" || true
  fi
}

run_one() {
  local trial=$1; local run_id=$2
  echo "=== [$ENDPOINT] flat_solo_da trial $trial → $run_id (judge=$JUDGE_BASE_URL) ==="
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
    max_concurrency=4 \
    run_id=$run_id 2>&1 | tail -60
}

summarize() {
  local run_id=$1
  if [ ! -f "output/runs/$run_id/results.json" ]; then
    echo "$run_id | NO RESULTS FILE"; return
  fi
  uv run python -c "
import json
d = json.load(open('output/runs/$run_id/results.json'))['summary']
acc = d.get('overall_accuracy'); n = d.get('scored_questions') or 0; c = d.get('correct') or 0
print(f'$run_id | judge={(acc or 0):.4f} | correct={c}/{n}')
"
}

echo "=== MMLongBench-Doc flat_solo_da chain on $ENDPOINT ($MODEL_TAG) start ==="
echo "Judge endpoint: $JUDGE_BASE_URL"
date
for i in 1 2 3; do
  RUN_ID=flat-solo-da-${TAG}-${ENDPOINT}-t$i
  run_one $i $RUN_ID
  out=$(summarize $RUN_ID)
  echo "$out"
  if [ -f "output/runs/$RUN_ID/results.json" ]; then
    notify "MMLongBench flat_solo_da cell done" "$out"
  fi
done
echo "=== chain DONE ==="
date
notify "MMLongBench flat_solo_da chain DONE" "endpoint=$ENDPOINT — 3 cells finished"
