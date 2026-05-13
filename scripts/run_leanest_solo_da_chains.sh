#!/usr/bin/env bash
# Dataset-aware leanest_solo on MP-DocVQA + MMLongBench-Doc.
# Runs the no-OCR scaffold with the profile prompt + judge so we can
# isolate (a) scaffold-vs-baseline and (b) OCR's incremental
# contribution (flat_solo_da minus leanest_solo_da).
#
# Endpoint defaults to local (8927). Override with ENDPOINT=remote.

set -uo pipefail
cd /home/baris/repos/docvqa

ENDPOINT=${ENDPOINT:-local}
if [ "$ENDPOINT" = "remote" ]; then
  MODEL_TAG=qwen-3_5-27b-vllm-remote
  JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://localhost:8927/v1}
else
  MODEL_TAG=qwen-3_5-27b-vllm-local
  JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://localhost:8928/v1}
fi
export QWEN_JUDGE_BASE_URL=$JUDGE_BASE_URL

notify() {
  local title=$1; local body=$2
  if [ -n "${NTFY_TOPIC:-}" ] && [ -x "$HOME/dotfiles/tools/notify.sh" ]; then
    echo "$body" | "$HOME/dotfiles/tools/notify.sh" "$NTFY_TOPIC" "$title" || true
  fi
}

summarize() {
  local run_id=$1
  if [ ! -f "output/runs/$run_id/results.json" ]; then
    echo "$run_id | NO RESULTS FILE"; return
  fi
  uv run python -c "
import json
s = json.load(open('output/runs/$run_id/results.json'))['summary']
acc = s.get('overall_accuracy'); n = s.get('scored_questions') or 0; c = s.get('correct') or 0
print(f'$run_id | acc={(acc or 0):.4f} | correct={c}/{n}')
"
}

run_cell() {
  local dataset=$1; local sample=$2; local concur=$3; local trial=$4; local run_id=$5
  echo "=== [$ENDPOINT] $dataset leanest_solo_da trial $trial → $run_id ==="
  uv run python evals.py \
    data.dataset=$dataset \
    data.split=val \
    data.num_samples=null \
    data.doc_ids_file=$sample \
    data.use_profile_scoring=true \
    lm=$MODEL_TAG \
    vlm=$MODEL_TAG \
    lm.enable_thinking=false \
    solver=leanest_solo_da \
    max_concurrency=$concur \
    run_id=$run_id 2>&1 | tail -60
}

echo "=== leanest_solo_da chains on $ENDPOINT ($MODEL_TAG) ==="
date

echo "--- MP-DocVQA ---"
for i in 1 2 3; do
  RUN_ID=leanest-solo-da-mpdv-${ENDPOINT}-t$i
  run_cell lmms-lab/MP-DocVQA data/mp-docvqa/val/sample_200q_doc_ids.txt 8 $i $RUN_ID
  out=$(summarize $RUN_ID); echo "$out"
  [ -f "output/runs/$RUN_ID/results.json" ] && notify "MP-DocVQA leanest_solo_da cell done" "$out"
done

echo "--- MMLongBench-Doc ---"
for i in 1 2 3; do
  RUN_ID=leanest-solo-da-mmlb-${ENDPOINT}-t$i
  run_cell yubo2333/MMLongBench-Doc data/mmlongbench-doc/val/sample_200q_doc_ids.txt 4 $i $RUN_ID
  out=$(summarize $RUN_ID); echo "$out"
  [ -f "output/runs/$RUN_ID/results.json" ] && notify "MMLongBench leanest_solo_da cell done" "$out"
done

echo "=== leanest_solo_da chains DONE ==="
date
notify "leanest_solo_da chains DONE" "endpoint=$ENDPOINT — 6 cells across MP-DocVQA + MMLongBench-Doc"
