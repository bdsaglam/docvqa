#!/usr/bin/env bash
# Dataset-aware baseline: no_loop_multi_da on MP-DocVQA + MMLongBench-Doc.
#
# Same prompt/format/scoring profile that flat_solo_da uses, so the
# baseline-vs-scaffold delta isolates the scaffold's contribution rather
# than a prompt mismatch. MMLongBench-Doc additionally bumps
# max_pages=80 (matches loader's render cap) so the truncation cap
# doesn't cripple the baseline on a 47pp-avg benchmark.
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
  local dataset=$1; local sample=$2; local extra_overrides=$3; local concur=$4; local trial=$5; local run_id=$6
  echo "=== [$ENDPOINT] $dataset no_loop_multi_da trial $trial → $run_id ==="
  uv run python evals.py \
    data.dataset=$dataset \
    data.split=val \
    data.num_samples=null \
    data.doc_ids_file=$sample \
    data.use_profile_scoring=true \
    lm=$MODEL_TAG \
    vlm=$MODEL_TAG \
    lm.enable_thinking=false \
    solver=no_loop_multi_da \
    $extra_overrides \
    max_concurrency=$concur \
    run_id=$run_id 2>&1 | tail -60
}

echo "=== no_loop_multi_da chains on $ENDPOINT ($MODEL_TAG) ==="
date

# --- MP-DocVQA (short docs, default max_pages=10 is plenty) ---
echo "--- MP-DocVQA ---"
for i in 1 2 3; do
  RUN_ID=no-loop-multi-da-mpdv-${ENDPOINT}-t$i
  run_cell lmms-lab/MP-DocVQA data/mp-docvqa/val/sample_200q_doc_ids.txt "" 8 $i $RUN_ID
  out=$(summarize $RUN_ID)
  echo "$out"
  if [ -f "output/runs/$RUN_ID/results.json" ]; then
    notify "MP-DocVQA no_loop_multi_da cell done" "$out"
  fi
done

# --- MMLongBench-Doc (long docs — bump max_pages to the loader cap) ---
echo "--- MMLongBench-Doc (max_pages=80) ---"
for i in 1 2 3; do
  RUN_ID=no-loop-multi-da-pages80-mmlb-${ENDPOINT}-t$i
  run_cell yubo2333/MMLongBench-Doc data/mmlongbench-doc/val/sample_200q_doc_ids.txt "solver.max_pages=80" 4 $i $RUN_ID
  out=$(summarize $RUN_ID)
  echo "$out"
  if [ -f "output/runs/$RUN_ID/results.json" ]; then
    notify "MMLongBench no_loop_multi_da cell done" "$out"
  fi
done

echo "=== no_loop_multi_da chains DONE ==="
date
notify "no_loop_multi_da chains DONE" "endpoint=$ENDPOINT — 6 cells across MP-DocVQA + MMLongBench-Doc"
