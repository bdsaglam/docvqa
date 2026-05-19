#!/usr/bin/env bash
# Prompt-scrub audit chain (post-f4f0cfd).
#
# 4 trials × {flat_solo, leanest_solo} × {val, test} = 16 evals.
#
# Goal: measure whether the val>test generalization gap narrows after
# stripping val-set leakage and flat_solo-specific verbs from category
# tips (commit f4f0cfd).
#
# Test predictions land in output/runs/<run_id>/; ICDAR submission is
# a separate manual step.
set -uo pipefail
cd /home/baris/repos/docvqa

NTFY_TOPIC="${NTFY_TOPIC:-claude-baris}"
NOTIFY=/home/baris/dotfiles/tools/notify.sh
LOGDIR=logs/prompt-scrub
mkdir -p "$LOGDIR"

run_trial() {
  local label="$1"
  local solver="$2"
  local split="$3"
  local trial="$4"
  local run_id="${label}-${split}-scrub-t${trial}"
  local log="${LOGDIR}/${run_id}.log"
  echo
  echo "=== $(date '+%F %H:%M:%S') START ${run_id} ==="
  uv run python evals.py \
    lm=qwen-3_5-27b-vllm-local \
    vlm=qwen-3_5-27b-vllm-local \
    lm.enable_thinking=false \
    solver="${solver}" \
    solver.rlm_type=lean \
    data.split="${split}" \
    data.num_samples=null \
    max_concurrency=24 \
    run_id="${run_id}" 2>&1 | tee "${log}" \
    || { echo "[FAIL] ${run_id} — see ${log}"; return 1; }
  echo "=== $(date '+%F %H:%M:%S') END   ${run_id} ==="
}

notify() {
  local title="$1"; shift
  local body="$*"
  if [ -x "$NOTIFY" ]; then
    printf '%s' "$body" | "$NOTIFY" "$NTFY_TOPIC" "$title" || true
  fi
}

START_TS=$(date '+%F %H:%M:%S')
echo "############# prompt-scrub chain start ${START_TS} #############"

# Phase 1: val (locally scorable).
for i in 1 2 3 4; do
  run_trial flat-solo  flat_solo  val "$i" || true
done
notify "docvqa: val flat_solo ×4 done" "Phase 1a complete. Logs at logs/prompt-scrub/"

for i in 1 2 3 4; do
  run_trial leanest-solo leanest_solo val "$i" || true
done
notify "docvqa: val leanest_solo ×4 done" "Phase 1 complete. All val trials in. Phase 2 (test) starting."

# Phase 2: test (predictions only — ICDAR submission required to score).
for i in 1 2 3 4; do
  run_trial flat-solo  flat_solo  test "$i" || true
done
notify "docvqa: test flat_solo ×4 done" "Phase 2a complete. Submission JSONs to be prepared."

for i in 1 2 3 4; do
  run_trial leanest-solo leanest_solo test "$i" || true
done

END_TS=$(date '+%F %H:%M:%S')
echo "############# prompt-scrub chain end ${END_TS} #############"
notify "docvqa: prompt-scrub ALL 16 trials done" \
  "Start: ${START_TS}  End: ${END_TS}  Next: report + ICDAR submission for test."
