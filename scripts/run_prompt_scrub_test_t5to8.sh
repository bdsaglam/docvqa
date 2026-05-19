#!/usr/bin/env bash
# Prompt-scrub chain — extension to reach n=8 across all four cells.
#
# Run AFTER scripts/run_prompt_scrub_chain.sh finishes. Adds trials t5..t8
# to:
#   - flat_solo val
#   - leanest_solo val
#   - flat_solo test
#   - leanest_solo test
#
# Ordered val-first (locally scorable, ~75 min/trial → quick feedback on
# whether the audit's val drop tightens at n=8) then test (~3-5h/trial,
# ICDAR-submission target).
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
echo "############# prompt-scrub n=8 extension start ${START_TS} #############"

# Phase 3a: flat_solo val t5..t8 (fastest feedback first)
for i in 5 6 7 8; do
  run_trial flat-solo  flat_solo  val "$i" || true
done
notify "docvqa: flat_solo val t5-t8 done" "Phase 3a done. flat_solo val SC-8 feasible."

# Phase 3b: leanest_solo val t5..t8
for i in 5 6 7 8; do
  run_trial leanest-solo leanest_solo val "$i" || true
done
notify "docvqa: leanest_solo val t5-t8 done" "Phase 3 (val n=8) complete. Phase 4 (test n=8) starting."

# Phase 4a: flat_solo test t5..t8
for i in 5 6 7 8; do
  run_trial flat-solo  flat_solo  test "$i" || true
done
notify "docvqa: flat_solo test t5-t8 done" "Phase 4a done. flat_solo test SC-8 feasible."

# Phase 4b: leanest_solo test t5..t8
for i in 5 6 7 8; do
  run_trial leanest-solo leanest_solo test "$i" || true
done

END_TS=$(date '+%F %H:%M:%S')
echo "############# prompt-scrub n=8 extension end ${END_TS} #############"
notify "docvqa: prompt-scrub n=8 ALL DONE" \
  "All 8 trials per cell. Start: ${START_TS}  End: ${END_TS}.  Next: SC-8 vote + ICDAR submit + final writeup."
