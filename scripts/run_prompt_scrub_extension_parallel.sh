#!/usr/bin/env bash
# Parallel companion to scripts/run_prompt_scrub_test_t5to8.sh.
#
# Same trials, same prompts, **but** run IDs are suffixed `-p-tN` so it
# can run alongside the queued t5..t8 chain without colliding on task
# directories. The queued chain still fires when the base chain ends.
# Net result: each of the 4 cells gets {t1..t4, p-t5..p-t8, t5..t8} =
# 12 trials, of which any 8 can feed SC-8 voting.
#
# Launch in its own tmux window: `tmux send-keys -t prompt-scrub:parallel
# 'bash scripts/run_prompt_scrub_extension_parallel.sh ...' Enter`.
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
  local run_id="${label}-${split}-scrub-p-t${trial}"
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
    max_concurrency=16 \
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
echo "############# prompt-scrub parallel extension start ${START_TS} #############"

# Phase 3a': flat_solo val p-t5..p-t8 (fast feedback)
for i in 5 6 7 8; do
  run_trial flat-solo  flat_solo  val "$i" || true
done
notify "docvqa: parallel flat_solo val p-t5-p-t8 done" "Phase 3a' done."

# Phase 3b': leanest_solo val p-t5..p-t8
for i in 5 6 7 8; do
  run_trial leanest-solo leanest_solo val "$i" || true
done
notify "docvqa: parallel leanest_solo val p-t5-p-t8 done" "Phase 3' (val) done. Test phase starting."

# Phase 4a': flat_solo test p-t5..p-t8
for i in 5 6 7 8; do
  run_trial flat-solo  flat_solo  test "$i" || true
done
notify "docvqa: parallel flat_solo test p-t5-p-t8 done" "Phase 4a' done. flat_solo test n=8 possible (with t1-t4 + p-t5-p-t8)."

# Phase 4b': leanest_solo test p-t5..p-t8
for i in 5 6 7 8; do
  run_trial leanest-solo leanest_solo test "$i" || true
done

END_TS=$(date '+%F %H:%M:%S')
echo "############# prompt-scrub parallel extension end ${END_TS} #############"
notify "docvqa: parallel extension ALL DONE" \
  "Start: ${START_TS}  End: ${END_TS}  All p-t5..p-t8 trials in. SC-8 voting feasible from {t1..t4, p-t5..p-t8}."
