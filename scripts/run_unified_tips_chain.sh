#!/usr/bin/env bash
# unified-tips chain on amax7: rvlm_unified t2..t8.
#
# t1 already completed on amax7 (45.0%, 36/80). This chain extends to n=8
# per user direction (skipping the D-008 n=2 escalation step).
#
# The matched-conditions baseline (rvlm t1..t8 with identical model,
# prompts, c) runs on amax1 via scripts/run_rvlm_paired_baseline.sh.
#
# Resumable per-trial: evals.py skips completed documents when run_id is
# reused, so re-running this script after a crash is safe.
#
# Env:
#   START_TRIAL = first trial number (default 2)
#   END_TRIAL   = last trial number (default 8)
#   CONC        = max_concurrency (default 32; matches t1)

set -uo pipefail
cd /home/baris/repos/docvqa

START_TRIAL=${START_TRIAL:-2}
END_TRIAL=${END_TRIAL:-8}
CONC=${CONC:-32}

echo "=== unified-tips chain: t${START_TRIAL}..t${END_TRIAL}, c=${CONC} ==="
for i in $(seq "${START_TRIAL}" "${END_TRIAL}"); do
  echo "--- unified-tips t${i} ---"
  uv run python evals.py \
    lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
    lm.enable_thinking=false \
    solver=rvlm_unified \
    data.split=val data.num_samples=null \
    max_concurrency="${CONC}" \
    run_id=rvlm-unified-val-t${i} || { echo "t${i} FAILED"; exit 1; }
done

echo "=== unified-tips chain DONE ==="
touch /tmp/rvlm-unified-chain.done
