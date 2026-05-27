#!/usr/bin/env bash
# rvlm paired-baseline chain on amax1: rvlm t1..t8 val.
#
# Matched-conditions baseline for the unified-tips comparison running on
# amax7 (rvlm_unified t1..t8). Identical model, prompts, c=32, val=80Q.
# Run IDs are paired by trial number so comparison is per-trial-pair.
#
# Prerequisite: Qwen 3.5 27B vllm container up on amax1 (matching amax7's
# localhost:8927 layout — i.e., reachable via the
# `qwen-3_5-27b-vllm-local` LM config).
#
# Resumable per-trial: evals.py skips completed documents when run_id is
# reused, so re-running this script after a crash is safe.
#
# Env:
#   START_TRIAL = first trial number (default 1)
#   END_TRIAL   = last trial number (default 8)
#   CONC        = max_concurrency (default 32; matches the unified chain)

set -uo pipefail
cd /home/baris/repos/docvqa

START_TRIAL=${START_TRIAL:-1}
END_TRIAL=${END_TRIAL:-8}
CONC=${CONC:-32}

echo "=== rvlm paired-baseline chain: t${START_TRIAL}..t${END_TRIAL}, c=${CONC} ==="
for i in $(seq "${START_TRIAL}" "${END_TRIAL}"); do
  echo "--- rvlm t${i} ---"
  uv run python evals.py \
    lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
    lm.enable_thinking=false \
    solver=rvlm \
    data.split=val data.num_samples=null \
    max_concurrency="${CONC}" \
    run_id=rvlm-val-t${i} || { echo "t${i} FAILED"; exit 1; }
done

echo "=== rvlm paired-baseline chain DONE ==="
touch /tmp/rvlm-paired-chain.done
