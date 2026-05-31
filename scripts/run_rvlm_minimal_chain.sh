#!/usr/bin/env bash
# rvlm_minimal chain on amax7: t1..t8 val.
#
# Tests the proposed-method generality claim: strip DocVQA-2026 category
# tips from the solver body, keep only generic document-shape guidance.
# This minimal variant is now the canonical `rvlm` solver; `solver=rvlm_minimal`
# below resolves via the temp alias kept for this in-flight chain.
#
# Resumable per-trial: evals.py skips completed run_ids.
#
# Env:
#   START_TRIAL = first trial number (default 1)
#   END_TRIAL   = last trial number (default 8)
#   CONC        = max_concurrency (default 32; matches unified-tips chain)

set -uo pipefail
cd /home/baris/repos/docvqa

START_TRIAL=${START_TRIAL:-1}
END_TRIAL=${END_TRIAL:-8}
CONC=${CONC:-32}

echo "=== rvlm_minimal chain: t${START_TRIAL}..t${END_TRIAL}, c=${CONC} ==="
for i in $(seq "${START_TRIAL}" "${END_TRIAL}"); do
  echo "--- rvlm-minimal t${i} ---"
  uv run python evals.py \
    lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
    lm.enable_thinking=false \
    solver=rvlm_minimal \
    data.split=val data.num_samples=null \
    max_concurrency="${CONC}" \
    run_id=rvlm-minimal-val-t${i} || { echo "t${i} FAILED"; exit 1; }
done

echo "=== rvlm_minimal chain DONE ==="
touch /tmp/rvlm-minimal-chain.done
