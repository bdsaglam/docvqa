#!/usr/bin/env bash
# ReAct baseline chain: react t1..t8 val.
#
# Same paired-comparison setup as run_rvlm_paired_baseline.sh: same model
# (Qwen 3.5 27B), same prompts (modulo ReAct-vs-REPL framing), same val
# split, run_ids `react-val-t1..t8`. Lets us compare ReAct (no Python
# REPL) vs RVLM (LeanRLM + subprocess REPL) per-trial.
#
# Prerequisite: Qwen 3.5 27B vllm container up on amax1 at
# localhost:8927 (the `qwen-3_5-27b-vllm-local` LM config).
#
# Resumable per-trial: evals.py skips completed documents when run_id
# is reused, so re-running this script after a crash is safe.
#
# Env:
#   START_TRIAL = first trial number (default 1)
#   END_TRIAL   = last trial number (default 8)
#   CONC        = max_concurrency (default 24, matches the rvlm baseline)

set -uo pipefail
cd /home/baris/repos/docvqa

START_TRIAL=${START_TRIAL:-1}
END_TRIAL=${END_TRIAL:-8}
CONC=${CONC:-24}

echo "=== react chain: t${START_TRIAL}..t${END_TRIAL}, c=${CONC} ==="
for i in $(seq "${START_TRIAL}" "${END_TRIAL}"); do
  echo "--- react t${i} ---"
  uv run python evals.py \
    lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
    lm.enable_thinking=false \
    solver=react \
    data.split=val data.num_samples=null \
    max_concurrency="${CONC}" \
    run_id=react-val-t${i} || { echo "t${i} FAILED"; exit 1; }
done

echo "=== react chain DONE ==="
touch /tmp/react-chain.done
