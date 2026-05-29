"""Overlap orchestrator: launch hybrid when naked hits 22/25 docs.

Replaces the sentinel-blocked `run_rvlm_hybrid_post.sh` so the GPU
doesn't sit idle on naked's long-tail (4 / 25 docs hit the 14400s
task_timeout in the rvlm_minimal chain — same expected for naked).

Flow:
  1. Wait for `rvlm-naked-val-t1/tasks/<doc>/result.json` count >= 22
     OR for naked's `results.json` to land (means naked closed early).
  2. Launch `rvlm-hybrid-val-t1` in tmux session `rvlm-hybrid`.
     Naked's long-tail keeps running in `rvlm-strip-chain` tmux —
     vllm at c=64 max in flight is fine on Qwen 27B 3-GPU.
  3. Poll for hybrid's `results.json`; touch
     `/tmp/rvlm-hybrid.done` when it lands.

Safety: each run_id is unique; per-doc resumability means a tmux
death is recoverable by re-running the same command.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path("/home/baris/repos/docvqa")
NAKED_TASKS = REPO / "output/runs/rvlm-naked-val-t1/tasks"
NAKED_RESULTS = REPO / "output/runs/rvlm-naked-val-t1/results.json"
HYBRID_DIR = REPO / "output/runs/rvlm-hybrid-val-t1"
HYBRID_RESULTS = HYBRID_DIR / "results.json"
TRIGGER_DOCS = 22
POLL_SECS = 30
SENTINEL = Path("/tmp/rvlm-hybrid.done")
LOG = REPO / "output/runs/rvlm-strip-chain.log"


def log(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(f"[overlap-orch {time.strftime('%H:%M:%S')}] {msg}\n")


def naked_docs_done() -> int:
    if not NAKED_TASKS.exists():
        return 0
    return sum(1 for sub in NAKED_TASKS.iterdir() if (sub / "result.json").exists())


def hybrid_already_running() -> bool:
    return HYBRID_DIR.exists()


def launch_hybrid() -> None:
    cmd = (
        f"cd {REPO} && "
        "exec uv run python evals.py "
        "lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local "
        "lm.enable_thinking=false solver=rvlm_hybrid "
        "data.split=val data.num_samples=null max_concurrency=32 "
        f"run_id=rvlm-hybrid-val-t1 >> {LOG} 2>&1"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", "rvlm-hybrid", "bash", "-c", cmd],
        check=True,
    )
    log("launched rvlm-hybrid-val-t1 in tmux rvlm-hybrid")


def main() -> None:
    log(f"overlap orch starting (trigger={TRIGGER_DOCS}/25 docs)")

    # Wait until naked is at the trigger OR has closed early.
    while True:
        if hybrid_already_running():
            log("hybrid dir already exists; skipping launch, jumping to wait phase")
            break
        done = naked_docs_done()
        closed = NAKED_RESULTS.exists()
        log(f"naked: {done}/25 docs (closed={closed})")
        if done >= TRIGGER_DOCS or closed:
            log(f"trigger reached (done={done}, closed={closed}); launching hybrid")
            launch_hybrid()
            break
        time.sleep(POLL_SECS)

    # Wait for hybrid to close and write the sentinel.
    while not HYBRID_RESULTS.exists():
        time.sleep(POLL_SECS * 2)

    SENTINEL.touch()
    log(f"hybrid done; wrote {SENTINEL}")


if __name__ == "__main__":
    main()
