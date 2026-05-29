"""n=2 chain orchestrator: skeletal t2 launches, hybrid t2 starts at 22/25.

Mirrors `rvlm_overlap_orch.py` but for the n=2 follow-up cells:
- Launches `rvlm-skeletal-val-t2` immediately in tmux `rvlm-skeletal-t2`.
- When skeletal-t2 hits 22/25 docs (or closes early), launches
  `rvlm-hybrid-val-t2` in tmux `rvlm-hybrid-t2`.
- Touches `/tmp/rvlm-strip-n2.done` once hybrid-t2's results.json lands.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path("/home/baris/repos/docvqa")
SKEL_TASKS = REPO / "output/runs/rvlm-skeletal-val-t2/tasks"
SKEL_RESULTS = REPO / "output/runs/rvlm-skeletal-val-t2/results.json"
HYB_DIR = REPO / "output/runs/rvlm-hybrid-val-t2"
HYB_RESULTS = HYB_DIR / "results.json"
TRIGGER_DOCS = 22
POLL_SECS = 30
SENTINEL = Path("/tmp/rvlm-strip-n2.done")
LOG = REPO / "output/runs/rvlm-strip-n2.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"[n2-orch {time.strftime('%H:%M:%S')}] {msg}\n")


def docs_done(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for sub in p.iterdir() if (sub / "result.json").exists())


def launch(run_id: str, solver: str, session: str) -> None:
    cmd = (
        f"cd {REPO} && "
        "exec uv run python evals.py "
        "lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local "
        "lm.enable_thinking=false "
        f"solver={solver} "
        "data.split=val data.num_samples=null max_concurrency=32 "
        f"run_id={run_id} >> {LOG} 2>&1"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "bash", "-c", cmd],
        check=True,
    )
    log(f"launched {run_id} in tmux {session}")


def main() -> None:
    log(f"n=2 chain orch starting (trigger={TRIGGER_DOCS}/25)")
    SENTINEL.unlink(missing_ok=True)

    # Stage 1: skeletal t2
    if not SKEL_TASKS.parent.exists():
        launch("rvlm-skeletal-val-t2", "rvlm_skeletal", "rvlm-skeletal-t2")
    else:
        log("skeletal-t2 dir already exists; skipping launch")

    # Stage 2: wait for trigger, launch hybrid t2
    while True:
        if HYB_DIR.exists():
            log("hybrid-t2 dir exists; skipping launch")
            break
        done = docs_done(SKEL_TASKS)
        closed = SKEL_RESULTS.exists()
        log(f"skeletal-t2: {done}/25 docs (closed={closed})")
        if done >= TRIGGER_DOCS or closed:
            log(f"trigger reached; launching hybrid-t2")
            launch("rvlm-hybrid-val-t2", "rvlm_hybrid", "rvlm-hybrid-t2")
            break
        time.sleep(POLL_SECS)

    # Stage 3: wait for hybrid-t2 to close.
    while not HYB_RESULTS.exists():
        time.sleep(POLL_SECS * 2)

    SENTINEL.touch()
    log(f"DONE — wrote {SENTINEL}")


if __name__ == "__main__":
    main()
