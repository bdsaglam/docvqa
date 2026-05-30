"""Hybrid n=8 chain orchestrator (post-skeletal).

Waits for `/tmp/skeletal-n8.done` to land, then drives hybrid t3..t8
with the 22/25 overlap pattern. Hybrid t1+t2 are already done
(35.00% each on common 21-doc / 68-Q subset; n=2 paired Δ vs minimal
= −8.09pp).

Per the queue-the-next-cell-immediately rule, this script runs in
tmux from now and idles on the wait-for-sentinel; when skeletal-n8
closes the loop hands off without idle GPU.

After hybrid-t8 closes, touches `/tmp/hybrid-n8.done`.

Each trial runs in its own tmux session `rvlm-hyb-tN`.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path("/home/baris/repos/docvqa")
RUNS = REPO / "output/runs"
TRIGGER_DOCS = 22
POLL_SECS = 60
START_TRIAL = 3
END_TRIAL = 8
WAIT_SENTINEL = Path("/tmp/skeletal-n8.done")
SENTINEL = Path("/tmp/hybrid-n8.done")
LOG = RUNS / "hybrid-n8-chain.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"[hyb-n8 {time.strftime('%H:%M:%S')}] {msg}\n")


def docs_done(trial: int) -> int:
    tasks = RUNS / f"rvlm-hybrid-val-t{trial}" / "tasks"
    if not tasks.exists():
        return 0
    return sum(1 for sub in tasks.iterdir() if (sub / "result.json").exists())


def is_finished(trial: int) -> bool:
    return (RUNS / f"rvlm-hybrid-val-t{trial}" / "results.json").exists()


def trial_exists(trial: int) -> bool:
    return (RUNS / f"rvlm-hybrid-val-t{trial}").exists()


def launch(trial: int) -> None:
    rid = f"rvlm-hybrid-val-t{trial}"
    session = f"rvlm-hyb-t{trial}"
    cmd = (
        f"cd {REPO} && "
        "exec uv run python evals.py "
        "lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local "
        "lm.enable_thinking=false solver=rvlm_hybrid "
        "data.split=val data.num_samples=null max_concurrency=32 "
        f"run_id={rid} >> {LOG} 2>&1"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "bash", "-c", cmd],
        check=True,
    )
    log(f"launched {rid} in tmux {session}")


def main() -> None:
    log(f"hybrid n=8 post-orch starting; waiting for {WAIT_SENTINEL}")
    SENTINEL.unlink(missing_ok=True)

    while not WAIT_SENTINEL.exists():
        time.sleep(POLL_SECS)
    log("skeletal n=8 sentinel landed; launching hybrid t3")

    # Stage 1: launch start trial.
    if not is_finished(START_TRIAL) and docs_done(START_TRIAL) == 0:
        launch(START_TRIAL)
    else:
        log(f"t{START_TRIAL} already started; advancing")

    frontier = START_TRIAL

    while frontier < END_TRIAL:
        while True:
            done = docs_done(frontier)
            finished = is_finished(frontier)
            log(f"t{frontier}: {done}/25 (finished={finished})")
            if done >= TRIGGER_DOCS or finished:
                break
            time.sleep(POLL_SECS)
        nxt = frontier + 1
        if trial_exists(nxt) and (is_finished(nxt) or docs_done(nxt) > 0):
            log(f"t{nxt} already in progress / done; skipping launch")
        else:
            launch(nxt)
        frontier = nxt

    while not is_finished(END_TRIAL):
        log(f"waiting on t{END_TRIAL} ({docs_done(END_TRIAL)}/25)")
        time.sleep(POLL_SECS * 2)

    SENTINEL.touch()
    log(f"DONE — wrote {SENTINEL}")


if __name__ == "__main__":
    main()
