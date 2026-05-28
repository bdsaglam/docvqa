"""Overlapping launcher for the rvlm_minimal n=8 chain.

When the current frontier trial hits 90% completion (22/25 docs), this
script launches the next trial in its own tmux session — without
waiting for the current trial's long-tail stragglers. Maximizes Qwen
vllm throughput across the 7h-ish wall of the n=8 chain.

Behavior on start:

1. Scans existing run dirs to find the current frontier trial.
2. If t1 was launched under the original sequential chain
   (``scripts/run_rvlm_minimal_chain.sh``), kills that shell at the
   first 90% trigger so it doesn't sequentially launch t2 (which would
   collide with the orchestrator's t2 launch). The chain's t1 python
   eval is then orphaned and continues to completion under init.
3. From there, waits for each frontier trial to reach 90%, then
   launches the next.
4. After t8 is launched, waits for ALL trials (including stragglers)
   to fully finish, then writes ``/tmp/rvlm-minimal-chain.done`` so
   any existing watcher fires.

Each trial runs in its own tmux session ``rvlm-min-tN`` so the user
can ``tmux attach -t rvlm-min-tN`` to inspect any specific trial.

Safety considerations:

- At the moment of overlap, vllm sees the previous trial's tail
  (typically only a few in-flight long-tail docs at the 90% mark)
  plus the new trial's full c=32. Aggregate is usually well below
  c=64; vllm handles this fine on Qwen 27B 3-GPU.
- Resumable evaluation: each trial's run_id is unique. The runner's
  per-doc persistence (commit 8309710) means restarting any trial
  picks up where it left off.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/baris/repos/docvqa")
RUNS = REPO / "output" / "runs"
VAL_DOCS = 25
TRIGGER_DOCS = 22  # 22/25 = 88% — closest integer to "90% trigger"
START_TRIAL = 1
END_TRIAL = 8
POLL_SECS = 30
SENTINEL = Path("/tmp/rvlm-minimal-chain.done")
LOG = RUNS / "rvlm-minimal-chain.log"


def _log(msg: str) -> None:
    print(f"[orch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def docs_done(trial: int) -> int:
    d = RUNS / f"rvlm-minimal-val-t{trial}" / "tasks"
    if not d.exists():
        return 0
    return sum(1 for sub in d.iterdir() if (sub / "result.json").exists())


def is_finished(trial: int) -> bool:
    return (RUNS / f"rvlm-minimal-val-t{trial}" / "results.json").exists()


def trial_exists(trial: int) -> bool:
    return (RUNS / f"rvlm-minimal-val-t{trial}").exists()


def kill_original_chain() -> bool:
    """Kill the sequential chain bash shell so it doesn't auto-launch t2.

    Returns True if any process was killed.
    """
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "run_rvlm_minimal_chain"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        _log("no original chain shell found (already gone)")
        return False
    killed = False
    for pid in out.split("\n"):
        if not pid:
            continue
        try:
            os.kill(int(pid), signal.SIGKILL)
            _log(f"killed chain shell PID {pid}")
            killed = True
        except ProcessLookupError:
            pass
    return killed


def launch_trial(trial: int) -> None:
    """Launch trial t{trial} in a fresh tmux session.

    Uses ``>> LOG 2>&1`` instead of ``| tee`` so a kill of the wrapping
    shell does not break a pipe and SIGPIPE the python eval. The python
    eval is exec'd directly so it inherits PID 1 of its process group
    and stays orphan-safe if the tmux session is ever destroyed.
    """
    session = f"rvlm-min-t{trial}"
    cmd = (
        f"cd {REPO} && "
        f"exec uv run python evals.py "
        f"lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local "
        f"lm.enable_thinking=false solver=rvlm_minimal "
        f"data.split=val data.num_samples=null max_concurrency=32 "
        f"run_id=rvlm-minimal-val-t{trial} >> {LOG} 2>&1"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "bash", "-c", cmd],
        check=True,
    )
    _log(f"launched t{trial} in tmux session {session}")


def wait_for_trigger(trial: int) -> None:
    while True:
        done = docs_done(trial)
        finished = is_finished(trial)
        _log(f"t{trial}: {done}/{VAL_DOCS} docs (finished={finished})")
        if done >= TRIGGER_DOCS or finished:
            return
        time.sleep(POLL_SECS)


def wait_for_completion(trial: int) -> None:
    while not is_finished(trial):
        _log(f"waiting on t{trial} ({docs_done(trial)}/{VAL_DOCS})")
        time.sleep(POLL_SECS)


def main() -> None:
    _log(f"orchestrator starting: trials {START_TRIAL}..{END_TRIAL}, "
         f"trigger={TRIGGER_DOCS}/{VAL_DOCS}, poll={POLL_SECS}s")

    # Find frontier (highest trial with any progress).
    frontier = 0
    for t in range(START_TRIAL, END_TRIAL + 1):
        if docs_done(t) > 0 or is_finished(t):
            frontier = t
    _log(f"discovered frontier: t{frontier}")

    chain_killed = False
    if frontier == 0:
        launch_trial(START_TRIAL)
        frontier = START_TRIAL
        chain_killed = True  # nothing to kill — orchestrator owns it

    while frontier < END_TRIAL:
        wait_for_trigger(frontier)
        if not chain_killed:
            kill_original_chain()
            chain_killed = True
        nxt = frontier + 1
        if is_finished(nxt) or docs_done(nxt) > 0:
            _log(f"t{nxt} already in progress / done; skipping launch")
        else:
            launch_trial(nxt)
        frontier = nxt

    _log("all trials launched; waiting for stragglers to finish")
    for t in range(START_TRIAL, END_TRIAL + 1):
        wait_for_completion(t)

    SENTINEL.touch()
    _log(f"DONE — wrote {SENTINEL}")


if __name__ == "__main__":
    main()
