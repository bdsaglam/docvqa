"""Skeletal n=8 chain orchestrator.

Drives skeletal t2..t8 plus refill for the 4 docs t2 lost to long-tail.
Pattern (same as rvlm_overlap_orch.py): launch next trial when current
hits 22/25 docs (or closes early). Never wait for the long-tail; keep
the GPU pegged.

Flow:
  1. Skeletal-t2 refill: same run_id resumes, picks up the 4 missing
     docs (business_report_1, engineering_drawing_1/_4, science_poster_1).
  2. When t2 hits 22/25 or closes, launch t3.
  3. Same overlap for t3 → t4 → ... → t8.
  4. After t8's results.json lands, touch `/tmp/skeletal-n8.done`.

Each trial runs in its own tmux session `rvlm-skel-tN`.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path("/home/baris/repos/docvqa")
RUNS = REPO / "output/runs"
TRIGGER_DOCS = 22
POLL_SECS = 30
START_TRIAL = 2
END_TRIAL = 8
SENTINEL = Path("/tmp/skeletal-n8.done")
LOG = RUNS / "skeletal-n8-chain.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"[skel-n8 {time.strftime('%H:%M:%S')}] {msg}\n")


def docs_done(trial: int) -> int:
    tasks = RUNS / f"rvlm-skeletal-val-t{trial}" / "tasks"
    if not tasks.exists():
        return 0
    return sum(1 for sub in tasks.iterdir() if (sub / "result.json").exists())


def is_finished(trial: int) -> bool:
    return (RUNS / f"rvlm-skeletal-val-t{trial}" / "results.json").exists()


def trial_exists(trial: int) -> bool:
    return (RUNS / f"rvlm-skeletal-val-t{trial}").exists()


def launch(trial: int) -> None:
    rid = f"rvlm-skeletal-val-t{trial}"
    session = f"rvlm-skel-t{trial}"
    cmd = (
        f"cd {REPO} && "
        "exec uv run python evals.py "
        "lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local "
        "lm.enable_thinking=false solver=rvlm_skeletal "
        "data.split=val data.num_samples=null max_concurrency=32 "
        f"run_id={rid} >> {LOG} 2>&1"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "bash", "-c", cmd],
        check=True,
    )
    log(f"launched {rid} in tmux {session}")


def main() -> None:
    log(f"skeletal n=8 orch starting (trials {START_TRIAL}..{END_TRIAL}, trigger={TRIGGER_DOCS}/25)")
    SENTINEL.unlink(missing_ok=True)

    # Stage 1: kick off t2 refill (same run_id resumes).
    if not is_finished(START_TRIAL) or docs_done(START_TRIAL) < 25:
        # Need refill or fresh launch. Either way, launching with the same
        # run_id is safe — runner is per-doc resumable.
        launch(START_TRIAL)
    else:
        log(f"t{START_TRIAL} already 25/25; advancing")

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

    # Wait for the final trial's results.json.
    while not is_finished(END_TRIAL):
        log(f"waiting on t{END_TRIAL} ({docs_done(END_TRIAL)}/25)")
        time.sleep(POLL_SECS * 2)

    SENTINEL.touch()
    log(f"DONE — wrote {SENTINEL}")


if __name__ == "__main__":
    main()
