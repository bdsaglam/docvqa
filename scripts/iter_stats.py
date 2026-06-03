#!/usr/bin/env python3
"""Report agent iteration-count efficiency per run.

Each agentic solver records its trajectory length per question in the
per-doc ``summary.md`` (``### Trajectory (N iterations)`` header, where
``N == len(trajectory)`` — the number of code/observation steps the agent
took). This reads those counts and reports mean / median / std / max
iterations-per-question for each run, plus the fraction of questions that
hit the iteration cap (a churn signal — e.g. ``direct_vlm`` pins the cap on
most questions).

Single-shot solvers (``raw_vlm_multi``, ``official``) have no agent loop, so
they report n/a (or ~1 iteration) — the metric only means something for the
iterative REPL/ReAct solvers.

Usage:
    python scripts/iter_stats.py <run_id> [<run_id> ...]

``run_id`` may be a glob — e.g. ``'rvlm-cmp-val-t*'`` reports each of the 8
trials as its own row. Pass several to compare runs side by side:
    python scripts/iter_stats.py 'rvlm-cmp-val-t*' 'codeact-b56-val-t*'
"""
from __future__ import annotations

import re
import statistics as st
import sys
from pathlib import Path

RUNS_DIR = Path("output/runs")
ITER_PAT = re.compile(r"Trajectory \((\d+) iterations\)")


def _run_dirs(arg: str) -> list[Path]:
    """Expand a run_id (possibly a glob) to existing run directories."""
    return sorted(p for p in RUNS_DIR.glob(arg) if (p / "tasks").is_dir())


def _iters_for_run(run_dir: Path) -> list[int]:
    iters: list[int] = []
    for f in run_dir.glob("tasks/*/summary.md"):
        iters += [int(m) for m in ITER_PAT.findall(f.read_text())]
    return iters


def main(argv: list[str]) -> int:
    args = argv[1:]
    if not args:
        print(__doc__)
        return 1

    hdr = (f"{'run_id':<30}{'Qs':>5}{'mean':>7}{'med':>5}{'std':>6}"
           f"{'max':>5}{'%@cap':>7}")
    print(hdr)
    print("-" * len(hdr))

    matched_any = False
    for arg in args:
        run_dirs = _run_dirs(arg)
        if not run_dirs:
            print(f"{arg:<30}{'(no matching run dir)':>30}")
            continue
        for rd in run_dirs:
            matched_any = True
            v = _iters_for_run(rd)
            if not v:
                print(f"{rd.name:<30}{0:>5}{'n/a (single-shot / no trajectory)':>40}")
                continue
            cap = max(v)  # per-run cap = max observed (iteration budget + page bonus)
            at_cap = 100.0 * sum(1 for x in v if x >= cap) / len(v)
            sd = st.stdev(v) if len(v) > 1 else 0.0
            print(f"{rd.name:<30}{len(v):>5}{st.mean(v):>7.1f}{st.median(v):>5.0f}"
                  f"{sd:>6.1f}{cap:>5}{at_cap:>6.0f}%")
    return 0 if matched_any else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
