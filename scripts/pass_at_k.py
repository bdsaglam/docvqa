"""Compute avg@1, pass@k, and SC@k (self-consistency vote) over trial groups.

A "cell" is one experiment config run as N trials (run_id suffixes ``-t1..-tN``).
For each cell this reports three numbers over the questions answered in every trial:

  - **avg@1**   mean per-trial accuracy (the standard headline number), ± std
  - **pass@k**  oracle upper bound: a question counts if *any* trial got it right
  - **SC@k**    self-consistency: majority-vote the k answers, then score the vote

Scoring is the DocVQA-2026 binary correctness from ``docvqa.metrics`` (the same
function the runner uses). Gold answers are loaded from the dataset named in each
run's ``config.yaml`` and cached under ``tmp/gold_cache/``.

Usage:
    # explicit cells (each arg is a run_id prefix; trials are <prefix>-t*)
    uv run python scripts/pass_at_k.py codeact-chat-val codeact-chat-4b-val

    # auto-discover every multi-trial cell under output/runs/
    uv run python scripts/pass_at_k.py --all [--min-trials 3]

    # markdown table output
    uv run python scripts/pass_at_k.py --all --markdown
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import yaml

from docvqa.metrics import evaluate_prediction
from scripts.vote_submissions import vote  # reuse the established SC-vote logic

RUNS_DIR = Path(__file__).resolve().parent.parent / "output" / "runs"
GOLD_CACHE = Path(__file__).resolve().parent.parent / "tmp" / "gold_cache"

_TRIAL_RE = re.compile(r"-t\d+$")


# --------------------------------------------------------------------------- #
# gold answers (per dataset+split), cached to disk
# --------------------------------------------------------------------------- #
def load_gold(dataset: str, split: str) -> dict[str, str]:
    GOLD_CACHE.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", f"{dataset}_{split}".lower())
    cache = GOLD_CACHE / f"{slug}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    from docvqa.data import load_documents

    docs = load_documents(dataset, split)
    gold: dict[str, str] = {}
    for d in docs:
        for q in d.questions:
            if q.answer is not None:
                gold[q.question_id] = q.answer
    cache.write_text(json.dumps(gold))
    return gold


# --------------------------------------------------------------------------- #
# cell discovery + loading
# --------------------------------------------------------------------------- #
def discover_cells(min_trials: int = 2) -> list[str]:
    """Group named run dirs by stripping the ``-tN`` suffix; keep cells with
    >= min_trials trials. Skips timestamp-named hydra dirs and smoke runs."""
    groups: dict[str, int] = {}
    for d in RUNS_DIR.iterdir():
        if not d.is_dir() or not _TRIAL_RE.search(d.name):
            continue
        if not (d / "submission.json").exists():
            continue
        stem = _TRIAL_RE.sub("", d.name)
        groups[stem] = groups.get(stem, 0) + 1
    return sorted(s for s, n in groups.items() if n >= min_trials)


def trial_dirs(stem: str) -> list[Path]:
    dirs = [
        d for d in RUNS_DIR.glob(f"{stem}-t*")
        if _TRIAL_RE.search(d.name) and (d / "submission.json").exists()
    ]
    return sorted(dirs, key=lambda p: int(p.name.rsplit("-t", 1)[1]))


def load_submission(run_dir: Path) -> dict[str, dict]:
    """{qid: {category, answer}} from submission.json."""
    sub = json.loads((run_dir / "submission.json").read_text())
    return {r["question_id"]: {"category": r["category"], "answer": r["answer"]} for r in sub}


def dataset_split(run_dir: Path) -> tuple[str, str]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    data = cfg.get("data", {})
    return data.get("dataset", "VLR-CVC/DocVQA-2026"), data.get("split", "val")


# --------------------------------------------------------------------------- #
# metrics for one cell
# --------------------------------------------------------------------------- #
def score_cell(stem: str) -> dict | None:
    dirs = trial_dirs(stem)
    if len(dirs) < 2:
        return None

    dataset, split = dataset_split(dirs[0])
    gold = load_gold(dataset, split)
    gold_qids = set(gold)

    all_subs = [load_submission(d) for d in dirs]
    # Drop incomplete trials: keep only those at the cell's max gold-question
    # coverage, so pass@k/SC@k are computed over the full question set rather
    # than a denominator shrunk by one drop-affected trial.
    covers = [len(set(s) & gold_qids) for s in all_subs]
    max_cov = max(covers)
    subs = [s for s, c in zip(all_subs, covers) if c == max_cov]
    dropped = len(all_subs) - len(subs)

    common = set.intersection(*[set(s) for s in subs]) & gold_qids
    if not common:
        return None
    qids = sorted(common)

    # per-trial correctness matrix: correct[qid] = [bool per trial]
    correct: dict[str, list[bool]] = {}
    for qid in qids:
        row = [evaluate_prediction(s[qid]["answer"], gold[qid])[0] for s in subs]
        correct[qid] = row

    k = len(subs)
    # avg@1: per-trial accuracy, then mean ± std across trials
    per_trial_acc = [sum(correct[q][i] for q in qids) / len(qids) for i in range(k)]
    avg1 = statistics.mean(per_trial_acc)
    std = statistics.stdev(per_trial_acc) if k > 1 else 0.0  # sample std (matches docs)

    # pass@k: any trial correct
    passk = sum(any(correct[q]) for q in qids) / len(qids)

    # SC@k: vote then score
    sc_correct = 0
    for qid in qids:
        voted = vote([s[qid]["answer"] for s in subs])
        sc_correct += int(evaluate_prediction(voted, gold[qid])[0])
    sck = sc_correct / len(qids)

    return {
        "cell": stem,
        "k": k,
        "n_q": len(qids),
        "avg1": avg1,
        "std": std,
        "passk": passk,
        "sck": sck,
        "dropped": dropped,
    }


# --------------------------------------------------------------------------- #
def fmt_row(r: dict, markdown: bool) -> str:
    a = f"{100 * r['avg1']:.2f} ± {100 * r['std']:.2f}"
    p = f"{100 * r['passk']:.2f}"
    s = f"{100 * r['sck']:.2f}"
    if markdown:
        return f"| {r['cell']} | {r['k']} | {r['n_q']} | {a} | {p} | {s} |"
    return f"{r['cell']:42s} k={r['k']:<2d} n={r['n_q']:<3d}  avg@1 {a:>14s}   pass@{r['k']} {p:>6s}   SC@{r['k']} {s:>6s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", help="run_id prefixes (trials are <prefix>-t*)")
    ap.add_argument("--all", action="store_true", help="auto-discover all multi-trial cells")
    ap.add_argument("--min-trials", type=int, default=2)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    if args.all:
        cells = discover_cells(args.min_trials)
    elif args.cells:
        cells = args.cells
    else:
        ap.error("provide cell prefixes or --all")

    results = []
    for stem in cells:
        try:
            r = score_cell(stem)
        except Exception as e:  # noqa: BLE001
            print(f"WARN {stem}: {e}", file=sys.stderr)
            continue
        if r:
            results.append(r)

    results.sort(key=lambda r: -r["avg1"])

    if args.markdown:
        print("| Cell | k | n_q | avg@1 (±std) | pass@k | SC@k |")
        print("|---|---|---|---|---|---|")
    for r in results:
        if r["dropped"] and not args.markdown:
            print(f"  (note: dropped {r['dropped']} incomplete trial(s) below max coverage)",
                  file=sys.stderr)
        print(fmt_row(r, args.markdown))


if __name__ == "__main__":
    main()
