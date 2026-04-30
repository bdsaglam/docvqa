"""Self-consistency voting over multiple eval runs.

Aggregates per-question answers from N completed runs and produces a single
voted submission.json. Optionally evaluates voted answers when ground truth is
available in the per-task result.json files.

Usage:
    uv run python scripts/vote_submissions.py \
        --runs output/runs/flat-solo-3_6-27b-val-t{1..8} \
        --output submissions/flat-solo-3_6-27b-val-sc8.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from docvqa.metrics import _clean_text, evaluate_prediction


def load_run(run_dir: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Returns (per-question record, ground truths)."""
    records: dict[str, dict] = {}
    ground_truth: dict[str, str] = {}
    tasks = run_dir / "tasks"
    if not tasks.exists():
        raise FileNotFoundError(f"{tasks} not found")
    for doc_dir in sorted(tasks.iterdir()):
        rp = doc_dir / "result.json"
        if not rp.exists():
            continue
        data = json.loads(rp.read_text())
        cat = data["doc_category"]
        for q in data["questions"]:
            qid = q["question_id"]
            ans = q.get("extracted_answer") or q.get("prediction") or ""
            full = q.get("prediction", ans)
            records[qid] = {"category": cat, "answer": ans, "full_answer": full}
            if q.get("ground_truth") is not None:
                ground_truth[qid] = q["ground_truth"]
    return records, ground_truth


def vote(answers: list[str]) -> str:
    """Majority vote over normalized answers; return canonical raw answer.

    Group by `_clean_text` form. Pick the group with most votes. On tie, prefer
    the group whose canonical raw answer is longest (more specific). If still
    tied, prefer the first-seen answer (deterministic by input order).
    """
    if not answers:
        return ""
    groups: dict[str, list[str]] = defaultdict(list)
    order: dict[str, int] = {}
    for i, a in enumerate(answers):
        key = _clean_text(a)
        if key not in order:
            order[key] = i
        groups[key].append(a)

    def sort_key(item: tuple[str, list[str]]) -> tuple:
        key, vals = item
        # higher count first, then longest canonical answer, then first-seen
        canonical = Counter(vals).most_common(1)[0][0]
        return (-len(vals), -len(canonical), order[key])

    _, best_vals = sorted(groups.items(), key=sort_key)[0]
    # canonical = most common raw form within the winning group
    return Counter(best_vals).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories to vote over")
    ap.add_argument("--output", required=True, help="Output submission JSON path")
    ap.add_argument("--no-eval", action="store_true", help="Skip eval even if GT present")
    args = ap.parse_args()

    run_dirs = [Path(p) for p in args.runs]
    print(f"Voting over {len(run_dirs)} runs:")
    for r in run_dirs:
        print(f"  - {r}")

    per_run_records: list[dict[str, dict]] = []
    gt: dict[str, str] = {}
    for r in run_dirs:
        recs, g = load_run(r)
        per_run_records.append(recs)
        gt.update(g)

    # Collect all qids
    all_qids: set[str] = set()
    for recs in per_run_records:
        all_qids.update(recs.keys())

    # Build voted submission
    voted: list[dict] = []
    cat_lookup: dict[str, str] = {}
    for qid in sorted(all_qids):
        answers = []
        for recs in per_run_records:
            if qid in recs:
                answers.append(recs[qid]["answer"])
                cat_lookup[qid] = recs[qid]["category"]
        voted_answer = vote(answers)
        voted.append({
            "category": cat_lookup[qid],
            "question_id": qid,
            "answer": voted_answer,
            "full_answer": voted_answer,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(voted, indent=2))
    print(f"\nSubmission file: {out} ({len(voted)} questions)")

    # Evaluate if GT available
    if not args.no_eval and gt:
        scored = [v for v in voted if v["question_id"] in gt]
        per_cat: dict[str, list[bool]] = defaultdict(list)
        correct = 0
        for v in scored:
            qid = v["question_id"]
            ok, _ = evaluate_prediction(v["answer"], gt[qid])
            per_cat[v["category"]].append(ok)
            correct += int(ok)
        n = len(scored)
        print("\n" + "=" * 60)
        print(f"SC-{len(run_dirs)} accuracy: {100 * correct / n:.1f}% ({correct}/{n})")
        print("Per category:")
        for cat in sorted(per_cat):
            vals = per_cat[cat]
            print(f"  {cat:20s}: {100 * sum(vals) / len(vals):5.1f}% ({sum(vals)}/{len(vals)})")


if __name__ == "__main__":
    main()
