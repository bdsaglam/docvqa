#!/usr/bin/env python3
"""Compare eval runs side-by-side: overall scores and per-doc breakdown.

Usage:
    python scripts/compare_runs.py run_id1 run_id2 [run_id3 ...]
    python scripts/compare_runs.py --pattern "flat-lm*"
"""

import argparse
import glob
import json
import sys
from pathlib import Path


def load_run(run_id: str) -> dict | None:
    path = Path(f"output/runs/{run_id}/results.json")
    if not path.exists():
        return None
    return json.loads(path.read_text())


def short_id(run_id: str, max_len: int = 25) -> str:
    return run_id[:max_len]


def compare_runs(run_ids: list[str]):
    runs = {}
    for rid in run_ids:
        data = load_run(rid)
        if data is None:
            print(f"  SKIP {rid} (no results.json)", file=sys.stderr)
            continue
        runs[rid] = data

    if not runs:
        print("No valid runs found.")
        return

    # Overall scores
    col_w = max(len(short_id(r)) for r in runs) + 2
    header = f"{'Run':<{col_w}}" + "  Score   "
    cats = sorted(next(iter(runs.values()))["summary"]["by_category"].keys())
    for c in cats:
        header += f" {c[:8]:>8}"
    print("=" * len(header))
    print("OVERALL")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for rid, data in sorted(runs.items(), key=lambda x: -x[1]["summary"]["overall_accuracy"]):
        s = data["summary"]
        row = f"{short_id(rid):<{col_w}}  {s['correct']:>2}/{s['total_questions']:<3} {s['overall_accuracy']*100:>4.1f}%"
        for c in cats:
            d = s["by_category"][c]
            row += f" {d['accuracy']*100:>7.0f}%"
        print(row)

    # Per-doc comparison
    # Collect all docs
    all_docs = {}
    for rid, data in runs.items():
        for doc in data["documents"]:
            did = doc["doc_id"]
            if did not in all_docs:
                all_docs[did] = {"category": doc["doc_category"]}
            all_docs[did][rid] = {
                "accuracy": doc["accuracy"],
                "elapsed": doc["elapsed"],
            }

    print()
    print("=" * len(header))
    print("PER-DOC")
    print("=" * len(header))

    doc_col = 26
    hdr = f"{'Doc':<{doc_col}} {'Cat':<12}"
    for rid in runs:
        hdr += f" {short_id(rid, 12):>12}"
    print(hdr)
    print("-" * len(hdr))

    # Sort by category then doc_id
    for did in sorted(all_docs, key=lambda d: (all_docs[d]["category"], d)):
        info = all_docs[did]
        row = f"{did:<{doc_col}} {info['category']:<12}"
        scores = []
        for rid in runs:
            if rid in info:
                acc = info[rid]["accuracy"]
                scores.append(acc)
                row += f" {acc*100:>11.0f}%"
            else:
                scores.append(None)
                row += f" {'—':>12}"
        # Highlight if there's a big difference
        valid = [s for s in scores if s is not None]
        if valid and max(valid) - min(valid) >= 0.2:
            row += "  ***"
        print(row)

    # Summary: wins per run
    print()
    print("WINS (per doc, excluding ties):")
    wins = {rid: 0 for rid in runs}
    for did, info in all_docs.items():
        scores = {rid: info[rid]["accuracy"] for rid in runs if rid in info}
        if not scores:
            continue
        best = max(scores.values())
        winners = [rid for rid, s in scores.items() if s == best]
        if len(winners) == 1:
            wins[winners[0]] += 1
    for rid in sorted(wins, key=lambda r: -wins[r]):
        print(f"  {short_id(rid):<{col_w}} {wins[rid]} wins")


def main():
    parser = argparse.ArgumentParser(description="Compare eval runs")
    parser.add_argument("run_ids", nargs="*", help="Run IDs to compare")
    parser.add_argument("--pattern", help="Glob pattern for run IDs (e.g. 'flat-lm*')")
    args = parser.parse_args()

    run_ids = list(args.run_ids)
    if args.pattern:
        matched = sorted(glob.glob(f"output/runs/{args.pattern}"))
        run_ids.extend(Path(p).name for p in matched)

    if not run_ids:
        parser.error("Provide run IDs or --pattern")

    compare_runs(run_ids)


if __name__ == "__main__":
    main()
