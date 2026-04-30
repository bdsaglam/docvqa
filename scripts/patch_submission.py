#!/usr/bin/env python3
"""Patch Unknown answers in a submission with answers from another run.

Only replaces answers that are "Unknown" in the base submission.
Creates a new patched submission file.

Usage:
    python scripts/patch_submission.py BASE_RUN_ID PATCH_RUN_ID [-o OUTPUT]

Example:
    python scripts/patch_submission.py t06-precise-test-c4 t06-retry-highbudget
    python scripts/patch_submission.py t06-precise-test-c4 t06-retry-highbudget -o output/runs/patched-submission.json
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Patch Unknown answers in submission")
    parser.add_argument("base", help="Base run ID (submission to patch)")
    parser.add_argument("patch", help="Patch run ID (source of replacement answers)")
    parser.add_argument("-o", "--output", help="Output path (default: output/runs/BASE-patched/submission.json)")
    args = parser.parse_args()

    base_path = Path(f"output/runs/{args.base}/submission.json")
    patch_path = Path(f"output/runs/{args.patch}/submission.json")

    if not base_path.exists():
        print(f"Error: {base_path} not found", file=sys.stderr)
        sys.exit(1)
    if not patch_path.exists():
        print(f"Error: {patch_path} not found", file=sys.stderr)
        sys.exit(1)

    base = json.loads(base_path.read_text())
    patch = json.loads(patch_path.read_text())

    # Build patch lookup
    patch_map = {q["question_id"]: q for q in patch}

    # Patch — only replace answer/full_answer, keep original category and other fields
    patched_count = 0
    still_unknown = 0
    for i, q in enumerate(base):
        if q["answer"] == "Unknown" and q["question_id"] in patch_map:
            new_ans = patch_map[q["question_id"]]["answer"]
            if new_ans != "Unknown":
                print(f"  {q['question_id']}: Unknown -> {new_ans[:60]}")
                base[i]["answer"] = new_ans
                base[i]["full_answer"] = patch_map[q["question_id"]].get("full_answer", new_ans)
                patched_count += 1
            else:
                still_unknown += 1

    total_unknown = sum(1 for q in base if q["answer"] == "Unknown")

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(f"output/runs/{args.base}-patched")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "submission.json"

    out_path.write_text(json.dumps(base, indent=2))

    print(f"\nPatched {patched_count} answers")
    print(f"Still Unknown: {total_unknown}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
