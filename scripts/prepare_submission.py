"""Prepare a submission JSON file from eval run results.

Usage:
    uv run python scripts/prepare_submission.py <run_id> [--output submission.json]

Example:
    uv run python scripts/prepare_submission.py flat-full-pro-flash-v2
    uv run python scripts/prepare_submission.py flat-full-pro-flash-v2 --output submissions/val_v2.json
"""

import argparse
import json
from pathlib import Path


def prepare_submission(run_dir: Path, output_path: Path) -> None:
    tasks_dir = run_dir / "tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

    submissions = []
    for doc_dir in sorted(tasks_dir.iterdir()):
        result_path = doc_dir / "result.json"
        if not result_path.exists():
            continue
        with open(result_path) as f:
            doc_result = json.load(f)

        category = doc_result["doc_category"]
        for q in doc_result["questions"]:
            # Extract clean answer from prediction
            answer = q.get("extracted_answer", q.get("prediction", ""))
            # Remove "FINAL ANSWER: " prefix if present
            if answer.startswith("FINAL ANSWER: "):
                answer = answer[len("FINAL ANSWER: "):]

            submissions.append({
                "category": category,
                "question_id": q["question_id"],
                "answer": answer,
                "full_answer": q.get("prediction", ""),
            })

    # Sort by question_id for consistency
    submissions.sort(key=lambda x: x["question_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submissions, f, indent=2)

    print(f"Submission file: {output_path}")
    print(f"Total questions: {len(submissions)}")

    # Show summary
    from collections import Counter
    cats = Counter(s["category"] for s in submissions)
    for cat, n in sorted(cats.items()):
        print(f"  {cat}: {n}")


def main():
    parser = argparse.ArgumentParser(description="Prepare submission JSON from eval results")
    parser.add_argument("run_id", help="Run ID (directory name under output/runs/)")
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    args = parser.parse_args()

    run_dir = Path("output/runs") / args.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_path = Path(args.output) if args.output else run_dir / "submission.json"
    prepare_submission(run_dir, output_path)


if __name__ == "__main__":
    main()
