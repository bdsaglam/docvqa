"""Analyze failures from an eval run, write to markdown file."""

import json
import sys
import time
from pathlib import Path


def classify_failure(pred: str, gt: str) -> str:
    pred_clean = pred.replace("FINAL ANSWER: ", "").strip()
    gt_clean = gt.strip()

    if gt_clean.lower() in ("unknown", "unkown"):
        return "Agent answered when GT is Unknown"
    if pred_clean.lower() == "unknown":
        return "Agent said Unknown but GT has an answer"
    try:
        pg = float(gt_clean.replace("%", "").replace(",", ""))
        pp = float(pred_clean.replace("%", "").replace(",", ""))
        if abs(pg - pp) / max(abs(pg), 1) < 0.15:
            return f"Close numeric miss ({pp} vs {pg})"
        return f"Wrong numeric value ({pp} vs {pg})"
    except (ValueError, ZeroDivisionError):
        pass
    return "Wrong answer"


def analyze_doc(result_path: Path) -> str:
    with open(result_path) as f:
        d = json.load(f)

    cat = d["doc_category"]
    acc = d.get("accuracy", 0)
    wrong = [q for q in d["questions"] if not q.get("is_correct")]

    lines = []
    if not wrong:
        lines.append(f"## {d['doc_id']} ({cat}) — {acc*100:.0f}% All correct\n")
    else:
        lines.append(f"## {d['doc_id']} ({cat}) — {acc*100:.0f}%\n")
        for q in wrong:
            lines.append(f"### {q['question_id']}\n")
            lines.append(f"**Q:** {q['question'][:200]}\n")
            lines.append(f"**Predicted:** {q['prediction'][:100]}\n")
            lines.append(f"**Ground Truth:** {q['ground_truth'][:100]}\n")
            failure = classify_failure(q["prediction"], q["ground_truth"])
            lines.append(f"**Failure type:** {failure}\n")
            lines.append(f"**Trajectory:** `{result_path.parent}/summary.md`\n")
            lines.append("")
    return "\n".join(lines)


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "flat-full-val-v5"
    log_file = Path(f"/tmp/flat-full-v5.log") if run_id == "flat-full-val-v5" else Path(f"/tmp/{run_id}.log")
    results_dir = Path(f"output/runs/{run_id}/tasks")
    out_file = Path("docs/failure-analysis-v5.md")

    out_file.write_text(f"# Failure Analysis — {run_id}\n\n")
    seen = set()

    while True:
        for result_file in sorted(results_dir.glob("*/result.json")):
            doc_id = result_file.parent.name
            if doc_id in seen:
                continue
            seen.add(doc_id)
            analysis = analyze_doc(result_file)
            with open(out_file, "a") as f:
                f.write(analysis + "\n")
            print(f"Analyzed {doc_id}")

        # Check if eval is done
        if log_file.exists() and "Overall accuracy" in log_file.read_text():
            with open(out_file, "a") as f:
                f.write("\n## Final Results\n```\n")
                lines = log_file.read_text().splitlines()
                for line in lines[-15:]:
                    f.write(line + "\n")
                f.write("```\n")
            print("Done!")
            break

        time.sleep(30)


if __name__ == "__main__":
    main()
