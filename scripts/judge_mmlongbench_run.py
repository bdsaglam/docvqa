"""Post-hoc Qwen-judge scoring for an MMLongBench-Doc eval run.

The runner stores predictions with ANLS-based scoring (``is_correct``).
For MMLongBench-Doc we want the official semantic judge: a single Qwen
call per (question, gt, prediction, answer_format) triple that extracts
a clean answer and emits a verdict.

This script:
  1. Re-loads MMLongBench-Doc so we have ``answer_format`` per question.
  2. Walks ``output/runs/<run_id>/tasks/*/result.json``.
  3. Calls ``qwen_judge`` on every question.
  4. Writes ``output/runs/<run_id>/results-judged.json`` (summary +
     per-document accuracy) and ``output/runs/<run_id>/tasks/<doc>/result-judged.json``
     (per-question verdicts).

Idempotent: skips doc files that already have a ``result-judged.json``
unless ``--force`` is passed.

Usage::

    uv run python scripts/judge_mmlongbench_run.py --run-id <run_id> \
        [--concurrency 8] [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset

from docvqa.datasets.mmlongbench_doc import HF_REPO_ID
from docvqa.judges.qwen_judge import qwen_judge

logger = logging.getLogger("judge_mmlongbench_run")


def _build_qid_to_format(split: str = "val") -> dict[str, str]:
    """Map question_id → answer_format directly from the HF dataset.

    Skips PDF rendering and image decoding — we only need the question
    text and answer_format columns. The qid synthesis matches the loader
    (``f"{doc_id}::{question[:60]}"``).
    """
    ds = load_dataset(HF_REPO_ID, split="train")
    out: dict[str, str] = {}
    for row in ds:
        qid = f"{row['doc_id']}::{row['question'][:60]}"
        out[qid] = row["answer_format"]
    return out


def _judge_one(question: str, gt: str | None, pred: str, fmt: str) -> tuple[bool, str]:
    if gt is None:
        return False, "[no-gt]"
    return qwen_judge(
        question=question,
        ground_truth=gt,
        prediction=pred,
        answer_format=fmt,
    )


def _judge_doc(doc_dir: Path, fmt_map: dict[str, str], force: bool) -> dict:
    out_path = doc_dir / "result-judged.json"
    res_path = doc_dir / "result.json"
    if not res_path.exists():
        return {}
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())

    res = json.loads(res_path.read_text())
    questions = res["questions"]
    judged_qs = []
    for q in questions:
        fmt = fmt_map.get(q["question_id"], "String")
        ok, raw = _judge_one(
            question=q["question"],
            gt=q.get("ground_truth"),
            pred=q.get("extracted_answer") or q.get("prediction") or "",
            fmt=fmt,
        )
        judged_qs.append(
            {
                "question_id": q["question_id"],
                "answer_format": fmt,
                "is_correct_judge": ok,
                "judge_response": raw,
                "is_correct_anls": q.get("is_correct"),
            }
        )

    scored = [j for j in judged_qs if j["is_correct_judge"] is not None]
    correct = sum(j["is_correct_judge"] for j in scored)
    summary = {
        "doc_id": res["doc_id"],
        "doc_category": res.get("doc_category"),
        "n_questions": len(judged_qs),
        "n_correct_judge": correct,
        "accuracy_judge": (correct / len(scored)) if scored else None,
        "questions": judged_qs,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def _aggregate(run_dir: Path, doc_summaries: list[dict]) -> dict:
    by_format: dict[str, list[bool]] = defaultdict(list)
    by_category: dict[str, list[bool]] = defaultdict(list)
    all_judged: list[bool] = []
    for d in doc_summaries:
        for q in d.get("questions", []):
            v = q["is_correct_judge"]
            all_judged.append(bool(v))
            by_format[q["answer_format"]].append(bool(v))
            by_category[d.get("doc_category") or "_"].append(bool(v))
    summary = {
        "n_questions": len(all_judged),
        "n_correct": sum(all_judged),
        "overall_accuracy_judge": (sum(all_judged) / len(all_judged)) if all_judged else None,
        "by_answer_format": {
            k: {
                "n": len(v),
                "correct": sum(v),
                "accuracy": (sum(v) / len(v)) if v else None,
            }
            for k, v in sorted(by_format.items())
        },
        "by_category": {
            k: {
                "n": len(v),
                "correct": sum(v),
                "accuracy": (sum(v) / len(v)) if v else None,
            }
            for k, v in sorted(by_category.items())
        },
    }
    (run_dir / "results-judged.json").write_text(
        json.dumps({"summary": summary, "documents": doc_summaries}, indent=2)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")

    run_dir = Path("output/runs") / args.run_id
    tasks_dir = run_dir / "tasks"
    if not tasks_dir.exists():
        raise SystemExit(f"no tasks dir: {tasks_dir}")

    print(f"[judge] loading MMLongBench-Doc to build qid → format map...")
    fmt_map = _build_qid_to_format(args.split)
    print(f"[judge] {len(fmt_map)} questions in dataset")

    doc_dirs = sorted(d for d in tasks_dir.iterdir() if d.is_dir())
    print(f"[judge] judging {len(doc_dirs)} docs (concurrency={args.concurrency})")

    summaries = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(_judge_doc, d, fmt_map, args.force): d for d in doc_dirs}
        for f in as_completed(futs):
            d = futs[f]
            try:
                s = f.result()
                if s:
                    summaries.append(s)
                    print(f"[judge] {d.name}: {s.get('accuracy_judge')} ({s.get('n_correct_judge')}/{s.get('n_questions')})")
            except Exception as e:  # pragma: no cover
                print(f"[judge] FAILED {d.name}: {e}")

    print("[judge] aggregating...")
    summary = _aggregate(run_dir, summaries)
    acc = summary.get("overall_accuracy_judge")
    print(f"[judge] overall accuracy (judge): {acc:.3f}" if acc is not None else "[judge] no scored questions")
    print(f"[judge] by_answer_format:")
    for k, v in (summary.get("by_answer_format") or {}).items():
        print(f"  {k:14s} {v['correct']:4d}/{v['n']:<4d} = {v['accuracy']:.3f}")


if __name__ == "__main__":
    main()
