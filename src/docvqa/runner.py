"""Evaluation runner with persistence and resumability."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import logfire

from docvqa.data import Document
from docvqa.metrics import evaluate_prediction

logger = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    question_id: str
    question: str
    ground_truth: str | None
    prediction: str
    extracted_answer: str
    is_correct: bool | None  # None if no ground truth


@dataclass
class DocumentResult:
    doc_id: str
    doc_category: str
    questions: list[QuestionResult]
    elapsed_seconds: float
    trace_id: str | None = None

    @property
    def accuracy(self) -> float | None:
        scored = [q for q in self.questions if q.is_correct is not None]
        if not scored:
            return None
        return sum(q.is_correct for q in scored) / len(scored)


def _load_completed(tasks_dir: Path) -> dict[str, DocumentResult]:
    """Load already-completed document results from disk."""
    completed = {}
    if not tasks_dir.exists():
        return completed
    for doc_dir in tasks_dir.iterdir():
        result_path = doc_dir / "result.json"
        if result_path.exists():
            data = json.loads(result_path.read_text())
            questions = [QuestionResult(**q) for q in data["questions"]]
            completed[data["doc_id"]] = DocumentResult(
                doc_id=data["doc_id"],
                doc_category=data["doc_category"],
                questions=questions,
                elapsed_seconds=data["elapsed_seconds"],
                trace_id=data.get("trace_id"),
            )
    return completed


def _save_result(
    tasks_dir: Path,
    result: DocumentResult,
    document: Document | None = None,
    trajectories: dict[str, list[dict]] | None = None,
) -> None:
    """Save a document result and inspection artifacts to disk."""
    doc_dir = tasks_dir / result.doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    # Save result JSON
    data = {
        "doc_id": result.doc_id,
        "doc_category": result.doc_category,
        "questions": [asdict(q) for q in result.questions],
        "elapsed_seconds": result.elapsed_seconds,
        "accuracy": result.accuracy,
    }
    if result.trace_id:
        data["trace_id"] = result.trace_id
        if logfire_project_url := os.environ.get("LOGFIRE_PROJECT_URL"):
            data["logfire_url"] = (
                f"{logfire_project_url}?q=trace_id%3D%27{result.trace_id}%27"
            )
    if trajectories:
        data["trajectories"] = trajectories
    (doc_dir / "result.json").write_text(json.dumps(data, indent=2))

    # Save document page images
    if document:
        for i, img in enumerate(document.images):
            img.save(doc_dir / f"page_{i}.jpg", format="JPEG", quality=85)

    # Save human-readable summary
    _save_summary_md(doc_dir, result, trajectories)


def _save_summary_md(
    doc_dir: Path,
    result: DocumentResult,
    trajectories: dict[str, list[dict]] | None = None,
) -> None:
    """Write a human-readable markdown summary for a document."""
    acc_str = f"{result.accuracy:.0%}" if result.accuracy is not None else "N/A"
    lines = [
        f"# {result.doc_id}",
        f"",
        f"**Category:** {result.doc_category}  ",
        f"**Accuracy:** {acc_str}  ",
        f"**Elapsed:** {result.elapsed_seconds:.1f}s  ",
        f"",
    ]

    for q in result.questions:
        status = (
            "CORRECT"
            if q.is_correct
            else ("WRONG" if q.is_correct is False else "UNSCORED")
        )
        lines.append(f"## [{status}] {q.question_id}")
        lines.append(f"")
        lines.append(f"**Question:** {q.question}  ")
        lines.append(f"**Ground truth:** {q.ground_truth}  ")
        lines.append(f"**Prediction:** {q.extracted_answer}  ")
        lines.append(f"")

        # Append trajectory if available
        traj = (trajectories or {}).get(q.question_id, [])
        if traj:
            lines.append(f"### Trajectory ({len(traj)} iterations)")
            lines.append(f"")
            for i, step in enumerate(traj):
                lines.append(f"#### Iteration {i + 1}")
                if step.get("reasoning"):
                    lines.append(f"**Reasoning:** {step['reasoning']}")
                    lines.append(f"")
                if step.get("code"):
                    lines.append(f"```python\n{step['code']}\n```")
                    lines.append(f"")
                if step.get("output"):
                    output = step["output"]
                    if len(output) > 2000:
                        output = output[:2000] + "\n... [truncated]"
                    lines.append(f"**Output:**")
                    lines.append(f"```\n{output}\n```")
                    lines.append(f"")

    (doc_dir / "summary.md").write_text("\n".join(lines))


def _solve_document(
    solver, document: Document
) -> tuple[DocumentResult, dict[str, list[dict]]]:
    """Run solver on a single document and evaluate predictions."""
    start = time.monotonic()
    if hasattr(solver, "solve_document"):
        predictions, trajectories = solver.solve_document(document)
    else:
        predictions = solver.solve(document)
        trajectories = solver.get_trajectories()
    elapsed = time.monotonic() - start

    question_results = []
    for q in document.questions:
        pred = predictions.get(q.question_id, "Unknown")
        if q.answer is not None:
            is_correct, extracted = evaluate_prediction(pred, q.answer)
        else:
            is_correct, extracted = None, pred.strip()

        question_results.append(
            QuestionResult(
                question_id=q.question_id,
                question=q.question,
                ground_truth=q.answer,
                prediction=pred,
                extracted_answer=extracted,
                is_correct=is_correct,
            )
        )

    result = DocumentResult(
        doc_id=document.doc_id,
        doc_category=document.doc_category,
        questions=question_results,
        elapsed_seconds=elapsed,
    )
    return result, trajectories


def _compute_summary(results: list[DocumentResult]) -> dict:
    """Compute aggregate metrics from document results."""
    all_questions = [q for r in results for q in r.questions]
    scored = [q for q in all_questions if q.is_correct is not None]

    if not scored:
        return {
            "overall_accuracy": None,
            "total_questions": len(all_questions),
            "scored_questions": 0,
        }

    overall_acc = sum(q.is_correct for q in scored) / len(scored)

    # Per-category breakdown
    categories: dict[str, list[QuestionResult]] = {}
    for r in results:
        categories.setdefault(r.doc_category, []).extend(
            q for q in r.questions if q.is_correct is not None
        )

    category_acc = {}
    for cat, qs in sorted(categories.items()):
        category_acc[cat] = {
            "accuracy": sum(q.is_correct for q in qs) / len(qs),
            "correct": sum(q.is_correct for q in qs),
            "total": len(qs),
        }

    return {
        "overall_accuracy": overall_acc,
        "total_questions": len(all_questions),
        "scored_questions": len(scored),
        "correct": sum(q.is_correct for q in scored),
        "by_category": category_acc,
    }


def evaluate(
    solver,
    documents: list[Document],
    output_dir: Path,
    max_concurrency: int = 1,
    task_timeout_seconds: int = 600,
) -> dict:
    """Run evaluation on all documents with persistence and resumability.

    Returns summary dict with metrics.
    """
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    # Load completed results for resumability
    completed = _load_completed(tasks_dir)
    if completed:
        print(
            f"Resuming: found {len(completed)} completed documents, {len(documents) - len(completed)} remaining"
        )

    results: list[DocumentResult] = list(completed.values())
    pending = [d for d in documents if d.doc_id not in completed]

    def _process_doc(doc: Document) -> DocumentResult:
        print(
            f"Solving {doc.doc_id} ({doc.doc_category}, {len(doc.questions)} questions)"
        )
        with logfire.span(
            "solve_document", doc_id=doc.doc_id, doc_category=doc.doc_category
        ) as doc_span:
            trace_id = (
                format(doc_span.context.trace_id, "032x") if doc_span.context else None
            )
            doc_span.set_attribute("num_pages", len(doc.images))
            # Use a thread with timeout to prevent hung API calls
            with ThreadPoolExecutor(max_workers=1) as timeout_pool:
                future = timeout_pool.submit(_solve_document, solver, doc)
                try:
                    result, trajectories = future.result(timeout=task_timeout_seconds)
                except Exception as e:
                    logger.warning(
                        "Document %s timed out or failed after %ds: %s",
                        doc.doc_id,
                        task_timeout_seconds,
                        e,
                    )
                    elapsed = float(task_timeout_seconds)
                    question_results = []
                    for q in doc.questions:
                        is_correct, extracted = (
                            (False, "Unknown") if q.answer else (None, "Unknown")
                        )
                        question_results.append(
                            QuestionResult(
                                question_id=q.question_id,
                                question=q.question,
                                ground_truth=q.answer,
                                prediction="Unknown",
                                extracted_answer="Unknown",
                                is_correct=is_correct,
                            )
                        )
                    result = DocumentResult(
                        doc_id=doc.doc_id,
                        doc_category=doc.doc_category,
                        questions=question_results,
                        elapsed_seconds=elapsed,
                    )
                    trajectories = {}
            result.trace_id = trace_id
            doc_span.set_attribute("num_questions", len(result.questions))
            scored = [q for q in result.questions if q.is_correct is not None]
            correct = sum(q.is_correct for q in scored)
            doc_span.set_attribute("num_scored", len(scored))
            doc_span.set_attribute("num_correct", correct)
            doc_span.set_attribute("accuracy", result.accuracy)
            doc_span.set_attribute("elapsed_seconds", result.elapsed_seconds)
        _save_result(tasks_dir, result, document=doc, trajectories=trajectories)
        acc_str = f"{result.accuracy:.0%}" if result.accuracy is not None else "N/A"
        print(f"  {doc.doc_id} -> {acc_str} ({result.elapsed_seconds:.1f}s)")
        return result

    run_id = output_dir.name
    with logfire.span(
        "eval-{run_id}",
        run_id=run_id,
        total_documents=len(documents),
        pending_documents=len(pending),
        resumed_documents=len(completed),
        max_concurrency=max_concurrency,
        output_dir=str(output_dir),
    ) as eval_span:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {executor.submit(_process_doc, doc): doc for doc in pending}
            for future in as_completed(futures):
                results.append(future.result())

        summary = _compute_summary(results)
        eval_span.set_attribute("overall_accuracy", summary.get("overall_accuracy"))
        eval_span.set_attribute("total_questions", summary.get("total_questions"))
        eval_span.set_attribute("correct", summary.get("correct"))

    # Save summary
    summary_path = output_dir / "results.json"
    logfire_project_url = os.environ.get("LOGFIRE_PROJECT_URL")
    summary_with_docs = {
        "summary": summary,
        "documents": [
            {
                "doc_id": r.doc_id,
                "doc_category": r.doc_category,
                "accuracy": r.accuracy,
                "elapsed": r.elapsed_seconds,
                "trace_id": r.trace_id,
                "logfire_url": f"{logfire_project_url}?q=trace_id%3D%27{r.trace_id}%27"
                if r.trace_id and logfire_project_url
                else None,
            }
            for r in results
        ],
    }
    summary_path.write_text(json.dumps(summary_with_docs, indent=2))

    # Generate submission file
    _save_submission(output_dir, results)

    return summary


def _save_submission(output_dir: Path, results: list[DocumentResult]) -> None:
    """Generate a submission JSON from eval results."""
    submissions = []
    for r in results:
        for q in r.questions:
            answer = q.extracted_answer or q.prediction
            if answer.startswith("FINAL ANSWER: "):
                answer = answer[len("FINAL ANSWER: ") :]
            submissions.append(
                {
                    "category": r.doc_category,
                    "question_id": q.question_id,
                    "answer": answer,
                    "full_answer": q.prediction,
                }
            )
    submissions.sort(key=lambda x: x["question_id"])
    submission_path = output_dir / "submission.json"
    submission_path.write_text(json.dumps(submissions, indent=2))
    print(f"Submission file: {submission_path} ({len(submissions)} questions)")
