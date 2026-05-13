"""Calibrate the MMLongBench-Doc Qwen judge against hand-marked triples.

Runs ``qwen_judge`` on a small fixed set of (question, gt, prediction,
answer_format) triples whose verdicts are hand-marked. Reports agreement
and flags disagreements per case. The triples span all 5 answer formats
plus the edge cases the official protocol cares about (unit-stripping,
percent equivalence, list order/incompleteness, "Not answerable").

If agreement is <70%, the handover doc says iterate the judge prompt
before trusting MMLongBench cell results.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from docvqa.judges.qwen_judge import qwen_judge


@dataclass
class Triple:
    question: str
    ground_truth: str
    prediction: str
    answer_format: str  # HF code: Int / Float / Str / List / None
    expected_correct: bool
    note: str = ""


# Hand-marked triples. Verdicts reflect the MMLongBench-Doc scoring rules
# (eval/eval_score.py + prompt_for_answer_extraction.md).
TRIPLES: list[Triple] = [
    # --- Integer ---
    Triple(
        question="How many employees were hired in 2021?",
        ground_truth="42",
        prediction="42",
        answer_format="Int",
        expected_correct=True,
        note="exact integer match",
    ),
    Triple(
        question="What is the total number of items sold?",
        ground_truth="42",
        prediction="42 units",
        answer_format="Int",
        expected_correct=True,
        note="integer with trailing unit — should strip",
    ),
    Triple(
        question="How many countries are listed?",
        ground_truth="7",
        prediction="six",
        answer_format="Int",
        expected_correct=False,
        note="off-by-one wrong",
    ),
    # --- Float ---
    Triple(
        question="What is the year-over-year growth rate?",
        ground_truth="0.25",
        prediction="25%",
        answer_format="Float",
        expected_correct=True,
        note="percent equivalence to fraction",
    ),
    Triple(
        question="What is the average rating?",
        ground_truth="4.5",
        prediction="4.6",
        answer_format="Float",
        expected_correct=False,
        note="outside ~1% tolerance (~2.2% off)",
    ),
    Triple(
        question="What is the standard deviation reported?",
        ground_truth="0.123",
        prediction="0.124",
        answer_format="Float",
        expected_correct=True,
        note="within ~1% tolerance",
    ),
    # --- String ---
    Triple(
        question="Which department published this report?",
        ground_truth="Department of Energy",
        prediction="department of energy",
        answer_format="Str",
        expected_correct=True,
        note="case-insensitive string match",
    ),
    Triple(
        question="Who is the CEO?",
        ground_truth="Sundar Pichai",
        prediction="Tim Cook",
        answer_format="Str",
        expected_correct=False,
        note="wrong named entity",
    ),
    # --- List ---
    Triple(
        question="List all cities mentioned.",
        ground_truth="['Paris', 'Tokyo', 'Lagos']",
        prediction="Tokyo, Paris, Lagos",
        answer_format="List",
        expected_correct=True,
        note="all items present, order doesn't matter",
    ),
    Triple(
        question="List all signatories.",
        ground_truth="['Alice', 'Bob', 'Carol']",
        prediction="Alice, Bob",
        answer_format="List",
        expected_correct=False,
        note="missing item",
    ),
    # --- Not answerable ---
    Triple(
        question="What is the CEO's home address?",
        ground_truth="Not answerable",
        prediction="The document does not contain this information.",
        answer_format="None",
        expected_correct=True,
        note="explicit refusal of an unanswerable question",
    ),
    Triple(
        question="What is the CEO's home address?",
        ground_truth="Not answerable",
        prediction="123 Main St, Springfield",
        answer_format="None",
        expected_correct=False,
        note="hallucinated answer to unanswerable question",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=None, help="Override QWEN_JUDGE_BASE_URL")
    parser.add_argument("--model", default=None, help="Override QWEN_JUDGE_MODEL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    kwargs = {}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    if args.model:
        kwargs["model"] = args.model

    print(f"[calibrate] running {len(TRIPLES)} hand-marked triples...")
    print()
    print(f"{'#':>2}  {'fmt':6} {'exp':5} {'judge':5}  {'agree':5}  {'note'}")
    print("-" * 100)
    agree = 0
    misses = []
    for i, t in enumerate(TRIPLES, 1):
        ok, raw = qwen_judge(
            question=t.question,
            ground_truth=t.ground_truth,
            prediction=t.prediction,
            answer_format=t.answer_format,
            **kwargs,
        )
        match = ok == t.expected_correct
        if match:
            agree += 1
        else:
            misses.append((i, t, ok, raw))
        print(
            f"{i:>2}  {t.answer_format:6} {str(t.expected_correct):5} {str(ok):5}  "
            f"{('✓' if match else '✗'):5}  {t.note}"
        )

    pct = agree / len(TRIPLES) * 100
    print()
    print(f"[calibrate] agreement: {agree}/{len(TRIPLES)} = {pct:.1f}%")
    threshold = 70.0
    if pct >= threshold:
        print(f"[calibrate] OK (>= {threshold:.0f}%). Judge is good enough to proceed.")
    else:
        print(f"[calibrate] BELOW THRESHOLD (< {threshold:.0f}%). Iterate the judge prompt.")
        for i, t, ok, raw in misses:
            print()
            print(f"--- miss #{i} ({t.answer_format}, {t.note}) ---")
            print(f"Q: {t.question}")
            print(f"GT: {t.ground_truth}")
            print(f"Pred: {t.prediction}")
            print(f"Expected: {t.expected_correct}, Judge: {ok}")
            print(f"Raw:\n{raw}")


if __name__ == "__main__":
    main()
