"""GEPA optimization for flat batch DocVQA solver via optimize_anything.

Components optimized:
- task_instructions: Main RLM agent prompt
- vlm_instructions: VLM perception prompt
- tip_{category}: Per-category tips (8 categories)

Each example is a full Document with all questions batched together.
Uses rich ASI (Actionable Side Information) with per-question feedback.

Usage:
    uv run python scripts/optimize_flat_batch.py
    uv run python scripts/optimize_flat_batch.py --max-metric-calls 60
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

from dotenv import load_dotenv

assert load_dotenv(), "Failed to load .env file"

from docvqa.obs import setup_observability

setup_observability()

import dspy
import gepa.optimize_anything as oa
import litellm
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, TrackingConfig, optimize_anything

from docvqa.data import Document, load_documents
from docvqa.metrics import evaluate_prediction, get_anls
from docvqa.prompts import CATEGORY_TIPS
from docvqa.search import get_or_build_index
from docvqa.solvers.flat_batch_solver import (
    TASK_INSTRUCTIONS,
    FlatBatchProgram,
    RunContext,
    _format_page_texts,
    _parse_answers,
)
from docvqa.types import LMConfig

litellm.drop_params = True
litellm.request_timeout = 300

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded eval doc IDs — one per category, fast + medium difficulty
# ---------------------------------------------------------------------------
EVAL_DOC_IDS = {
    "business_report_3",   # 2q, 89p, ~536s, 50%
    "comics_4",            # 2q, 69p, ~566s, 50%
    "engineering_drawing_2",  # 2q, 1p, ~240s, 100%
    "infographics_1",      # 2q, 1p, ~506s, 50%
    "maps_1",              # 2q, 1p, ~736s, 50%
    "science_paper_2",     # 1q, 19p, ~297s, 0%
    "science_poster_1",    # 5q, 1p, ~1207s, 80%
    "slide_1",             # 2q, 36p, ~308s, 50%
}

# ---------------------------------------------------------------------------
# VLM / LM setup
# ---------------------------------------------------------------------------

VLM_CONFIG = LMConfig(
    model="vertex_ai/gemini-3-flash-preview",
    max_tokens=4096,
)

STUDENT_LM_CONFIG = LMConfig(
    model="vertex_ai/gemini-3-pro-preview",
    temperature=1.0,
)

DEFAULT_VLM_INSTRUCTIONS = (
    "Analyze the image content strictly to answer the query. "
    "Transcribe numbers and characters exactly. "
    "For technical drawings, trace leader lines and arrows to connect labels to their specific parts. "
    "Output ONLY the concise answer. If the information is missing, output 'Unknown'."
)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _prepare_document(doc: Document) -> dict[str, Any]:
    """Pre-compute document context (images saved, search index built)."""
    tmpdir = tempfile.mkdtemp(prefix=f"docvqa_gepa_{doc.doc_id}_")
    for i, img in enumerate(doc.images):
        img.save(os.path.join(tmpdir, f"page_{i}.png"), format="PNG")

    search_index = None
    if doc.page_texts:
        search_index = get_or_build_index(doc.doc_id, doc.page_texts)

    return {
        "document": doc,
        "tmpdir": tmpdir,
        "search_index": search_index,
    }


def evaluate(candidate: dict[str, str], example: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Evaluate a candidate on a single document.

    Returns (score, side_info) where side_info contains rich per-question
    feedback as ASI for GEPA reflection.
    """
    doc: Document = example["document"]
    tmpdir: str = example["tmpdir"]
    search_index = example["search_index"]

    # Build program with candidate instructions
    vlm_lm = VLM_CONFIG.to_dspy_lm()
    student_lm = STUDENT_LM_CONFIG.to_dspy_lm()

    program = FlatBatchProgram(vlm_lm=vlm_lm)
    program.apply_candidate({
        "task_instructions": candidate["task_instructions"],
        "vlm_instructions": candidate.get("vlm_instructions", DEFAULT_VLM_INSTRUCTIONS),
    })

    ctx = RunContext(
        page_dir=tmpdir,
        num_pages=len(doc.images),
        search_index=search_index,
        page_texts=doc.page_texts,
    )

    doc_info = f"Category: {doc.doc_category}, Pages: {len(doc.images)}"
    page_texts = _format_page_texts(doc.page_texts) if doc.page_texts else None

    # Get category tips from candidate
    tip_key = f"tip_{doc.doc_category}"
    category_tips = candidate.get(tip_key, "")

    # Scale RLM limits
    num_q = len(doc.questions)
    program.main_rlm.max_iterations = program.base_iterations + program.iterations_per_question * num_q
    program.main_rlm.max_llm_calls = program.base_llm_calls + program.llm_calls_per_question * num_q

    # Build short question IDs
    short_to_full = {}
    questions_list = []
    for i, q in enumerate(doc.questions):
        short_id = f"q{i + 1}"
        short_to_full[short_id] = q.question_id
        questions_list.append({"question_id": short_id, "question": q.question})
    questions_json = json.dumps(questions_list)
    short_ids = set(short_to_full.keys())

    # Run solver
    try:
        with dspy.context(lm=student_lm):
            result = program(
                questions=questions_json,
                doc_info=doc_info,
                page_texts=page_texts,
                category_tips=category_tips,
                run_context=ctx,
            )
        raw_answers = result.answers
    except Exception as e:
        oa.log(f"ERROR: Solver failed for {doc.doc_id}: {e}")
        return 0.0, {
            "Error": str(e),
            "Feedback": f"Solver crashed on {doc.doc_id} ({doc.doc_category}). "
                        "The task_instructions may have caused the agent to fail.",
        }

    # Parse and score
    parsed = _parse_answers(raw_answers, short_ids)
    answers_dict = {short_to_full[k]: v for k, v in parsed.items() if k in short_to_full}

    correct_count = 0
    total_scored = 0
    question_feedback = []

    for q in doc.questions:
        if q.answer is None:
            continue
        total_scored += 1
        answer = answers_dict.get(q.question_id, "Unknown")
        if not answer or answer.strip() == "":
            answer = "Unknown"
        if not answer.startswith("FINAL ANSWER:"):
            answer = "FINAL ANSWER: " + answer

        is_correct, extracted = evaluate_prediction(answer, q.answer)
        anls = get_anls(extracted.lower().strip(), q.answer.lower().strip())

        if is_correct:
            correct_count += 1
            question_feedback.append(f"  CORRECT: Q='{q.question}' -> '{extracted}'")
        else:
            if extracted.lower().strip() in ("unknown", "unkown"):
                issue = "gave up (answer exists in document)"
            elif anls > 0.5:
                issue = f"close but imprecise (ANLS={anls:.2f})"
            else:
                issue = f"wrong answer (ANLS={anls:.2f})"
            question_feedback.append(
                f"  WRONG: Q='{q.question}' predicted='{extracted}' expected='{q.answer}' — {issue}"
            )

    score = correct_count / total_scored if total_scored > 0 else 0.0

    # Build ASI (Actionable Side Information)
    feedback_lines = [
        f"Document: {doc.doc_id} (category: {doc.doc_category}, {len(doc.images)} pages, {num_q} questions)",
        f"Score: {correct_count}/{total_scored} ({score*100:.0f}%)",
        f"Answered: {len(answers_dict)}/{num_q} questions",
        "",
    ]
    feedback_lines.extend(question_feedback)

    side_info: dict[str, Any] = {
        "Feedback": "\n".join(feedback_lines),
        "scores": {"accuracy": score},
    }

    # Add per-component specific info so GEPA knows which component to blame
    tip_key = f"tip_{doc.doc_category}"
    side_info[f"{tip_key}_specific_info"] = {
        "Feedback": (
            f"Category: {doc.doc_category}\n"
            f"Score with current tips: {correct_count}/{total_scored}\n"
            + "\n".join(question_feedback)
        ),
    }

    oa.log(f"Doc {doc.doc_id}: {correct_count}/{total_scored} correct ({score*100:.0f}%)")
    for line in question_feedback:
        oa.log(line)

    return score, side_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GEPA optimization for flat batch DocVQA")
    parser.add_argument("--max-metric-calls", type=int, default=50)
    parser.add_argument("--run-dir", type=str, default="output/gepa-flat-batch")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Load dataset and split
    logger.info("Loading val dataset...")
    documents = load_documents("VLR-CVC/DocVQA-2026", "val")
    documents = [d for d in documents if any(q.answer is not None for q in d.questions)]
    logger.info("Loaded %d documents with ground truth", len(documents))

    # Prepare document contexts (save images, build indexes)
    logger.info("Preparing document contexts...")
    all_examples = []
    for doc in documents:
        all_examples.append(_prepare_document(doc))

    # Split into train/val
    train_examples = [ex for ex in all_examples if ex["document"].doc_id not in EVAL_DOC_IDS]
    val_examples = [ex for ex in all_examples if ex["document"].doc_id in EVAL_DOC_IDS]
    logger.info("Train: %d docs, Val: %d docs", len(train_examples), len(val_examples))
    for ex in val_examples:
        doc = ex["document"]
        logger.info("  VAL: %s (%s, %dq)", doc.doc_id, doc.doc_category, len(doc.questions))

    # Build seed candidate
    seed_candidate: dict[str, str] = {
        "task_instructions": TASK_INSTRUCTIONS,
        "vlm_instructions": DEFAULT_VLM_INSTRUCTIONS,
    }
    # Add per-category tips
    for cat, tips in CATEGORY_TIPS.items():
        seed_candidate[f"tip_{cat}"] = tips

    logger.info("Seed candidate components (%d):", len(seed_candidate))
    for name, text in seed_candidate.items():
        logger.info("  %s: %d chars", name, len(text))

    # Configure GEPA
    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=args.max_metric_calls,
            run_dir=args.run_dir,
            seed=args.seed,
            parallel=True,
            max_workers=2,  # Low concurrency to avoid rate limits
            display_progress_bar=True,
            use_cloudpickle=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm="vertex_ai/gemini-3-pro-preview",
            reflection_minibatch_size=3,
            module_selector="round_robin",
        ),
        tracking=TrackingConfig(
            use_wandb=True,
            wandb_init_kwargs={"project": "docvqa", "entity": "bdsaglam"},
        ),
    )

    # Run optimization
    logger.info("Starting GEPA optimization (max_metric_calls=%d)...", args.max_metric_calls)
    result = optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=evaluate,
        dataset=train_examples,
        valset=val_examples,
        objective=(
            "Optimize the instructions for a Document Visual Question Answering agent. "
            "The agent uses an LLM to write Python code that calls vision tools (look, batch_look, search) "
            "to answer questions about documents (business reports, maps, comics, engineering drawings, etc.). "
            "Maximize the accuracy of extracted answers across all document categories."
        ),
        background=(
            "The system has these optimizable text components:\n"
            "- task_instructions: The main prompt for the RLM agent that reasons about documents\n"
            "- vlm_instructions: The prompt for the VLM that reads images\n"
            "- tip_{category}: Per-category tips injected for specific document types\n\n"
            "Key constraints:\n"
            "- Answers must match exact format (dates as YYYY-MM-DD, numbers without commas, etc.)\n"
            "- The agent has limited iterations — efficiency matters\n"
            "- VLM calls are expensive — crop before querying for fine details\n"
            "- OCR text is available but may be inaccurate — visual verification is important\n"
            "- 'Unknown' should only be used when information truly doesn't exist in the document"
        ),
        config=config,
    )

    # Save results
    logger.info("GEPA optimization complete.")
    best_score = result.val_aggregate_scores[result.best_idx]
    logger.info("Best score: %.3f (candidate %d of %d)", best_score, result.best_idx, len(result.candidates))

    output_path = os.path.join(args.run_dir, "best_candidate.json")
    os.makedirs(args.run_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.best_candidate, f, indent=2)
    logger.info("Saved best candidate to %s", output_path)

    print("\n" + "=" * 60)
    print("OPTIMIZED COMPONENTS")
    print("=" * 60)
    for name, text in result.best_candidate.items():
        print(f"\n--- {name} ({len(text)} chars) ---")
        print(text[:300])
        if len(text) > 300:
            print(f"... ({len(text)} chars total)")


if __name__ == "__main__":
    main()
