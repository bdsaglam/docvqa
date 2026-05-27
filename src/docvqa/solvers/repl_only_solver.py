"""REPL-only ablation per D-006 prediction 3.

Fork of ``leanest_solo_solver`` with the VLM sub-call (``batch_look``)
removed. The agent retains the REPL + Python execution + ``SUBMIT``
affordance but has NO image perception (and no OCR text channel — by
design, this is a *pure* no-perception ablation, not an "OCR-only" one).

Predicted behavior: the agent collapses to roughly the no-loop baseline
(~17-25% val on DocVQA-2026), because it cannot see the document. This
tests whether the recursive VLM sub-call is the load-bearing mechanism
of the scaffold (predictions 1 & 2 in D-006 already vary lift along
the model-size and document-length axes).

Per D-007 (2026-05-27), this solver owns its category-tip prompts
inline (``CATEGORY_TIPS`` below). The shared dicts in
``docvqa.prompts`` are deprecated for paper solvers — do not import
them here. The tips are reconciled from leanest's reconciled dict
with all vision-channel references (``batch_look``, crop, zoom, tile,
pixel sizes, etc.) stripped. What remains is the semantic shell that's
still meaningful without perception (e.g. UNKNOWN rules, named-entity
match, PART vs ITEM number, "broken down into" semantics). Some
categories collapse to near-empty after stripping — that's expected
for an ablation; no re-tuning is done.
"""

from __future__ import annotations

import logging
import math

import dspy
import logfire
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from docvqa.data import Document
from docvqa.metrics import evaluate_prediction
from docvqa.prompts import ANSWER_FORMATTING_RULES
from docvqa.rlm import LeanRLM, CodeRLM, ThinkingRLM, RLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

TASK_INSTRUCTIONS = (
    "You are a Document Visual Question Answering agent operating under a "
    "REPL-only configuration: you have Python code execution and a SUBMIT "
    "affordance, but NO image perception tools are available in this run.\n\n"

    "## DATA\n"
    "- `question`: The question you must answer.\n"
    "- `doc_info`: Document metadata (category and page count). No image or text content of the document itself is exposed.\n\n"

    "## TOOLS\n"
    "- Python REPL: standard library is available for any pure-Python reasoning, arithmetic, or string manipulation.\n"
    "- No vision tools (no `look`, no `batch_look`, no image access).\n"
    "- No document text tools (no `search`, no `page_texts`, no OCR access).\n\n"

    "## APPROACH\n"
    "Because the document is not visible to you in this configuration, you cannot extract facts from it. "
    "Do not guess or hallucinate values from the question text or category alone — answers MUST come from "
    "the document, and the document is not accessible here.\n\n"

    "1. If the question can be answered without perceiving the document (extremely rare — usually only "
    "tautological or self-referential questions qualify), reason it out in Python and SUBMIT the answer.\n"
    "2. Otherwise, SUBMIT 'Unknown'. This is the correct answer when the requested information cannot be "
    "verified against the document.\n\n"

    "## GUIDELINES\n"
    "- NEVER use outside/world knowledge. ALL answers MUST come from the document, and you cannot read the document.\n"
    "- Do NOT fabricate plausible-sounding values, dates, names, or numbers. 'Unknown' is preferred over a guess.\n"
    "- Do NOT extrapolate from the question's phrasing or the document category. A question about a 'maps' "
    "document does not authorize you to invent map contents.\n"
    "- COMPUTATION caveat: even when a question requests an arithmetic operation (sum, total, difference), "
    "you need the underlying values from the document, which you cannot access here — SUBMIT 'Unknown'.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string.\n"
    '- Example: SUBMIT(answer="Unknown")\n'
    "- If an answer is genuinely warranted, it must follow these formatting rules:\n\n"

    + ANSWER_FORMATTING_RULES
)


# ---------------------------------------------------------------------------
# Category-specific tips (owned inline per D-007).
#
# Derived from leanest's reconciled CATEGORY_TIPS with all vision-channel
# references stripped (no `batch_look`, no crop, no zoom, no tile, no
# pixel sizes, no per-region perception verbs). What remains is the
# semantic shell that's still meaningful without perception: named-entity
# match rules, ontology hints (PART vs ITEM number), UNKNOWN rules,
# percentage-point semantics, "broken down into" semantics, literal vs
# figurative qualifiers, etc.
#
# Some categories collapse to near-empty after stripping (e.g. maps,
# science_poster, infographics). That's expected — this is an ablation,
# not a re-tuned solver. We keep the keys present for parity with
# leanest's surface; an empty value would mean "no semantic content
# survives without perception."
#
# REPL-only tool surface: REPL + SUBMIT. No `batch_look()`, no `look()`,
# no `search()`, no `page_texts`. References to those tools are NOT in
# this dict.
# ---------------------------------------------------------------------------

CATEGORY_TIPS: dict[str, str] = {
    "engineering_drawing": (
        "- BOM has two parallel numbering systems: ITEM NUMBERS (sequential index in the parts list) and "
        "PART / IDENTIFYING NUMBERS (the actual hardware identifier, often alphanumeric with dashes). "
        "Questions about 'part number' / 'identifying number' refer to the latter; 'item number' refers to the former.\n"
        "- 'VIEW IN DIRECTION X' labels indicate a viewing direction. The answer is the direction letter alone, "
        "not prefixed with 'Direction'.\n"
        "- DIMENSIONS: 'Width' typically refers to the shorter cross-sectional dimension (from a Section view), "
        "not the longest overall dimension (which is 'Length'). Dimensions tagged 'REF' (reference) are valid answers.\n"
        "- Without perception, you cannot read part numbers, BOM rows, or dimension values. SUBMIT 'Unknown' for "
        "any question that requires reading the drawing.\n"
    ),
    "business_report": (
        "- 'Broken down into' refers to immediate sub-categories only, not sub-sub-categories.\n"
        "- TEXT TRUNCATION: When a question asks for a phrase truncated at a punctuation boundary "
        "(first words before a punctuation mark, first sentence, etc.), the underlying passage must come from "
        "the document. Without perception, that passage is not available — SUBMIT 'Unknown'.\n"
        "- Without perception, you cannot read tables, chart values, or text passages. SUBMIT 'Unknown' for "
        "any question that requires extracting a value or phrase from the report.\n"
    ),
    "comics": (
        "- LITERAL VS FIGURATIVE: When a question contains qualifiers like 'in reality', 'actually', or 'truly', "
        "the answer likely contradicts the surface label/title shown in the panel. Distinguishing the two still "
        "requires reading the panels, which is not available here.\n"
        "- Without perception, you cannot identify characters, count events, or read dialogue. SUBMIT 'Unknown' "
        "for any question that requires inspecting the comic.\n"
    ),
    "maps": (
        "- Without perception, you cannot count objects on the map, identify landmarks, read legends, or "
        "determine grid coordinates. SUBMIT 'Unknown' for any question that requires inspecting the map.\n"
    ),
    "science_paper": (
        "- If a question references a specific entity (layer number, model variant, dataset name) that requires "
        "checking the document, and the document is not accessible, answer 'Unknown' — do not extrapolate from "
        "a similar-sounding entity, and do not draw on outside knowledge of the paper or its authors.\n"
        "- CITED PAPER FINDINGS: To find what a cited work claims, you would need the bibliography and the body "
        "text. Without perception, neither is accessible — SUBMIT 'Unknown' rather than hallucinating from "
        "the cited work's title or your training knowledge.\n"
        "- Without perception, you cannot read citations, abstracts, ablation tables, or figures. SUBMIT "
        "'Unknown' for any question that requires reading the paper.\n"
    ),
    "science_poster": (
        "- 'Percentage improvement' refers to the absolute difference in percentage points (e.g., 80% − 50% "
        "= 30 percentage points), not the relative change. (Definitional — independent of perception.)\n"
        "- GROUPED BAR CHARTS: A 'set of columns' / 'group of bars' refers to the bars at one x-axis position "
        "(one category, one benchmark), not all bars of one color across positions. (Definitional.)\n"
        "- Without perception, you cannot read chart values, table cells, or annotations. SUBMIT 'Unknown' "
        "for any question that requires extracting a value from the poster.\n"
    ),
    "infographics": (
        "- SYSTEMATIC ENUMERATION: When a question asks for a first/last/only item that has or lacks some "
        "property, the answer requires enumerating items in the document. Without perception, that enumeration "
        "is not possible — SUBMIT 'Unknown'.\n"
        "- Without perception, you cannot read icons, sections, or data points. SUBMIT 'Unknown' for any "
        "question that requires inspecting the infographic.\n"
    ),
    "slide": (
        "- EXACT ENTITY MATCHING: If a question references a specific column name, variable, or equation "
        "that requires checking the slide, and the slide is not accessible, answer 'Unknown'. Do NOT "
        "substitute a similar-sounding name.\n"
        "- COMPUTATION: When a question says 'total', 'sum', or 'considering X and Y', the underlying values "
        "must come from the slides. Without perception, those values are not accessible — SUBMIT 'Unknown'.\n"
        "- Without perception, you cannot read slide titles, tables, or body text. SUBMIT 'Unknown' for any "
        "question that requires reading the deck.\n"
    ),
}


def _get_category_tips(category: str) -> str:
    """Get per-category tips for a document type. Returns empty string if none."""
    tips = CATEGORY_TIPS.get(category, "")
    if tips:
        return f"## CATEGORY-SPECIFIC TIPS ({category})\n{tips}"
    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_signature(instructions: str = TASK_INSTRUCTIONS) -> dspy.Signature:
    fields: dict = {
        "question": (
            str,
            dspy.InputField(desc="The question to answer about the document"),
        ),
        "doc_info": (str, dspy.InputField(desc="Document metadata: category and page count")),
        "answer": (
            str,
            dspy.OutputField(desc="The answer string for the question."),
        ),
    }
    return dspy.Signature(fields, instructions)


# No tools are exposed in the REPL-only ablation. The RLM injects SUBMIT
# automatically; we deliberately add no `batch_look`, no `look`, no
# `search`, no `page_texts`.
def _create_tools() -> list:
    return []


# No sandbox bootstrap is needed for REPL-only — there are no images to
# load and no helpers to define. The RLM's interpreter starts clean with
# stdlib available.
def _build_sandbox_code() -> str | None:
    return None


# ---------------------------------------------------------------------------
# ReplOnlyProgram
# ---------------------------------------------------------------------------

class ReplOnlyProgram:
    """REPL-only ablation — each question solved with Python execution but no perception."""

    def __init__(
        self,
        max_iterations: int = 20,
        rlm_type: str = "lean",
        page_factor: float = 1.5,
        question_concurrency: int = 1,
    ):
        self.max_iterations = max_iterations
        self.rlm_type = rlm_type
        self.page_factor = page_factor
        self.question_concurrency = question_concurrency

    def solve_document(self, document: Document) -> tuple[dict[str, str], dict[str, list[dict]]]:
        """Solve all questions for a document, one question at a time, with no perception."""
        doc_info = f"Category: {document.doc_category}, Pages: {len(document.images)}"

        num_pages = len(document.images)
        page_bonus = min(10, self.page_factor * math.ceil(math.sqrt(max(0, num_pages - 9))))
        max_iter = self.max_iterations + int(page_bonus)

        tips = _get_category_tips(document.doc_category)
        instructions = TASK_INSTRUCTIONS + ("\n" + tips if tips else "")
        tools = _create_tools()
        sandbox_code = _build_sandbox_code()

        def _solve_question(q):
            """Solve a single question. Returns (question_id, answer, trajectory)."""
            with logfire.span(
                "solve_repl_only",
                doc_id=document.doc_id,
                question_id=q.question_id,
                question=q.question[:200],
            ) as q_span:
                RLMClass = {"code": CodeRLM, "lean": LeanRLM, "thinking": ThinkingRLM}.get(self.rlm_type, RLM)
                rlm = RLMClass(
                    signature=_build_signature(instructions),
                    max_iterations=max_iter,
                    max_llm_calls=max_iter * 3,
                    tools=tools,
                    verbose=True,
                    sandbox_code=sandbox_code,
                )
                logger.info(
                    "REPL-only (%s) Q %s: max_iterations=%d (page_bonus=%d)",
                    self.rlm_type, q.question_id, max_iter, int(page_bonus),
                )

                def _is_rate_limit(e: BaseException) -> bool:
                    return "429" in str(e) or "RateLimit" in type(e).__name__ or "RESOURCE_EXHAUSTED" in str(e)

                @retry(
                    retry=retry_if_exception(_is_rate_limit),
                    stop=stop_after_attempt(4),
                    wait=wait_exponential(multiplier=30, min=30, max=120),
                    before_sleep=lambda rs: logger.warning(
                        "Rate limit, retry %d in %.0fs", rs.attempt_number, rs.next_action.sleep  # type: ignore[union-attr]
                    ),
                    reraise=True,
                )
                def _solve_one():
                    return rlm(
                        question=q.question,
                        doc_info=doc_info,
                    )

                result = _solve_one()
                answer = str(result.answer or "").strip()
                trajectory = result.trajectory

                if not answer:
                    answer = "Unknown"

                q_span.set_attribute("num_iterations", len(trajectory))
                q_span.set_attribute("prediction", answer[:200])

                if q.answer is not None:
                    is_correct, extracted = evaluate_prediction(answer, q.answer)
                    q_span.set_attribute("is_correct", is_correct)
                    q_span.set_attribute("ground_truth", q.answer[:200])
                    q_span.set_attribute("extracted_answer", extracted[:200])
                    logger.info(
                        "REPL-only Q %s: %s (GT=%s, PRED=%s)",
                        q.question_id,
                        "CORRECT" if is_correct else "WRONG",
                        q.answer[:40],
                        extracted[:40],
                    )

                return q.question_id, answer, trajectory

        # Run questions with configurable concurrency
        predictions = {}
        trajectories = {}
        correct_count = 0
        scored_count = 0

        if self.question_concurrency <= 1:
            for q in document.questions:
                qid, answer, trajectory = _solve_question(q)
                predictions[qid] = answer
                trajectories[qid] = trajectory
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_w = min(self.question_concurrency, len(document.questions))
            logger.info("REPL-only: running %d questions with concurrency=%d", len(document.questions), max_w)
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                futures = {pool.submit(_solve_question, q): q for q in document.questions}
                for future in as_completed(futures):
                    qid, answer, trajectory = future.result()
                    predictions[qid] = answer
                    trajectories[qid] = trajectory

        # Score
        for q in document.questions:
            if q.answer is not None:
                scored_count += 1
                is_correct, _ = evaluate_prediction(predictions[q.question_id], q.answer)
                if is_correct:
                    correct_count += 1

        if scored_count > 0:
            logger.info(
                "REPL-only doc %s: %d/%d = %.1f%%",
                document.doc_id, correct_count, scored_count,
                100 * correct_count / scored_count,
            )

        return predictions, trajectories


# ---------------------------------------------------------------------------
# Factory for hydra instantiation
# ---------------------------------------------------------------------------

def create_repl_only_program(
    max_iterations: int = 20,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
) -> ReplOnlyProgram:
    """Factory for the REPL-only solver. No VLM is configured or used."""
    return ReplOnlyProgram(
        max_iterations=max_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
    )
