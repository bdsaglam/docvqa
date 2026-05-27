"""No-loop multi-image baseline solver — direct VLM Q&A with native multi-image input.

Same single-call, no-REPL, no-tools, no-agent shape as ``no_loop_solver`` but
sends the document as a sequence of page images (up to ``max_pages``) in one
chat-completion request instead of concatenating them into a single composite
that gets crushed by a height cap. This tests whether the original no-loop
baseline's weakness on long docs is an artifact of composite-image rescaling
rather than a model-reasoning failure.

Truncation policy: take the first ``max_pages`` pages. Documents longer than
that are *not* downscaled — we send the head at native resolution, which is
the most defensible "raw VLM with limited budget" framing. Strided sampling or
relevance-aware selection would itself be an agent decision.

Per D-007 (docs/paper/decisions.md, 2026-05-27), this solver owns its
category-tip prompts inline (`BASELINE_CATEGORY_TIPS` below). The shared
dicts in ``docvqa.prompts`` are deprecated for paper solvers — do not
import them here.
"""

from __future__ import annotations

import logging
from typing import Any

import dspy
import logfire
from PIL import Image as PILImage
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from docvqa.data import Document
from docvqa.metrics import evaluate_prediction
from docvqa.prompts import ANSWER_FORMATTING_RULES
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)


TASK_INSTRUCTIONS = (
    "You are answering a question about a document. The document is shown as a sequence "
    "of page images, in order (page 0 first, page 1 next, etc.). Look at the images and "
    "the question, then output a single concise answer.\n\n"
    "## OUTPUT FORMAT\n"
    "- Output ONLY the final answer string — no explanation, no preamble.\n"
    "- The answer must follow these formatting rules:\n\n"
    + ANSWER_FORMATTING_RULES
)


# ---------------------------------------------------------------------------
# Category-specific tips (owned inline per D-007).
# Single-shot baseline surface — no REPL, no tools, no agent loop. Only
# semantic and question-interpretation hints survive; tool-routing verbs
# (crop/zoom/search/REPL/Python/batch_look) are stripped.
# Reconciled with paper-solver canonical content on 2026-05-27.
# ---------------------------------------------------------------------------

BASELINE_CATEGORY_TIPS: dict[str, str] = {
    "engineering_drawing": (
        "- BOM has two parallel numbering systems: ITEM NUMBERS (sequential index in the parts list) and "
        "PART / IDENTIFYING NUMBERS (the actual hardware identifier, often alphanumeric with dashes). "
        "Questions about 'part number' / 'identifying number' refer to the latter; 'item number' refers to "
        "the former.\n"
        "- 'VIEW IN DIRECTION X' labels indicate a viewing direction. The answer is the direction letter "
        "alone, not prefixed with 'Direction'.\n"
        "- OCR CONFUSION: Part numbers are almost always digits + dashes. Common confusions: I↔1, O↔0, l↔1.\n"
        "- LEADER LINES: when a label points to a part via a leader line, verify each label is correctly "
        "associated with the part it connects to — follow the line, not just proximity on the page.\n"
        "- DIMENSIONS: 'Width' typically refers to the shorter cross-sectional dimension (from a Section "
        "view), not the longest overall dimension (which is 'Length'). Dimensions tagged 'REF' (reference) "
        "are valid answers.\n"
    ),
    "business_report": (
        "- Multiple tables may contain similar-looking data. Verify the table you're reading matches the "
        "question's subject before extracting values.\n"
        "- 'Broken down into' refers to immediate sub-categories only, not sub-sub-categories.\n"
        "- TEXT TRUNCATION: For a phrase truncated at a punctuation boundary (first words before a "
        "punctuation mark, first sentence, etc.), read the full passage and do the truncation yourself — "
        "do not over-shorten.\n"
        "- If a qualitative description (e.g., an adjective) does not appear in a table, it may be in "
        "surrounding text paragraphs or footnotes.\n"
    ),
    "comics": (
        "- For multi-story anthologies, each story has its own title, page range, and characters. Match "
        "question keywords to the correct story.\n"
        "- LITERAL VS FIGURATIVE: When a question contains qualifiers like 'in reality', 'actually', or "
        "'truly', the answer likely contradicts the surface label/title — distinguish what something is "
        "called from what it factually is.\n"
        "- CHARACTER IDENTIFICATION: Use the exact term that appears in the speech bubbles when available.\n"
        "- For COUNTING EVENTS, use strict inclusion criteria — exclude near-misses, past events referenced "
        "in dialogue, and aftermath. Sound effects or weapons in a panel do not by themselves prove an "
        "action occurred.\n"
    ),
    "maps": (
        "- LEGEND: Map symbols and line styles are defined in the legend. For road-type questions, compare "
        "the line style of the specific road segment to legend entries.\n"
        "- COUNTING OBJECTS ON MAPS: For 'how many X are on the map', do not estimate from a single glance — "
        "scan the map systematically (region by region), list each candidate object with an approximate "
        "position, then count the unique objects.\n"
        "- GRID COORDINATES: Cross-reference what is visible in the grid cell with any feature index that "
        "lists entries by grid coordinate.\n"
    ),
    "science_paper": (
        "- CITATION NUMBERS: Citations appear as [N] (or (Author, Year)) in body text. Distinguish body-text "
        "citations from table headers and figure captions, which are often numbered separately.\n"
        "- CITED PAPER FINDINGS: To find what a cited work claims, locate the reference number in the "
        "bibliography, then find where that number is discussed in the body text. If the cited paper's "
        "actual content isn't in this document, answer 'Unknown' rather than hallucinating from the title.\n"
        "- ABLATION STUDIES: Papers often have multiple ablation studies on different components. Verify "
        "the section you're reading is about the specific component the question asks about, not a "
        "different subsystem.\n"
        "- If a question references a specific entity (layer number, model variant, dataset name) that "
        "does not appear in the document, answer 'Unknown' — do not extrapolate from a similar-sounding "
        "entity.\n"
    ),
    "science_poster": (
        "- CHART ANNOTATIONS: If a chart has numeric labels printed directly on bars/lines, use those "
        "labels rather than estimating from bar heights.\n"
        "- 'Percentage improvement' refers to the absolute difference in percentage points (e.g., 80% − "
        "50% = 30 percentage points), not the relative change.\n"
        "- GROUPED BAR CHARTS: A 'set of columns' / 'group of bars' refers to the bars at one x-axis "
        "position (one category, one benchmark), not all bars of one color across positions.\n"
    ),
    "infographics": (
        "- SYSTEMATIC ENUMERATION: When a question asks for a first/last/only item that has or lacks some "
        "property, enumerate ALL items and their status before answering — don't stop after finding a few.\n"
    ),
    "slide": (
        "- PAGE NAVIGATION: When a question refers to 'the page before X' or 'the page that contains Y', "
        "locate X or Y in the document, then take the page directly preceding or containing it. Off-by-one "
        "errors on page indexing are common — verify by checking page headers/titles.\n"
        "- EXACT ENTITY MATCHING: If a question references a specific column name, variable, or equation "
        "that does not exist in the document, answer 'Unknown'. Do NOT substitute a similar-sounding name.\n"
        "- COMPUTATION: When a question says 'total', 'sum', or 'considering X and Y', extract all "
        "referenced values and compute the result explicitly before deciding.\n"
    ),
}


def _get_category_tips(category: str) -> str:
    """Get baseline-adapted tips for a single-shot VLM call (no agent verbs)."""
    tips = BASELINE_CATEGORY_TIPS.get(category, "")
    if tips:
        return f"## CATEGORY-SPECIFIC TIPS ({category})\n{tips}"
    return ""


def _build_messages(
    question: str,
    doc_info: str,
    pages: list[PILImage.Image],
    instructions: str = TASK_INSTRUCTIONS,
) -> list[dict[str, Any]]:
    """Build one user message: instructions + interleaved page labels and images."""
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": f"{instructions}\n\nDocument metadata: {doc_info}\n"},
    ]
    for i, p in enumerate(pages):
        parts.append({"type": "text", "text": f"\n[Page {i}]"})
        formatted = dspy.Image(p).format()
        if isinstance(formatted, list):
            parts.extend(formatted)
        else:
            parts.append({"type": "image_url", "image_url": {"url": formatted}})
    parts.append({"type": "text", "text": f"\nQuestion: {question}\n\nAnswer:"})
    return [{"role": "user", "content": parts}]


class NoLoopMultiProgram:
    """Direct VLM Q&A — multi-image, one call per question."""

    def __init__(
        self,
        vlm_lm: dspy.LM,
        question_concurrency: int = 4,
        max_pages: int = 10,
        use_category_tips: bool = True,
    ):
        self.vlm_lm = vlm_lm
        self.question_concurrency = question_concurrency
        self.max_pages = max_pages
        self.use_category_tips = use_category_tips

    def solve_document(
        self, document: Document
    ) -> tuple[dict[str, str], dict[str, list[dict]]]:
        pages = document.images[: self.max_pages]
        truncated = len(pages) < len(document.images)
        doc_info = (
            f"Category: {document.doc_category}, "
            f"Pages shown: {len(pages)} of {len(document.images)}"
            + (" (truncated to first N pages)" if truncated else "")
        )
        if self.use_category_tips:
            tips = _get_category_tips(document.doc_category)
            instructions = TASK_INSTRUCTIONS + ("\n" + tips if tips else "")
        else:
            instructions = TASK_INSTRUCTIONS

        def _solve_question(q):
            with logfire.span(
                "solve_no_loop_multi",
                doc_id=document.doc_id,
                question_id=q.question_id,
                question=q.question[:200],
            ) as q_span:

                def _is_rate_limit(e: BaseException) -> bool:
                    return (
                        "429" in str(e)
                        or "RateLimit" in type(e).__name__
                        or "RESOURCE_EXHAUSTED" in str(e)
                    )

                @retry(
                    retry=retry_if_exception(_is_rate_limit),
                    stop=stop_after_attempt(4),
                    wait=wait_exponential(multiplier=30, min=30, max=120),
                    before_sleep=lambda rs: logger.warning(
                        "Rate limit, retry %d in %.0fs",
                        rs.attempt_number,
                        rs.next_action.sleep,  # type: ignore[union-attr]
                    ),
                    reraise=True,
                )
                def _call():
                    messages = _build_messages(q.question, doc_info, pages, instructions)
                    response: Any = self.vlm_lm.forward(messages=messages)
                    msg = response.choices[0].message
                    text = msg.content
                    if not text:
                        text = getattr(msg, "reasoning_content", "") or ""
                    return str(text or "").strip()

                try:
                    answer = _call() or "Unknown"
                except Exception as e:
                    logger.warning("NoLoopMulti failed for Q '%s': %s", q.question_id, e)
                    answer = "Unknown"

                if not answer:
                    answer = "Unknown"

                q_span.set_attribute("prediction", answer[:200])

                if q.answer is not None:
                    is_correct, extracted = evaluate_prediction(answer, q.answer)
                    q_span.set_attribute("is_correct", is_correct)
                    q_span.set_attribute("ground_truth", q.answer[:200])
                    q_span.set_attribute("extracted_answer", extracted[:200])
                    logger.info(
                        "NoLoopMulti Q %s: %s (GT=%s, PRED=%s)",
                        q.question_id,
                        "CORRECT" if is_correct else "WRONG",
                        q.answer[:40],
                        extracted[:40],
                    )

                return q.question_id, answer, []

        predictions: dict[str, str] = {}
        trajectories: dict[str, list[dict]] = {}

        if self.question_concurrency <= 1:
            for q in document.questions:
                qid, answer, traj = _solve_question(q)
                predictions[qid] = answer
                trajectories[qid] = traj
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            max_w = min(self.question_concurrency, len(document.questions))
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                futures = {pool.submit(_solve_question, q): q for q in document.questions}
                for future in as_completed(futures):
                    qid, answer, traj = future.result()
                    predictions[qid] = answer
                    trajectories[qid] = traj

        correct = 0
        scored = 0
        for q in document.questions:
            if q.answer is not None:
                scored += 1
                if evaluate_prediction(predictions[q.question_id], q.answer)[0]:
                    correct += 1
        if scored > 0:
            logger.info(
                "NoLoopMulti doc %s: %d/%d = %.1f%%",
                document.doc_id,
                correct,
                scored,
                100 * correct / scored,
            )

        return predictions, trajectories


def create_no_loop_multi_program(
    vlm: dict[str, Any] | None = None,
    question_concurrency: int = 4,
    max_pages: int = 10,
    use_category_tips: bool = True,
) -> NoLoopMultiProgram:
    vlm_config = (
        LMConfig(
            model=vlm["model"],
            api_base=vlm.get("api_base"),
            api_key=vlm.get("api_key"),
            max_tokens=vlm.get("max_tokens", 65536),
            temperature=vlm.get("temperature", 1.0),
            top_p=vlm.get("top_p"),
            top_k=vlm.get("top_k"),
            presence_penalty=vlm.get("presence_penalty"),
            enable_thinking=vlm.get("enable_thinking", False),
            vertex_location=vlm.get("vertex_location"),
        )
        if vlm and vlm.get("model")
        else LMConfig()
    )
    vlm_lm = vlm_config.to_dspy_lm()
    return NoLoopMultiProgram(
        vlm_lm=vlm_lm,
        question_concurrency=question_concurrency,
        max_pages=max_pages,
        use_category_tips=use_category_tips,
    )
