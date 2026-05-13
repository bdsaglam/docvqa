"""Official DocVQA 2026 baseline solver — multi-image VLM with the
competition's published `MASTER_PROMPT`.

This is the *literal* official baseline: send all document pages to the
VLM in one chat-completion request, use the kit's `MASTER_PROMPT` (which
enforces the competition's mandatory reasoning protocol and FINAL ANSWER:
output format), and extract the text after `FINAL ANSWER:`. No category
tips, no truncation by default (matches how the kit's Gemini/GPT
baselines are scored in the README results table).

The prompt is vendored verbatim from
`tmp/DocVQA2026/eval_utils.py:get_evaluation_prompt()` to keep this file
self-contained and ensure it doesn't drift from the official version
silently — when the kit updates, refresh this constant manually.

Reference: https://github.com/VLR-CVC/DocVQA2026
"""

from __future__ import annotations

import logging
import re
from typing import Any

import dspy
import logfire
from PIL import Image as PILImage
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# Match the official DocVQA 2026 kit (eval_utils.py) — no decompression-bomb
# limit. As of 2026-05-13 the largest test page is 246M pixels (maps_5 p0)
# which already fits under our other solvers' 500M cap (set in data.py), so
# this is a defensive parity setting, not a fix to an active problem.
PILImage.MAX_IMAGE_PIXELS = None

from docvqa.data import Document
from docvqa.metrics import evaluate_prediction
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)


# Verbatim from DocVQA2026/eval_utils.py get_evaluation_prompt().
# Refresh manually if upstream changes.
MASTER_PROMPT = (
    "ACT AS an expert Document Visual Question Answering (DocVQA) system. "
    "ANALYZE the provided images to extract precise information.\n\n"
    "### MANDATORY RESPONSE RULES:\n"
    '1. SOURCE ADHERENCE: If the question is unanswerable from the document, respond ONLY with "Unknown".\n'
    '2. LIST FORMATTING: List multiple answers in order of appearance, separated by a comma and a single space (e.g., "Answer A, Answer B"). Do NOT use "and".\n'
    "3. NUMBERS & UNITS:\n"
    '   - Convert units to their standardized abbreviation (e.g., use "kg" not "kilograms", "m" not "meters").\n'
    '   - Place a single space between the number and the unit (e.g., "50 kg", "10 USD").\n'
    "4. PERCENTAGES: For percentages, attach the '%' symbol directly to the number with NO space (e.g., \"50%\", not \"50 %\").\n"
    '5. DATE FORMATTING: Convert all dates to YYYY-MM-DD format (e.g., convert "Jan 1st 24" to "2024-01-01").\n'
    '6. DECIMAL FORMATTING: Decimals should be separated by a single period (e.g., "3.14", not "3,14").\n'
    '7. THOUSANDS SEPARATOR: Do NOT use commas as thousands separators (e.g., "1000", not "1,000").\n'
    '8. NO FILLER: Output ONLY the result. Do not frame with sentences like "The answer is...".'
    "\n\n### REASONING PROTOCOL:\n"
    "1. Perform exhaustive step-by-step reasoning to locate and verify the data.\n"
    "2. Verify if the data contains a date, number, or unit.\n"
    "3. Step-by-step, transform the data to match the MANDATORY RESPONSE RULES (e.g., converting date format).\n"
    "\n\n### OUTPUT FORMAT:\n"
    "After your analysis, you MUST provide the final result in the following format:\n"
    "FINAL ANSWER: [Your exact formatted answer]\n"
    "Ensure the content inside [FINAL ANSWER] strictly follows the MANDATORY RESPONSE RULES."
)


def _extract_final_answer(raw: str) -> str:
    """Return the text after the last 'FINAL ANSWER:' marker, stripped of
    bracket framing. Falls back to the full string if no marker found.

    The marker is matched case-insensitively; the kit's prompt asks for
    'FINAL ANSWER:' verbatim, but models occasionally vary capitalisation
    or add brackets / asterisks around the marker.
    """
    if not raw:
        return ""
    text = raw.strip()
    # Find the LAST occurrence (some models echo the prompt's example)
    matches = list(re.finditer(r"FINAL\s*ANSWER\s*:?\s*", text, flags=re.IGNORECASE))
    if not matches:
        return text  # no marker → return raw
    tail = text[matches[-1].end() :].strip()
    # Strip surrounding brackets, asterisks, quotes
    tail = tail.lstrip("[*\"' ").rstrip(" ]*\"'.\n")
    # If multi-line, prefer the first non-empty line (the answer should be terse)
    for line in tail.splitlines():
        line = line.strip().lstrip("[*\"' ").rstrip(" ]*\"'.")
        if line:
            return line
    return tail


def _build_messages(
    question: str,
    pages: list[PILImage.Image],
) -> list[dict[str, Any]]:
    """Single user message: master prompt + page images + question."""
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": MASTER_PROMPT},
    ]
    for p in pages:
        formatted = dspy.Image(p).format()
        if isinstance(formatted, list):
            parts.extend(formatted)
        else:
            parts.append({"type": "image_url", "image_url": {"url": formatted}})
    parts.append({"type": "text", "text": f"\n\nQuestion: {question}"})
    return [{"role": "user", "content": parts}]


class OfficialBaselineProgram:
    """Multi-image VLM with the official DocVQA 2026 MASTER_PROMPT.

    No category tips, no agent loop, no tools. Matches the framing of
    the published Gemini/GPT baselines in the kit README results table.
    """

    def __init__(
        self,
        vlm_lm: dspy.LM,
        question_concurrency: int = 4,
        max_pages: int | None = None,
    ):
        self.vlm_lm = vlm_lm
        self.question_concurrency = question_concurrency
        # max_pages=None => send all pages. Set a value to truncate.
        self.max_pages = max_pages

    def solve_document(
        self, document: Document
    ) -> tuple[dict[str, str], dict[str, list[dict]]]:
        pages = document.images
        if self.max_pages is not None and self.max_pages < len(pages):
            pages = pages[: self.max_pages]
        truncated = len(pages) < len(document.images)

        def _solve_question(q):
            with logfire.span(
                "solve_official_baseline",
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
                def _call() -> str:
                    messages = _build_messages(q.question, pages)
                    response: Any = self.vlm_lm.forward(messages=messages)
                    msg = response.choices[0].message
                    text = msg.content
                    if not text:
                        text = getattr(msg, "reasoning_content", "") or ""
                    return str(text or "").strip()

                try:
                    raw = _call()
                except Exception as e:
                    logger.warning("OfficialBaseline failed for Q '%s': %s", q.question_id, e)
                    return q.question_id, "Unknown", []

                answer = _extract_final_answer(raw) or "Unknown"
                q_span.set_attribute("raw_response", raw[:2000])
                q_span.set_attribute("prediction", answer[:200])

                if q.answer is not None:
                    is_correct, extracted = evaluate_prediction(answer, q.answer)
                    q_span.set_attribute("is_correct", is_correct)
                    q_span.set_attribute("ground_truth", q.answer[:200])
                    q_span.set_attribute("extracted_answer", extracted[:200])
                    logger.info(
                        "OfficialBaseline Q %s: %s (GT=%s, PRED=%s)",
                        q.question_id,
                        "CORRECT" if is_correct else "WRONG",
                        q.answer[:40],
                        extracted[:40],
                    )

                return q.question_id, answer, [{"raw": raw, "extracted": answer}]

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
                "OfficialBaseline doc %s: %d/%d = %.1f%% (pages_used=%d/%d%s)",
                document.doc_id,
                correct,
                scored,
                100 * correct / scored,
                len(pages),
                len(document.images),
                ", truncated" if truncated else "",
            )

        return predictions, trajectories


def create_official_baseline_program(
    vlm: dict[str, Any] | None = None,
    question_concurrency: int = 4,
    max_pages: int | None = None,
) -> OfficialBaselineProgram:
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
    return OfficialBaselineProgram(
        vlm_lm=vlm_lm,
        question_concurrency=question_concurrency,
        max_pages=max_pages,
    )
