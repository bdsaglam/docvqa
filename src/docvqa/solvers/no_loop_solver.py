"""No-loop baseline solver — direct VLM Q&A.

Single forward pass: question + all page images → answer. No REPL, no tools,
no agent loop. This is the "raw model" point used as a matched-baseline
ablation to show the scaffold's contribution above an unaided VLM call.

Pages are concatenated vertically into a single composite image so the call
stays a single VLM forward pass regardless of page count. If a model accepts
native multi-image input, that path is preferable but requires a different
provider integration; the composite is the defensible single-call baseline.
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
    "You are answering a question about a document. The document is shown as a single composite "
    "image with one or more pages stacked vertically. Look at the image and the question, "
    "then output a single concise answer.\n\n"
    "## OUTPUT FORMAT\n"
    "- Output ONLY the final answer string — no explanation, no preamble.\n"
    "- The answer must follow these formatting rules:\n\n"
    + ANSWER_FORMATTING_RULES
)


def _build_signature() -> dspy.Signature:
    fields: dict = {
        "question": (str, dspy.InputField(desc="The question to answer about the document")),
        "doc_info": (str, dspy.InputField(desc="Document metadata: category and page count")),
        "image": (
            dspy.Image,
            dspy.InputField(desc="Composite image of the document pages stacked vertically"),
        ),
        "answer": (str, dspy.OutputField(desc="The final concise answer string.")),
    }
    return dspy.Signature(fields, TASK_INSTRUCTIONS)


def _stack_pages(pages: list[PILImage.Image], max_height: int = 16384) -> PILImage.Image:
    """Stack pages vertically into one image. Resizes pages to a common width and
    caps total height at ``max_height`` (downscales proportionally if exceeded)."""
    if not pages:
        raise ValueError("no pages")
    resample = PILImage.Resampling.LANCZOS
    target_width = max(p.width for p in pages)
    resized: list[PILImage.Image] = []
    for p in pages:
        if p.width != target_width:
            new_h = int(p.height * target_width / p.width)
            resized.append(p.resize((target_width, new_h), resample))
        else:
            resized.append(p)
    total_h = sum(p.height for p in resized)
    composite = PILImage.new("RGB", (target_width, total_h), color=(255, 255, 255))
    y = 0
    for p in resized:
        composite.paste(p.convert("RGB"), (0, y))
        y += p.height
    if total_h > max_height:
        scale = max_height / total_h
        composite = composite.resize(
            (max(1, int(target_width * scale)), max_height), resample
        )
    return composite


class NoLoopProgram:
    """Direct VLM Q&A — one call per question, no agent loop."""

    def __init__(
        self,
        vlm_lm: dspy.LM,
        question_concurrency: int = 4,
        max_height: int = 16384,
    ):
        self.vlm_lm = vlm_lm
        self.question_concurrency = question_concurrency
        self.max_height = max_height
        self.predict = dspy.Predict(_build_signature())

    def solve_document(self, document: Document) -> tuple[dict[str, str], dict[str, list[dict]]]:
        composite = _stack_pages(document.images, max_height=self.max_height)
        composite_dspy = dspy.Image(composite)
        doc_info = f"Category: {document.doc_category}, Pages: {len(document.images)}"

        def _solve_question(q):
            with logfire.span(
                "solve_no_loop",
                doc_id=document.doc_id,
                question_id=q.question_id,
                question=q.question[:200],
            ) as q_span:

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
                def _call():
                    with dspy.context(lm=self.vlm_lm):
                        return self.predict(
                            question=q.question,
                            doc_info=doc_info,
                            image=composite_dspy,
                        )

                try:
                    result = _call()
                    answer = str(result.answer or "").strip()
                except Exception as e:
                    logger.warning("No-loop failed for Q '%s': %s", q.question_id, e)
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
                        "NoLoop Q %s: %s (GT=%s, PRED=%s)",
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
                "NoLoop doc %s: %d/%d = %.1f%%",
                document.doc_id, correct, scored, 100 * correct / scored,
            )

        return predictions, trajectories


def create_no_loop_program(
    vlm: dict[str, Any] | None = None,
    question_concurrency: int = 4,
    max_height: int = 16384,
) -> NoLoopProgram:
    vlm_config = LMConfig(
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
    ) if vlm and vlm.get("model") else LMConfig()
    vlm_lm = vlm_config.to_dspy_lm()
    return NoLoopProgram(
        vlm_lm=vlm_lm,
        question_concurrency=question_concurrency,
        max_height=max_height,
    )
