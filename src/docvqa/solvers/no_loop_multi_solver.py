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


def _build_messages(
    question: str, doc_info: str, pages: list[PILImage.Image]
) -> list[dict[str, Any]]:
    """Build one user message: instructions + interleaved page labels and images."""
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": f"{TASK_INSTRUCTIONS}\n\nDocument metadata: {doc_info}\n"},
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
    ):
        self.vlm_lm = vlm_lm
        self.question_concurrency = question_concurrency
        self.max_pages = max_pages

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
                    messages = _build_messages(q.question, doc_info, pages)
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
    )
