"""Dataset-aware no_loop_multi baseline.

A clone of :mod:`docvqa.solvers.no_loop_multi_solver` with a
:class:`docvqa.datasets.profile.DatasetProfile` injected via the
constructor, so the baseline gets the same dataset-tuned answer
formatting, category tips, per-question format hint, and judge
scoring as ``flat_solo_da``. This keeps the
baseline-vs-scaffold delta cleanly attributable to the scaffold
itself rather than to a prompt mismatch.
"""

from __future__ import annotations

import logging
from typing import Any

import dspy
import logfire
from PIL import Image as PILImage
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from docvqa.data import Document, Question
from docvqa.datasets.profile import DatasetProfile, get_profile
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)


_TASK_BODY = (
    "You are answering a question about a document. The document is shown as a sequence "
    "of page images, in order (page 0 first, page 1 next, etc.). Look at the images and "
    "the question, then output a single concise answer.\n\n"
    "## OUTPUT FORMAT\n"
    "- Output ONLY the final answer string — no explanation, no preamble.\n"
    "- The answer must follow these formatting rules:\n\n"
)


def _build_task_instructions(profile: DatasetProfile) -> str:
    return _TASK_BODY + profile.answer_formatting_rules


def _build_messages(
    question: str,
    doc_info: str,
    pages: list[PILImage.Image],
    instructions: str,
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


class NoLoopMultiDAProgram:
    """Dataset-aware direct VLM Q&A baseline."""

    def __init__(
        self,
        vlm_lm: dspy.LM,
        profile: DatasetProfile,
        question_concurrency: int = 4,
        max_pages: int = 10,
    ):
        self.vlm_lm = vlm_lm
        self.profile = profile
        self.question_concurrency = question_concurrency
        self.max_pages = max_pages

    def _per_question_text(self, q: Question) -> str:
        text = q.question
        if self.profile.question_format_hint_fn is not None:
            hint = self.profile.question_format_hint_fn(q)
            if hint:
                text = f"{text}\n{hint}"
        return text

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
        base_instructions = _build_task_instructions(self.profile)
        tips = self.profile.baseline_category_tips_fn(document.doc_category)
        instructions = base_instructions + ("\n" + tips if tips else "")

        def _solve_question(q: Question):
            with logfire.span(
                "solve_no_loop_multi_da",
                doc_id=document.doc_id,
                question_id=q.question_id,
                question=q.question[:200],
                profile=self.profile.name,
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
                    question_text = self._per_question_text(q)
                    messages = _build_messages(question_text, doc_info, pages, instructions)
                    response: Any = self.vlm_lm.forward(messages=messages)
                    msg = response.choices[0].message
                    text = msg.content
                    if not text:
                        text = getattr(msg, "reasoning_content", "") or ""
                    return str(text or "").strip()

                try:
                    answer = _call() or "Unknown"
                except Exception as e:
                    logger.warning("NoLoopMultiDA failed for Q '%s': %s", q.question_id, e)
                    answer = "Unknown"

                if not answer:
                    answer = "Unknown"

                q_span.set_attribute("prediction", answer[:200])

                if q.answer is not None:
                    is_correct, extracted = self.profile.score_fn(answer, q.answer, q)
                    q_span.set_attribute("is_correct", is_correct)
                    q_span.set_attribute("ground_truth", q.answer[:200])
                    q_span.set_attribute("extracted_answer", extracted[:200])
                    logger.info(
                        "NoLoopMultiDA[%s] Q %s: %s (GT=%s, PRED=%s)",
                        self.profile.name,
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
                if self.profile.score_fn(predictions[q.question_id], q.answer, q)[0]:
                    correct += 1
        if scored > 0:
            logger.info(
                "NoLoopMultiDA[%s] doc %s: %d/%d = %.1f%%",
                self.profile.name,
                document.doc_id,
                correct,
                scored,
                100 * correct / scored,
            )

        return predictions, trajectories


def create_no_loop_multi_da_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    vlm: dict[str, Any] | None = None,
    question_concurrency: int = 4,
    max_pages: int = 10,
) -> NoLoopMultiDAProgram:
    """Hydra factory. See ``flat_solo_da_solver.create_flat_solo_da_program``."""
    from docvqa.datasets.profile import _PROFILES  # type: ignore[attr-defined]

    if profile_name is not None:
        for p in _PROFILES.values():
            if p.name == profile_name:
                profile = p
                break
        else:
            profile = get_profile(profile_name)
    elif dataset is not None:
        profile = get_profile(dataset)
    else:
        profile = get_profile("VLR-CVC/DocVQA-2026")

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
    return NoLoopMultiDAProgram(
        vlm_lm=vlm_lm,
        profile=profile,
        question_concurrency=question_concurrency,
        max_pages=max_pages,
    )
