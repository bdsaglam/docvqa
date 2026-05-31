"""Raw-VLM single-image baseline — direct VLM Q&A on a vertical-stacked composite.

Single forward pass: question + a composite image (all pages stacked
vertically into one image) → answer. No REPL, no tools, no agent loop.

This is the "raw model" / single-image point used alongside
``raw_vlm_multi`` (which takes native multi-image input) as a matched
baseline for the scaffold-vs-raw lift. Pages are concatenated vertically
into a single composite so the call stays a single VLM forward pass
regardless of page count — defensible single-call baseline for providers
that don't expose native multi-image input.

Dataset-aware via injected :class:`docvqa.datasets.profile.DatasetProfile`;
DocVQA-2026 default. Engineering name per D-010 (formerly ``no_loop``).

Minimal prompt (matches ``rvlm_minimal``): the task instructions are the
generic body + ``profile.answer_formatting_rules``, with NO hand-crafted
per-category tips. This keeps the raw-VLM baseline at the same prompt
sophistication as the rvlm method, so the method-vs-baseline gap isn't
confounded by prompt tuning. Dataset-awareness stays: the profile still
supplies ``answer_formatting_rules``, the per-question hint, and the
``score_fn``.
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
from docvqa.retry_utils import is_retryable_lm_error

logger = logging.getLogger(__name__)

_TASK_BODY = (
    "You are answering a question about a document. The document is shown as a single composite "
    "image with one or more pages stacked vertically. Look at the image and the question, "
    "then output a single concise answer.\n\n"
    "## OUTPUT FORMAT\n"
    "- Output ONLY the final answer string — no explanation, no preamble.\n"
    "- The answer must follow these formatting rules:\n\n"
)

def _build_task_instructions(profile: DatasetProfile) -> str:
    return _TASK_BODY + profile.answer_formatting_rules

def _build_signature(instructions: str) -> dspy.Signature:
    fields: dict = {
        "question": (str, dspy.InputField(desc="The question to answer about the document")),
        "doc_info": (str, dspy.InputField(desc="Document metadata: category and page count")),
        "image": (
            dspy.Image,
            dspy.InputField(desc="Composite image of the document pages stacked vertically"),
        ),
        "answer": (str, dspy.OutputField(desc="The final concise answer string.")),
    }
    return dspy.Signature(fields, instructions)

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

class RawVlmSingleProgram:
    """Direct VLM Q&A — one call per question, no agent loop, single composite image."""

    def __init__(
        self,
        vlm_lm: dspy.LM,
        profile: DatasetProfile,
        question_concurrency: int = 4,
        max_height: int = 16384,
    ):
        self.vlm_lm = vlm_lm
        self.profile = profile
        self.question_concurrency = question_concurrency
        self.max_height = max_height
        # Minimal prompt: generic body + profile.answer_formatting_rules.
        # No category tips, no per-document dispatch — the body is the body.
        self._predict = dspy.Predict(
            _build_signature(_build_task_instructions(self.profile))
        )

    def _per_question_text(self, q: Question) -> str:
        text = q.question
        if self.profile.question_format_hint_fn is not None:
            hint = self.profile.question_format_hint_fn(q)
            if hint:
                text = f"{text}\n{hint}"
        return text

    def solve_document(self, document: Document) -> tuple[dict[str, str], dict[str, list[dict]]]:
        composite = _stack_pages(document.images, max_height=self.max_height)
        composite_dspy = dspy.Image(composite)
        doc_info = f"Category: {document.doc_category}, Pages: {len(document.images)}"
        predict = self._predict

        def _solve_question(q: Question):
            with logfire.span(
                "solve_raw_vlm_single",
                doc_id=document.doc_id,
                question_id=q.question_id,
                question=q.question[:200],
                profile=self.profile.name,
            ) as q_span:

                @retry(
                    retry=retry_if_exception(is_retryable_lm_error),
                    stop=stop_after_attempt(4),
                    wait=wait_exponential(multiplier=30, min=30, max=120),
                    before_sleep=lambda rs: logger.warning(
                        "Rate limit, retry %d in %.0fs", rs.attempt_number, rs.next_action.sleep  # type: ignore[union-attr]
                    ),
                    reraise=True,
                )
                def _call():
                    with dspy.context(lm=self.vlm_lm):
                        return predict(
                            question=self._per_question_text(q),
                            doc_info=doc_info,
                            image=composite_dspy,
                        )

                try:
                    result = _call()
                    answer = str(result.answer or "").strip()
                except Exception as e:
                    logger.warning("Raw-VLM single failed for Q '%s': %s", q.question_id, e)
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
                        "RawVlmSingle[%s] Q %s: %s (GT=%s, PRED=%s)",
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
                "RawVlmSingle[%s] doc %s: %d/%d = %.1f%%",
                self.profile.name,
                document.doc_id, correct, scored, 100 * correct / scored,
            )

        return predictions, trajectories

def create_raw_vlm_single_baseline_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    vlm: dict[str, Any] | None = None,
    question_concurrency: int = 4,
    max_height: int = 16384,
) -> RawVlmSingleProgram:
    """Hydra factory. See ``rvlm_full_solver.create_rvlm_full_program`` for profile resolution."""
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
    return RawVlmSingleProgram(
        vlm_lm=vlm_lm,
        profile=profile,
        question_concurrency=question_concurrency,
        max_height=max_height,
    )
