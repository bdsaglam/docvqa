"""REPL-only ablation per D-006 prediction 3.

Fork of :mod:`docvqa.solvers.rvlm_solver` with the VLM sub-call
(``batch_look``) removed. The agent retains the REPL + Python execution +
``SUBMIT`` affordance but has NO image perception (and no OCR text channel
— by design, this is a *pure* no-perception ablation, not an "OCR-only"
one).

Predicted behavior: the agent collapses to roughly the no-loop baseline
(~17-25% val on DocVQA-2026), because it cannot see the document. This
tests whether the recursive VLM sub-call is the load-bearing mechanism
of the scaffold.

Dataset-aware via injected :class:`docvqa.datasets.profile.DatasetProfile`
per D-009 (DocVQA-2026 default). This is a **minimal-prompt** baseline:
the solver body carries only generic guidance and no hand-crafted
per-category tips, matching the minimalism of ``rvlm_minimal`` so the
baseline-vs-method comparison is fair (no benchmark-specific prompt
tuning on either side). The only thing this solver changes vs
``rvlm_minimal`` is the tool surface (no perception).

``ANSWER_FORMATTING_RULES`` comes from ``profile.answer_formatting_rules``,
not from :mod:`docvqa.prompts`.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import dspy
import logfire
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from docvqa.data import Document, Question
from docvqa.retry_utils import is_retryable_lm_error
from docvqa.datasets.profile import DatasetProfile, get_profile
from docvqa.rlm import LeanRLM, CodeRLM, ThinkingRLM, RLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt body (formatting rules substituted from the profile)
# ---------------------------------------------------------------------------

_TASK_BODY = (
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
)

def _build_task_instructions(profile: DatasetProfile) -> str:
    return _TASK_BODY + profile.answer_formatting_rules

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_signature(instructions: str) -> dspy.Signature:
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
        profile: DatasetProfile,
        max_iterations: int = 20,
        rlm_type: str = "lean",
        page_factor: float = 1.5,
        question_concurrency: int = 1,
    ):
        self.profile = profile
        self.max_iterations = max_iterations
        self.rlm_type = rlm_type
        self.page_factor = page_factor
        self.question_concurrency = question_concurrency

    def _per_question_prefix(self, q: Question) -> str:
        if self.profile.question_format_hint_fn is None:
            return ""
        hint = self.profile.question_format_hint_fn(q)
        return f"\n{hint}\n" if hint else ""

    def solve_document(self, document: Document) -> tuple[dict[str, str], dict[str, list[dict]]]:
        """Solve all questions for a document, one question at a time, with no perception."""
        doc_info = f"Category: {document.doc_category}, Pages: {len(document.images)}"

        num_pages = len(document.images)
        page_bonus = min(10, self.page_factor * math.ceil(math.sqrt(max(0, num_pages - 9))))
        max_iter = self.max_iterations + int(page_bonus)

        # No category tips — minimal prompt = generic body + profile.answer_formatting_rules.
        instructions = _build_task_instructions(self.profile)
        tools = _create_tools()
        sandbox_code = _build_sandbox_code()

        def _solve_question(q: Question):
            """Solve a single question. Returns (question_id, answer, trajectory)."""
            with logfire.span(
                "solve_repl_only",
                doc_id=document.doc_id,
                question_id=q.question_id,
                question=q.question[:200],
                profile=self.profile.name,
            ) as q_span:
                question_text = q.question + self._per_question_prefix(q)
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
                    "REPL-only [%s] (%s) Q %s: max_iterations=%d (page_bonus=%d)",
                    self.profile.name, self.rlm_type, q.question_id, max_iter, int(page_bonus),
                )

                @retry(
                    retry=retry_if_exception(is_retryable_lm_error),
                    stop=stop_after_attempt(4),
                    wait=wait_exponential(multiplier=30, min=30, max=120),
                    before_sleep=lambda rs: logger.warning(
                        "Rate limit, retry %d in %.0fs", rs.attempt_number, rs.next_action.sleep  # type: ignore[union-attr]
                    ),
                    reraise=True,
                )
                def _solve_one():
                    return rlm(
                        question=question_text,
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
                    is_correct, extracted = self.profile.score_fn(answer, q.answer, q)
                    q_span.set_attribute("is_correct", is_correct)
                    q_span.set_attribute("ground_truth", q.answer[:200])
                    q_span.set_attribute("extracted_answer", extracted[:200])
                    logger.info(
                        "REPL-only[%s] Q %s: %s (GT=%s, PRED=%s)",
                        self.profile.name,
                        q.question_id,
                        "CORRECT" if is_correct else "WRONG",
                        q.answer[:40],
                        extracted[:40],
                    )

                return q.question_id, answer, trajectory

        # Run questions with configurable concurrency
        predictions: dict[str, str] = {}
        trajectories: dict[str, list[dict]] = {}

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
        correct = 0
        scored = 0
        for q in document.questions:
            if q.answer is not None:
                scored += 1
                if self.profile.score_fn(predictions[q.question_id], q.answer, q)[0]:
                    correct += 1

        if scored > 0:
            logger.info(
                "REPL-only [%s] doc %s: %d/%d = %.1f%%",
                self.profile.name, document.doc_id, correct, scored,
                100 * correct / scored,
            )

        return predictions, trajectories

# ---------------------------------------------------------------------------
# Factory for hydra instantiation
# ---------------------------------------------------------------------------

def create_repl_only_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 20,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    vlm: dict[str, Any] | None = None,  # unused — REPL-only has no perception
) -> ReplOnlyProgram:
    """Factory for the REPL-only solver. No VLM is configured or used."""
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

    return ReplOnlyProgram(
        profile=profile,
        max_iterations=max_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
    )
