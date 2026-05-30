"""Minimal-prompt direct_vlm variant: tests the architecture's generality.

Why this exists. The current ``direct_vlm`` solver carries a
DocVQA-2026-specific ``TOOL_HINTS`` dict in the solver body (7
hand-crafted per-category overlays: engineering_drawing, business_report,
comics, maps, science_paper, science_poster, infographics, slide), and
a few body bullets tuned to DocVQA-2026 answer conventions (e.g. the
"Unknown" sentinel rule, which is already encoded in the profile's
answer-formatting rules). A reviewer can legitimately argue *"the
direct_vlm result reflects benchmark prompt-tuning, not the REPL +
multimodal architecture."*

``direct_vlm_minimal`` strips the benchmark-tuned content from the
solver body. What remains is:

1. **Sandbox + tool docs** (brief): how ``pages`` is preloaded, what
   ``display(image)`` does, when to crop; the ``SUBMIT(answer=...)``
   terminal action.
2. **Approach** (universal): SURVEY → LOCATE → EXTRACT → VERIFY → SUBMIT.
3. **Generic agent guidance**: conflict resolution between conflicting
   reads, superlative enumeration, the faithfulness rule.

Dropped vs ``direct_vlm``:

- The entire ``TOOL_HINTS`` dict and the ``_get_category_tips``
  dispatch (per-category ``display()`` routing tips).
- The DocVQA-2026 "Unknown" sentinel bullet in the body — this is
  already in ``profile.answer_formatting_rules`` and appended at
  runtime, so the duplicate body bullet is benchmark-tuned residue.
- The "SUPERLATIVES" / "COMPUTATION" / "tables" / "spatial" specific
  framings are kept (they are universal Python-REPL agent patterns,
  not DocVQA-2026 tuned) but compressed.

Dataset-specific content (answer format rules, percentage-difference
convention, judge-specific Unknown handling) stays in
``profile.answer_formatting_rules`` and is appended unchanged. The
solver body itself is byte-identical across DocVQA-2026, MP-DocVQA,
and MMLongBench-Doc — only the profile changes.

This is the *direct-VLM* counterpart to ``rvlm_minimal``: same
generality test, different perception channel (the LLM's own
multimodal input rather than a recursive VLM sub-call). Together
they give the paper a clean 2x2: {minimal, full} x {direct, rvlm}.

Engineering name only per D-010 — paper-facing name TBD.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import dspy
import logfire
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from docvqa.data import Document, Question
from docvqa.datasets.profile import DatasetProfile, get_profile
from docvqa.rlm import MultimodalRLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal task body: sandbox + tool docs + approach + universal guidance.
# Zero benchmark-category names. Zero per-category dispatch. Dataset
# specific answer rules are appended by ``profile.answer_formatting_rules``
# at runtime.
# ---------------------------------------------------------------------------

_TASK_BODY = (
    "You are a Document Visual Question Answering agent. You answer a question about a document by "
    "displaying page images, examining them visually, and reasoning step by step in Python.\n\n"

    "## PRE-LOADED SANDBOX\n"
    "The REPL already has these variables defined — use them directly. "
    "DO NOT import PIL or open files from disk; the images are NOT on your CWD.\n"
    "- `pages`: list of page images as PIL Images (0-indexed), already loaded in memory.\n"
    "  Access a page: `pages[0]`, `pages[1]`, ... Dimensions: `pages[i].size` → (width, height).\n"
    "  Crop a region: `pages[i].crop((left, top, right, bottom))`.\n\n"

    "## DATA\n"
    "- `question`: The question you must answer.\n\n"

    "## TOOLS\n"
    "- `display(image)` — Show a PIL Image inline. You will SEE the image in the next step. "
    "`image` can be a full page (e.g. `pages[0]`), a crop (e.g. `pages[0].crop((l,t,r,b))`), "
    "or any processed PIL Image. Full pages are downscaled — for fine details, crop first.\n"
    "- `print()` — ALWAYS print to see text results (numbers, strings, computed values).\n"
    "- `SUBMIT(answer=\"...\")` — deliver the final answer and terminate.\n\n"

    "## APPROACH\n"
    "1. SURVEY — `display()` page(s) at full size to build a mental map: what sections, "
    "tables, figures, and labels are present, and roughly where.\n"
    "2. LOCATE — identify the page(s) and region(s) that contain the answer.\n"
    "3. EXTRACT — `display()` tight crops with `pages[i].crop((l,t,r,b))` to read exact "
    "values. Full pages are downscaled; for fine detail, CROP FIRST.\n"
    "4. VERIFY — for any precise value (numbers, fine text, small labels), do not commit "
    "a reading you've only seen once. If two reads disagree, crop tighter on the specific "
    "detail and do one tie-breaking read; trust the higher-resolution crop.\n"
    "5. SUBMIT — call `SUBMIT(answer=\"...\")` once you have the answer.\n\n"

    "Never use outside or world knowledge. Every answer must come from the document.\n\n"

    "## GUIDELINES\n"
    "- Do not re-display the same full page hoping to see more detail — crop instead.\n"
    "- After displaying, describe what you see in your reasoning — this helps you think clearly.\n"
    "- SUPERLATIVES / 'all of' questions (\"largest\", \"first\", \"list all...\"): enumerate "
    "ALL candidates first, then select programmatically. Do NOT stop at the first match.\n"
    "- COMPUTATION: when a question requires arithmetic (\"total\", \"considering X and Y\"), "
    "extract the values visually and compute explicitly in Python.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string.\n"
    '- Example: SUBMIT(answer="42")\n'
    "- The answer must follow these formatting rules:\n\n"
)


def _build_task_instructions(profile: DatasetProfile) -> str:
    return _TASK_BODY + profile.answer_formatting_rules


SEED_TASK_INSTRUCTIONS_LENGTH = len(_TASK_BODY)  # for paper-quotable sizing


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


def _build_sandbox_code(page_dir: str, num_pages: int) -> str:
    """Build sandbox code that loads pages as PIL Images."""
    return f'''
import os
from PIL import Image

# Load all pages as PIL Images
Image.MAX_IMAGE_PIXELS = 500_000_000
pages = []
for i in range({num_pages}):
    path = os.path.join({page_dir!r}, f"page_{{i}}.png")
    assert os.path.exists(path), f"Page image not found: {{path}}"
    pages.append(Image.open(path))
assert len(pages) == {num_pages}, f"Expected {num_pages} pages, got {{len(pages)}}"
'''


# ---------------------------------------------------------------------------
# DirectVlmMinimalProgram
# ---------------------------------------------------------------------------

class DirectVlmMinimalProgram:
    """Minimal-prompt direct-VLM solver. See module docstring."""

    def __init__(
        self,
        profile: DatasetProfile,
        max_iterations: int = 20,
        max_messages: int = 8,
        max_image_pixels: int = 1_000_000,
        question_concurrency: int = 4,
    ):
        self.profile = profile
        self.max_iterations = max_iterations
        self.max_messages = max_messages
        self.max_image_pixels = max_image_pixels
        self.question_concurrency = question_concurrency

    def _per_question_prefix(self, q: Question) -> str:
        if self.profile.question_format_hint_fn is None:
            return ""
        hint = self.profile.question_format_hint_fn(q)
        return f"\n{hint}\n" if hint else ""

    def solve_document(self, document: Document) -> tuple[dict[str, str], dict[str, list[dict]]]:
        """Solve all questions for a document, one question per MultimodalRLM session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, img in enumerate(document.images):
                img.save(os.path.join(tmpdir, f"page_{i}.png"), format="PNG")

            doc_info = f"Category: {document.doc_category}, Pages: {len(document.images)}"
            sandbox_code = _build_sandbox_code(tmpdir, len(document.images))

            # No category tips, no per-document dispatch — the body is the body.
            instructions = _build_task_instructions(self.profile)

            def _solve_question(q: Question):
                """Solve a single question. Returns (question_id, answer, trajectory)."""
                with logfire.span(
                    "solve_direct_vlm",
                    doc_id=document.doc_id,
                    question_id=q.question_id,
                    question=q.question[:200],
                    profile=self.profile.name,
                ) as q_span:
                    question_text = q.question + self._per_question_prefix(q)
                    rvlm = MultimodalRLM(
                        signature=_build_signature(instructions),
                        max_iterations=self.max_iterations,
                        max_llm_calls=self.max_iterations * 3,
                        tools=[],
                        verbose=True,
                        sandbox_code=sandbox_code,
                        max_messages=self.max_messages,
                        max_image_pixels=self.max_image_pixels,
                    )
                    logger.info(
                        "DIRECT-MIN [%s] Q %s: max_iterations=%d",
                        self.profile.name, q.question_id, self.max_iterations,
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
                        return rvlm(question=question_text, doc_info=doc_info)

                    try:
                        result = _solve_one()
                        answer = str(result.answer or "").strip()
                        trajectory = result.trajectory
                    except Exception as e:
                        logger.warning("DIRECT-MIN failed for Q '%s': %s", q.question_id, e)
                        answer = "Unknown"
                        trajectory = []

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
                            "DIRECT-MIN[%s] Q %s: %s (GT=%s, PRED=%s)",
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
                logger.info("DIRECT-MIN: running %d questions with concurrency=%d", len(document.questions), max_w)
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
                    "DIRECT-MIN [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name, document.doc_id, correct, scored,
                    100 * correct / scored,
                )

            return predictions, trajectories


# ---------------------------------------------------------------------------
# Factory for hydra instantiation
# ---------------------------------------------------------------------------

def create_direct_vlm_minimal_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 20,
    max_messages: int = 8,
    max_image_pixels: int = 1_000_000,
    question_concurrency: int = 4,
    vlm: dict[str, Any] | None = None,  # unused — direct VLM doesn't need a separate VLM
) -> DirectVlmMinimalProgram:
    """Hydra factory. Profile resolution mirrors ``direct_vlm_solver.create_direct_vlm_program``.

    Note: ``use_category_tips`` is intentionally absent — this variant
    has no per-category dispatch by design.
    """
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

    return DirectVlmMinimalProgram(
        profile=profile,
        max_iterations=max_iterations,
        max_messages=max_messages,
        max_image_pixels=max_image_pixels,
        question_concurrency=question_concurrency,
    )
