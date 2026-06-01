"""Direct-VLM solver — single multimodal model perceives images via `display()`.

The agent calls ``display(image)`` to show a PIL Image inline, and sees it in
the next iteration as a native image in the LLM message. This only works
with multimodal LLMs (e.g., Gemini Pro). No VLM tool calls
(``look``/``batch_look``) — perception is *direct* via the LLM's own
multimodal channel, not delegated to a recursive sub-call. That contrast
with :mod:`docvqa.solvers.rvlm_solver` is what the name encodes per D-010
(``rvlm`` vs ``direct_vlm`` along the recursive-vs-direct axis).

Engineering name only per D-010 — paper-facing name TBD.

The solver prompt carries no benchmark-specific per-category content (the
per-category ``display()`` tool-routing overlay was removed in the D-010
solver cleanup, mirroring the minimal ``rvlm`` solver). Dataset-specific
answer-formatting rules still come from the profile
(``profile.answer_formatting_rules``), not from :mod:`docvqa.prompts`.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import dspy
import logfire

from docvqa.data import Document, Question
from docvqa.datasets.profile import DatasetProfile, get_profile
from docvqa.rlm import MultimodalRLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt body (formatting rules substituted from the profile)
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
    "- `question`: The question you must answer.\n"
    "- `doc_info`: Document category and page count.\n\n"

    "## TOOLS\n"
    "- `display(image)` — Show a PIL Image inline. You will SEE the image in the next step. "
    "`image` can be a full page (e.g. `pages[0]`), a crop (e.g. `pages[0].crop((l,t,r,b))`), "
    "or any processed PIL Image. Full pages are downscaled — for fine details, crop first.\n"
    "- `print()` — ALWAYS print to see text results (numbers, strings, computed values).\n"
    "- `RESET_HISTORY(summary=\"...\")` — Compact your history: clears all past steps and their "
    "images from view, keeping only your summary text. Variables (incl. `pages`) persist, so you "
    "can re-`display()` later if needed. Use it after you've extracted what you need from images.\n\n"

    "## APPROACH\n"
    "1. EXPLORE: Start with `display(pages[0])` (and further pages if multi-page) to see the layout. "
    "Build a mental map: what sections, tables, figures, and labels are present and where.\n"
    "2. LOCATE: Find the specific region(s) relevant to the question.\n"
    "3. EXTRACT: `display()` tight crops with `pages[i].crop((l,t,r,b))` to read exact values.\n"
    "4. VERIFY: Cross-check extracted values if ambiguous.\n"
    "5. SUBMIT: Once you have the answer, call `SUBMIT(answer=\"...\")`.\n\n"

    "## GUIDELINES\n"
    "- LOOK, THEN NOTE: after each `display()`, WRITE DOWN what you see (key text, values, "
    "positions) in your reasoning and with `print()`. Display a few related crops per step as "
    "needed — once you've noted what an image shows, it has done its job.\n"
    "- CONTEXT IS A SLIDING WINDOW: only your last several steps stay in view; older steps (and "
    "their images) drop off. Your written notes persist, so record what matters as you go and rely "
    "on the notes rather than expecting old images to still be visible.\n"
    "- COMPACT OFTEN: call `RESET_HISTORY(summary='<all findings so far>')` FREQUENTLY — e.g. after "
    "finishing a page/region or every several displays — to clear accumulated images and keep your "
    "context small and focused. Variables (incl. `pages`) persist, so you can re-`display()` later "
    "if you need another look. Compacting regularly keeps you fast and avoids overloading on images.\n"
    "- Full-page `display()` gives an overview; for fine details CROP FIRST using pixel coordinates "
    "from `pages[i].size`. Do not re-display the same full page hoping to see more detail — crop instead.\n"
    "- After displaying, describe what you see in your reasoning — this helps you think clearly.\n"
    "- CONFLICT RESOLUTION: If you read conflicting values across displays, crop TIGHTER on the "
    "specific detail and do one tie-breaking read. Trust the higher-resolution crop.\n"
    "- SUPERLATIVES: For 'largest', 'first', 'last', 'only' — enumerate ALL candidates first, "
    "then select programmatically. Do NOT stop at the first match.\n"
    "- NEVER use outside/world knowledge. ALL answers MUST come from the document.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string.\n"
    '- Example: SUBMIT(answer="42")\n'
    "- The answer must follow these formatting rules:\n\n"
)

# Legacy pre-compaction prompt (no RESET_HISTORY / sliding-window / compact-often
# language). Byte-exact copy of the prompt that produced il_n=3 = 43.2%. Used to
# isolate the prompt as the regression lever when the window is effectively
# unbounded (max_messages huge). Select via legacy_prompt=True.
_TASK_BODY_LEGACY = (
    "You are a Document Visual Question Answering agent. You answer a question about a document by "
    "displaying page images, examining them visually, and reasoning step by step in Python.\n\n"

    "## PRE-LOADED SANDBOX\n"
    "The REPL already has these variables defined — use them directly. "
    "DO NOT import PIL or open files from disk; the images are NOT on your CWD.\n"
    "- `pages`: list of page images as PIL Images (0-indexed), already loaded in memory.\n"
    "  Access a page: `pages[0]`, `pages[1]`, ... Dimensions: `pages[i].size` → (width, height).\n"
    "  Crop a region: `pages[i].crop((left, top, right, bottom))`.\n\n"

    "## DATA\n"
    "- `question`: The question you must answer.\n"
    "- `doc_info`: Document category and page count.\n\n"

    "## TOOLS\n"
    "- `display(image)` — Show a PIL Image inline. You will SEE the image in the next step. "
    "`image` can be a full page (e.g. `pages[0]`), a crop (e.g. `pages[0].crop((l,t,r,b))`), "
    "or any processed PIL Image. Full pages are downscaled — for fine details, crop first.\n"
    "- `print()` — ALWAYS print to see text results (numbers, strings, computed values).\n\n"

    "## APPROACH\n"
    "1. EXPLORE: Start with `display(pages[0])` (and further pages if multi-page) to see the layout. "
    "Build a mental map: what sections, tables, figures, and labels are present and where.\n"
    "2. LOCATE: Find the specific region(s) relevant to the question.\n"
    "3. EXTRACT: `display()` tight crops with `pages[i].crop((l,t,r,b))` to read exact values.\n"
    "4. VERIFY: Cross-check extracted values if ambiguous.\n"
    "5. SUBMIT: Once you have the answer, call `SUBMIT(answer=\"...\")`.\n\n"

    "## GUIDELINES\n"
    "- Full-page `display()` gives an overview; for fine details CROP FIRST using pixel coordinates "
    "from `pages[i].size`. Do not re-display the same full page hoping to see more detail — crop instead.\n"
    "- After displaying, describe what you see in your reasoning — this helps you think clearly.\n"
    "- CONFLICT RESOLUTION: If you read conflicting values across displays, crop TIGHTER on the "
    "specific detail and do one tie-breaking read. Trust the higher-resolution crop.\n"
    "- SUPERLATIVES: For 'largest', 'first', 'last', 'only' — enumerate ALL candidates first, "
    "then select programmatically. Do NOT stop at the first match.\n"
    "- NEVER use outside/world knowledge. ALL answers MUST come from the document.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string.\n"
    '- Example: SUBMIT(answer="42")\n'
    "- The answer must follow these formatting rules:\n\n"
)

def _build_task_instructions(profile: DatasetProfile, legacy: bool = False) -> str:
    body = _TASK_BODY_LEGACY if legacy else _TASK_BODY
    return body + profile.answer_formatting_rules

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
# DirectVlmProgram
# ---------------------------------------------------------------------------

class DirectVlmProgram:
    """Direct-VLM solver — single multimodal model with inline image display, per-question."""

    def __init__(
        self,
        profile: DatasetProfile,
        max_iterations: int = 20,
        max_messages: int = 8,
        images_for_last_n: int = 10_000,
        max_image_pixels: int = 1_000_000,
        question_concurrency: int = 4,
        legacy_prompt: bool = False,
    ):
        self.profile = profile
        self.max_iterations = max_iterations
        self.max_messages = max_messages
        self.images_for_last_n = images_for_last_n
        self.max_image_pixels = max_image_pixels
        self.question_concurrency = question_concurrency
        self.legacy_prompt = legacy_prompt

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

            instructions = _build_task_instructions(self.profile, legacy=self.legacy_prompt)

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
                        images_for_last_n=self.images_for_last_n,
                        max_image_pixels=self.max_image_pixels,
                    )
                    logger.info(
                        "Direct-VLM [%s] Q %s: max_iterations=%d",
                        self.profile.name, q.question_id, self.max_iterations,
                    )

                    def _solve_one():
                        return rvlm(question=question_text, doc_info=doc_info)

                    try:
                        result = _solve_one()
                        answer = str(result.answer or "").strip()
                        trajectory = result.trajectory
                    except Exception as e:
                        logger.warning("Direct-VLM failed for Q '%s': %s", q.question_id, e)
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
                            "Direct-VLM[%s] Q %s: %s (GT=%s, PRED=%s)",
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
                logger.info("Direct-VLM: running %d questions with concurrency=%d", len(document.questions), max_w)
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
                    "Direct-VLM [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name, document.doc_id, correct, scored,
                    100 * correct / scored,
                )

            return predictions, trajectories

# ---------------------------------------------------------------------------
# Factory for hydra instantiation
# ---------------------------------------------------------------------------

def create_direct_vlm_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 20,
    max_messages: int = 8,
    images_for_last_n: int = 10_000,
    max_image_pixels: int = 1_000_000,
    question_concurrency: int = 4,
    legacy_prompt: bool = False,
    vlm: dict[str, Any] | None = None,  # unused — direct VLM doesn't need a separate VLM
) -> DirectVlmProgram:
    """Hydra factory. Profile resolution mirrors ``rvlm_solver.create_rvlm_program``."""
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

    return DirectVlmProgram(
        profile=profile,
        max_iterations=max_iterations,
        max_messages=max_messages,
        images_for_last_n=images_for_last_n,
        max_image_pixels=max_image_pixels,
        question_concurrency=question_concurrency,
        legacy_prompt=legacy_prompt,
    )
