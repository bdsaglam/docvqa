"""Direct-VLM solver — single multimodal model perceives images via `display()`.

The agent calls ``display(image)`` to show a PIL Image inline, and sees it in
the next iteration as a native image in the LLM message. This only works
with multimodal LLMs (e.g., Gemini Pro). No VLM tool calls
(``look``/``batch_look``) — perception is *direct* via the LLM's own
multimodal channel, not delegated to a recursive sub-call. That contrast
with :mod:`docvqa.solvers.rvlm_solver` is what the name encodes per D-010
(``rvlm`` vs ``direct_vlm`` along the recursive-vs-direct axis).

Engineering name only per D-010 — paper-facing name TBD.

Per D-009 (docs/paper/decisions.md, 2026-05-27), tool-agnostic semantic
content comes from the dataset profile
(``profile.category_tips_fn(category)``). Tool-routing for ``display()``
lives in this solver — in :data:`TOOL_HINTS` below — and is layered on top
of the profile tips at the call site. ``ANSWER_FORMATTING_RULES`` comes
from the profile, not from :mod:`docvqa.prompts`.
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
    "- UNKNOWN RULES: Answer 'Unknown' when:\n"
    "  (a) A specific named entity does not exist after thorough visual search.\n"
    "  (b) A chart/table explicitly shows N/A or missing data for the requested item.\n"
    "  Do NOT substitute a similar-sounding entity or extrapolate from nearby data.\n"
    "  Do NOT use narrative/descriptive text when a chart explicitly shows N/A.\n"
    "- COMPUTATION: When a question says 'total' or 'considering X and Y', it may require arithmetic. "
    "Extract all referenced values visually and compute explicitly in Python.\n"
    "- For tables: crop overlapping horizontal strips and read them one strip per step, noting the "
    "rows you read before moving to the next strip.\n"
    "- For spatial questions: display relevant regions and describe positions in your reasoning.\n"
    "- NEVER use outside/world knowledge. ALL answers MUST come from the document.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string.\n"
    '- Example: SUBMIT(answer="42")\n'
    "- The answer must follow these formatting rules:\n\n"
)


def _build_task_instructions(profile: DatasetProfile) -> str:
    return _TASK_BODY + profile.answer_formatting_rules


# ---------------------------------------------------------------------------
# Per-category tool-routing overlay (solver-owned per D-009).
#
# Semantic content comes from the dataset profile. This overlay adds
# ``display()``-specific tool-routing examples on top of the semantic tips,
# mirroring the FLAT_SOLO_TOOL_HINTS pattern from rvlm_full / rvlm_ocr
# (adapted to direct-vlm's tool surface).
# ---------------------------------------------------------------------------

TOOL_HINTS: dict[str, str] = {
    "engineering_drawing": (
        "- TOOL ROUTING: BOM tables are dense — display in overlapping horizontal strips at full "
        "resolution. Read each strip carefully, stitch rows in code.\n"
        "- For counting parts, display the full BOM, read ALL rows with their QTY column, "
        "then sum in Python — don't estimate visually.\n"
        "- Leader lines: display the label AND the part it connects to separately to confirm.\n"
    ),
    "business_report": (
        "- TOOL ROUTING: Display the table title/headers first to confirm you're reading the "
        "correct one before extracting values.\n"
        "- For YoY calculations, display the specific table cells, extract raw numbers, compute in Python.\n"
        "- Pictograms: display each icon individually at high zoom and describe it, rather than "
        "scanning all icons at once.\n"
    ),
    "comics": (
        "- TOOL ROUTING: Display pages and read speech bubbles visually.\n"
        "- For counting events: display each page, ask yourself 'what happens in each panel?' "
        "Build a list of events with strict inclusion criteria, then count in code.\n"
        "- After collecting event candidates, crop the specific panel tightly and re-display to "
        "ask a disconfirming question.\n"
    ),
    "maps": (
        "- TOOL ROUTING: COARSE-TO-FINE — display the full page first for rough layout, then "
        "crop ~800px regions, then ~400px for small text.\n"
        "- For tile-based counting, use `pages[i].crop((l,t,r,b))` for each tile and "
        "`display(...)` it; list every visible object in your reasoning.\n"
        "- LEGEND + ROAD TYPES: crop the legend early; crop the specific road segment at HIGH "
        "resolution alongside the legend to compare line styles directly.\n"
    ),
    "science_paper": (
        "- TOOL ROUTING: Display pages to locate relevant sections, then crop for details.\n"
        "- CITATION NUMBERS: display the relevant paragraph at full resolution and read [N] "
        "patterns directly.\n"
        "- CITED PAPER FINDINGS: display the bibliography page(s) to locate the reference number, "
        "then display body-text occurrences.\n"
    ),
    "science_poster": (
        "- TOOL ROUTING: Crop specific sections for precise values rather than re-displaying full page.\n"
        "- For table values, always crop the specific cell at full resolution before displaying.\n"
        "- COLOR-CODED VALUES: crop the table at MAXIMUM resolution and describe colors of "
        "individual cells in your reasoning.\n"
    ),
    "infographics": (
        "- TOOL ROUTING: For precise numbers or dates, crop the specific data point. "
        "For layout/overview, full-page display is fine.\n"
    ),
    "slide": (
        "- TOOL ROUTING: Display pages to find the right slide, then crop for details.\n"
        "- Verify page indices by displaying the page header/title — off-by-one errors are common.\n"
        "- Tables in slides are small — crop at full resolution before reading values.\n"
    ),
}


def _get_category_tips(profile: DatasetProfile, category: str | None) -> str:
    """Compose profile semantic tips + this solver's display() tool-routing overlay."""
    if not category:
        return ""
    base = profile.category_tips_fn(category)
    tool = TOOL_HINTS.get(category, "")
    if not base and not tool:
        return ""
    if tool and base:
        # profile.category_tips_fn already wraps with the
        # ``## CATEGORY-SPECIFIC TIPS`` header; append the tool overlay.
        return base + tool
    if tool:
        return f"## CATEGORY-SPECIFIC TIPS ({category})\n{tool}"
    return base


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
        max_image_pixels: int = 1_000_000,
        use_category_tips: bool = True,
        question_concurrency: int = 4,
    ):
        self.profile = profile
        self.max_iterations = max_iterations
        self.max_messages = max_messages
        self.max_image_pixels = max_image_pixels
        self.use_category_tips = use_category_tips
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

            instructions = _build_task_instructions(self.profile)
            if self.use_category_tips:
                tips = _get_category_tips(self.profile, document.doc_category)
                if tips:
                    instructions = instructions + "\n" + tips

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
                        "Direct-VLM [%s] Q %s: max_iterations=%d",
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
    max_image_pixels: int = 1_000_000,
    use_category_tips: bool = True,
    question_concurrency: int = 4,
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
        max_image_pixels=max_image_pixels,
        use_category_tips=use_category_tips,
        question_concurrency=question_concurrency,
    )
