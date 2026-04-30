"""RVLM solver — multimodal agent that sees images directly via display().

No VLM tool calls (look/batch_look) needed. The agent calls display(image) to
show a PIL Image inline, and sees it in the next iteration as a native image
in the LLM message. This only works with multimodal LLMs (e.g., Gemini Pro).
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import dspy
import logfire
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from docvqa.data import Document
from docvqa.metrics import evaluate_prediction
from docvqa.prompts import ANSWER_FORMATTING_RULES
from docvqa.rlm import RVLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

TASK_INSTRUCTIONS = (
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
    "- UNKNOWN RULES: Answer 'Unknown' when:\n"
    "  (a) A specific named entity does not exist after thorough visual search.\n"
    "  (b) A chart/table explicitly shows N/A or missing data for the requested item.\n"
    "  Do NOT substitute a similar-sounding entity or extrapolate from nearby data.\n"
    "  Do NOT use narrative/descriptive text when a chart explicitly shows N/A.\n"
    "- COMPUTATION: When a question says 'total' or 'considering X and Y', it may require arithmetic. "
    "Extract all referenced values visually and compute explicitly in Python.\n"
    "- For tables: crop overlapping horizontal strips and display each to read rows reliably.\n"
    "- For spatial questions: display relevant regions and describe positions in your reasoning.\n"
    "- NEVER use outside/world knowledge. ALL answers MUST come from the document.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string.\n"
    '- Example: SUBMIT(answer="42")\n'
    "- The answer must follow these formatting rules:\n\n"

    + ANSWER_FORMATTING_RULES
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Category-specific tips for RVLM (direct image display, no VLM tool calls)
# ---------------------------------------------------------------------------

RVLM_CATEGORY_TIPS: dict[str, str] = {
    "engineering_drawing": (
        "## CATEGORY TIPS (engineering_drawing)\n"
        "- BOM tables are dense — display in overlapping horizontal strips at full resolution. "
        "Read each strip carefully, stitch rows in code.\n"
        "- ITEM NUMBER vs PART/IDENTIFYING NUMBER: Questions asking for 'identifying number' or "
        "'part number' want the PART NUMBER column, not the item number.\n"
        "- Part numbers are almost always numeric (0-9 plus dashes). If you see I, O, l where "
        "digits 1, 0 would be expected, re-display at tighter crop. Common: I↔1, O↔0, l↔1.\n"
        "- For counting parts, display the full BOM, read ALL rows with their QTY column, "
        "then sum in Python — don't estimate visually.\n"
        "- Leader lines: display the label AND the part it connects to separately to confirm.\n"
        "- Dimensions: 'Width' = shorter cross-section (from a Section view), not overall length.\n"
        "- 'VIEW IN DIRECTION X' labels: answer with just the letter (e.g., 'D').\n"
    ),
    "business_report": (
        "## CATEGORY TIPS (business_report)\n"
        "- Multiple tables may look similar — display the table title/headers first to confirm "
        "you're reading the correct one before extracting values.\n"
        "- For YoY calculations, display the specific table cells, extract raw numbers, compute in Python.\n"
        "- CHART VALUES: Do NOT re-display the same chart to 'verify' — your readings vary between "
        "displays. Use the first clear reading.\n"
        "- 'Broken down into' = immediate sub-categories only, not sub-sub-categories.\n"
        "- Pictograms: display each icon individually at high zoom and describe it, rather than "
        "scanning all icons at once.\n"
        "- Qualitative descriptions (adjectives) may be in footnotes or body text, not in tables.\n"
    ),
    "comics": (
        "## CATEGORY TIPS (comics)\n"
        "- OCR is useless for speech bubbles — display pages and read them visually.\n"
        "- STORY MAP FIRST: For anthologies, scan pages to build an index: "
        "(story title, page range, key characters). Match question keywords to the correct story.\n"
        "- For counting events: display each page, ask yourself 'what happens in each panel?' "
        "Build a list of events with strict inclusion criteria, then count in code. "
        "Exclude mentions, near-misses, and aftermath.\n"
        "- VERIFY COUNTS: The VLM hallucinates actions in busy panels — it infers events from context clues "
        "even when no action is depicted. After collecting candidates, re-examine each: "
        "crop the specific panel tightly and ask a disconfirming question "
        "('Did this ACTUALLY occur, or is it a near-miss/different action/aftermath?'). "
        "Expect many initial candidates to be false positives.\n"
        "- LITERAL VS FIGURATIVE: 'in reality', 'actually', 'truly' means the answer contradicts "
        "the surface description. Read carefully for the distinction between title/alias and fact.\n"
        "- Character names: use the exact term from speech bubbles.\n"
    ),
    "maps": (
        "## CATEGORY TIPS (maps)\n"
        "- COARSE-TO-FINE: display full page first for rough layout, then crop ~800px regions, "
        "then ~400px for small text. Each step refines the previous.\n"
        "- LOCATE INDEPENDENTLY: For each landmark, crop a tile and list what's visible. "
        "Record approximate positions using tile offsets + relative position.\n"
        "- Compute spatial relationships in Python using collected coordinates — distances, "
        "directions, relative positions.\n"
        "- LEGEND + ROAD TYPES: crop the legend early. For road type questions, crop the specific "
        "road segment at HIGH resolution alongside the legend. Compare line styles directly.\n"
        "- GRID COORDINATES: crop the grid cell to see what's there. Cross-reference with "
        "other grid cells by displaying them.\n"
    ),
    "science_paper": (
        "## CATEGORY TIPS (science_paper)\n"
        "- Papers are long — display pages to locate relevant sections, then crop for details.\n"
        "- CITATION NUMBERS: display the relevant paragraph at full resolution and read [N] "
        "patterns directly.\n"
        "- Distinguish body text citations from table headers and figure captions.\n"
        "- Ablation studies: papers often have multiple — verify the section matches the specific "
        "component the question asks about.\n"
        "- If a specific entity (layer number, model variant) isn't found after thorough search, "
        "answer 'Unknown'. Don't extrapolate from similar entities.\n"
    ),
    "science_poster": (
        "## CATEGORY TIPS (science_poster)\n"
        "- Single dense page — crop specific sections for precise values rather than re-displaying full page.\n"
        "- Chart annotations: if bars/lines have percentage labels, read those directly instead of "
        "estimating from heights.\n"
        "- For table values, always crop the specific cell at full resolution.\n"
        "- 'Percentage improvement' = absolute difference in percentage points (80% - 50% = 30%).\n"
        "- Color-coded values: crop the table at MAXIMUM resolution. Describe colors of individual cells.\n"
        "- GROUPED BAR CHARTS: 'set of columns' = the group of bars at one x-axis position, "
        "not bars of one color across all positions.\n"
    ),
    "infographics": (
        "## CATEGORY TIPS (infographics)\n"
        "- OCR describes images rather than reading text — display cropped regions to read actual text.\n"
        "- For precise numbers or dates, crop the specific data point. For layout/overview, full-page display is fine.\n"
        "- SYSTEMATIC ENUMERATION: When a question asks 'which item is the last/first to have/lack X', "
        "enumerate ALL items and their X status in a list before answering. Don't stop after finding a few.\n"
    ),
    "slide": (
        "## CATEGORY TIPS (slide)\n"
        "- Many pages — display pages to find the right slide, then crop for details.\n"
        "- Off-by-one page errors are common — verify page indices by displaying the page header/title.\n"
        "- Tables in slides are small — crop at full resolution before reading values.\n"
        "- EXACT ENTITY MATCHING: if a column name, variable, or equation doesn't exist after "
        "thorough search, answer 'Unknown'. Don't substitute similar-sounding names.\n"
        "- For 'last word on page X', crop the bottom portion and read carefully.\n"
        "- COMPUTATION: 'total' or 'considering X and Y' means extract all values and compute in Python.\n"
    ),
}


def _get_rvlm_tips(category: str | None) -> str:
    if not category or category not in RVLM_CATEGORY_TIPS:
        return ""
    return "\n" + RVLM_CATEGORY_TIPS[category]


def _build_signature(instructions: str = TASK_INSTRUCTIONS) -> dspy.Signature:
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
# RVLMProgram
# ---------------------------------------------------------------------------

class RVLMProgram:
    """RVLM solver — multimodal agent with inline image display, per-question."""

    def __init__(
        self,
        max_iterations: int = 20,
        images_for_last_n: int = 1,
        max_image_pixels: int = 1_000_000,
        use_category_tips: bool = True,
        question_concurrency: int = 4,
    ):
        self.max_iterations = max_iterations
        self.images_for_last_n = images_for_last_n
        self.max_image_pixels = max_image_pixels
        self.use_category_tips = use_category_tips
        self.question_concurrency = question_concurrency

    def solve_document(self, document: Document) -> tuple[dict[str, str], dict[str, list[dict]]]:
        """Solve all questions for a document, one question per RVLM session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, img in enumerate(document.images):
                img.save(os.path.join(tmpdir, f"page_{i}.png"), format="PNG")

            doc_info = f"Category: {document.doc_category}, Pages: {len(document.images)}"
            sandbox_code = _build_sandbox_code(tmpdir, len(document.images))

            instructions = TASK_INSTRUCTIONS
            if self.use_category_tips:
                instructions += _get_rvlm_tips(document.doc_category)

            def _solve_question(q):
                """Solve a single question. Returns (question_id, answer, trajectory)."""
                with logfire.span(
                    "solve_rvlm_solo",
                    doc_id=document.doc_id,
                    question_id=q.question_id,
                    question=q.question[:200],
                ) as q_span:
                    rvlm = RVLM(
                        signature=_build_signature(instructions),
                        max_iterations=self.max_iterations,
                        max_llm_calls=self.max_iterations * 3,
                        tools=[],
                        verbose=True,
                        sandbox_code=sandbox_code,
                        images_for_last_n=self.images_for_last_n,
                        max_image_pixels=self.max_image_pixels,
                    )
                    logger.info("RVLM solo Q %s: max_iterations=%d", q.question_id, self.max_iterations)

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
                        return rvlm(question=q.question, doc_info=doc_info)

                    try:
                        result = _solve_one()
                        answer = str(result.answer or "").strip()
                        trajectory = result.trajectory
                    except Exception as e:
                        logger.warning("RVLM solo failed for Q '%s': %s", q.question_id, e)
                        answer = "Unknown"
                        trajectory = []

                    if not answer:
                        answer = "Unknown"

                    q_span.set_attribute("num_iterations", len(trajectory))
                    q_span.set_attribute("prediction", answer[:200])

                    if q.answer is not None:
                        is_correct, extracted = evaluate_prediction(answer, q.answer)
                        q_span.set_attribute("is_correct", is_correct)
                        q_span.set_attribute("ground_truth", q.answer[:200])
                        q_span.set_attribute("extracted_answer", extracted[:200])
                        logger.info(
                            "RVLM Q %s: %s (GT=%s, PRED=%s)",
                            q.question_id,
                            "CORRECT" if is_correct else "WRONG",
                            q.answer[:40],
                            extracted[:40],
                        )

                    return q.question_id, answer, trajectory

            # Run questions with configurable concurrency
            predictions = {}
            trajectories = {}
            correct_count = 0
            scored_count = 0

            if self.question_concurrency <= 1:
                for q in document.questions:
                    qid, answer, trajectory = _solve_question(q)
                    predictions[qid] = answer
                    trajectories[qid] = trajectory
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                max_w = min(self.question_concurrency, len(document.questions))
                logger.info("RVLM solo: running %d questions with concurrency=%d", len(document.questions), max_w)
                with ThreadPoolExecutor(max_workers=max_w) as pool:
                    futures = {pool.submit(_solve_question, q): q for q in document.questions}
                    for future in as_completed(futures):
                        qid, answer, trajectory = future.result()
                        predictions[qid] = answer
                        trajectories[qid] = trajectory

            # Score
            for q in document.questions:
                if q.answer is not None:
                    scored_count += 1
                    is_correct, _ = evaluate_prediction(predictions[q.question_id], q.answer)
                    if is_correct:
                        correct_count += 1

            if scored_count > 0:
                logger.info(
                    "RVLM solo doc %s: %d/%d = %.1f%%",
                    document.doc_id, correct_count, scored_count,
                    100 * correct_count / scored_count,
                )

            return predictions, trajectories


# ---------------------------------------------------------------------------
# Factory for hydra instantiation
# ---------------------------------------------------------------------------

def create_rvlm_program(
    max_iterations: int = 20,
    images_for_last_n: int = 1,
    max_image_pixels: int = 1_000_000,
    use_category_tips: bool = True,
    question_concurrency: int = 4,
    vlm: dict[str, Any] | None = None,  # unused — RVLM doesn't need a VLM
) -> RVLMProgram:
    return RVLMProgram(
        max_iterations=max_iterations,
        images_for_last_n=images_for_last_n,
        max_image_pixels=max_image_pixels,
        use_category_tips=use_category_tips,
        question_concurrency=question_concurrency,
    )
