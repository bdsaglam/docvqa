"""Leanest+OCR solver — M+OCR extension cell per D-006.

Tool surface = leanest's ``batch_look`` + OCR's ``search`` + ``page_texts`` in
scope. **No** single-image ``look()`` registration: callers wanting a single
visual query use ``batch_look([(image, query)])[0]`` (same idiom as leanest).

This is a clean fork of ``leanest_solo_solver``. Distinct from
``flat_solo_solver``, which confounds the OCR extension with a single-image
``look()`` ergonomic wrapper.

Per D-007 (docs/paper/decisions.md, 2026-05-27), this solver owns its
category-tip prompts inline (``CATEGORY_TIPS`` + ``TOOL_HINTS`` below). The
shared dicts in ``docvqa.prompts`` are deprecated for paper solvers — do not
import them here.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import dspy
import logfire
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from docvqa.data import Document
from docvqa.metrics import evaluate_prediction
from docvqa.prompts import ANSWER_FORMATTING_RULES
from docvqa.rlm import LeanRLM, CodeRLM, ThinkingRLM, RLM
from docvqa.search import get_or_build_index
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

TASK_INSTRUCTIONS = (
    "You are a Document Visual Question Answering agent. You answer a question about a document by "
    "writing Python code, calling vision tools iteratively, and reasoning programmatically.\n\n"

    "## DATA\n"
    "- `question`: The question you must answer.\n"
    "- `page_texts`: OCR-extracted text per page (list of strings, 0-indexed). "
    "May be inaccurate — verify critical values visually via `batch_look()`.\n"
    "- `pages`: list of page images (PIL Images) (0-indexed). Pass to `batch_look()`, "
    "e.g. `batch_look([(pages[0], 'describe layout')])[0]`.\n\n"

    "## TOOLS\n"
    "- search(query, k=5) -> list[dict]: BM25 search over OCR text. Returns [{page, score, text}]. "
    "Useful for multi-page documents to locate relevant pages cheaply before any visual call. "
    "For single-page docs, read `page_texts` directly.\n"
    "- batch_look(requests) -> list[str]: Send one or more images to the VLM in parallel. "
    "Input: list of (image, query) tuples where image is any PIL Image "
    "(a page from `pages`, a crop via `pages[i].crop((l,t,r,b))`, etc). "
    "Returns: list of answers in same order. ALL visual queries go through this tool.\n"
    "  Example: batch_look([(pages[0], 'describe layout'), (pages[0].crop((0,0,500,500)), 'read text')])\n"
    "  For a single query, use: batch_look([(image, query)])[0]\n\n"

    "## APPROACH\n"
    "1. EXPLORE: Before answering, understand the document structure. "
    "Read `page_texts` and/or use `search()` to locate candidate pages cheaply, "
    "then use `batch_look` to verify or read regions OCR cannot (figures, tables, charts) — "
    "e.g. `batch_look([(pages[0], 'Describe layout...'), (pages[1], 'Describe layout...')])`.\n"
    "Build a mental map of the document.\n"
    "2. LOCATE: Find the specific region(s) relevant to the question. Use `search()` on multi-page "
    "documents to narrow the candidate set before any visual call.\n"
    "3. EXTRACT: Use `batch_look` with tight crops to read exact values. "
    "For fine details, crop first: `batch_look([(pages[i].crop((l,t,r,b)), query)])[0]`.\n"
    "4. VERIFY: Cross-check extracted values if ambiguous.\n"
    "5. SUBMIT: Once you have the answer, SUBMIT it.\n\n"

    "## GUIDELINES\n"
    "- Use `search()` + `page_texts` to locate relevant pages cheaply, then `batch_look()` to verify. "
    "Browsing every page visually on a long document is wasteful.\n"
    "- OCR text may be wrong or missing. For exact numbers, labels in figures, or any chart/table value, "
    "verify visually with `batch_look()` on a tight crop.\n"
    "- Full-page batch_look gives a broad overview. For fine details, crop first: `batch_look([(pages[i].crop((l,t,r,b)), query)])[0]`.\n"
    "- Use `pages[i].size` to get dimensions for cropping.\n"
    "- Ask the VLM ONE simple factual question per call. Do NOT combine multiple questions or ask it to reason. "
    "Extract raw facts, then count/compare/compute in Python.\n"
    "- VLM CONFLICT RESOLUTION: The VLM gives different answers across calls for the same region. "
    "When readings conflict, crop TIGHTER on the specific detail and do ONE tie-breaking read. "
    "Give more weight to higher-resolution crops. Never silently adopt a new number from a 'verification' pass.\n"
    "- SUPERLATIVES: For 'largest', 'first', 'last', 'only' questions — enumerate ALL candidates first, "
    "then select programmatically. Do NOT stop at the first match.\n"
    "- UNKNOWN RULES: Answer 'Unknown' when:\n"
    "  (a) A specific named entity (column name, layer number, variable) does not exist after thorough search.\n"
    "  (b) A chart/table explicitly shows N/A or missing data for the requested item.\n"
    "  Do NOT substitute a similar-sounding entity or extrapolate from nearby data.\n"
    "  Do NOT use narrative/descriptive text when a chart explicitly shows N/A.\n"
    "- COMPUTATION: When a question says 'total' or 'considering X and Y', it may require arithmetic. "
    "Extract all referenced values and compute explicitly in Python.\n"
    "- NEVER use outside/world knowledge. ALL answers MUST come from the document.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string.\n"
    '- Example: SUBMIT(answer="42")\n'
    "- The answer must follow these formatting rules:\n\n"

    + ANSWER_FORMATTING_RULES
)


# ---------------------------------------------------------------------------
# Category-specific tips (owned inline per D-007).
# Copied from leanest_solo_solver (the reconciled source-of-truth as of
# 2026-05-27 — val-leak-scrubbed, Leader-lines included, TEXT TRUNCATION
# bullet present on business_report, COUNTING-OBJECTS bullet on maps,
# CITED PAPER FINDINGS bullet on science_paper).
#
# Tool-surface translation: all visual references stay as `batch_look()`
# (we do not register `look()`). Where flat_solo's FLAT_SOLO_TOOL_HINTS
# adds an OCR-routing overlay (science_paper, slide, infographics), we
# include a `TOOL_HINTS` overlay below — adapted to use `batch_look()`
# instead of `look()` so the wording matches the actual tool surface.
# ---------------------------------------------------------------------------

CATEGORY_TIPS: dict[str, str] = {
    "engineering_drawing": (
        "- PRECISION IS CRITICAL: Crop tables and labels at full resolution before reading values.\n"
        "- BOM has two parallel numbering systems: ITEM NUMBERS (sequential index in the parts list) and "
        "PART / IDENTIFYING NUMBERS (the actual hardware identifier, often alphanumeric with dashes). "
        "Questions about 'part number' / 'identifying number' refer to the latter; 'item number' refers to the former.\n"
        "- 'VIEW IN DIRECTION X' labels indicate a viewing direction. The answer is the direction letter alone, "
        "not prefixed with 'Direction'.\n"
        "- For counting parts in the BOM (clamps, bolts, etc.), read the QTY column row-by-row and sum in code — "
        "don't estimate visually.\n"
        "- VLM OCR CONFUSION: Part numbers are almost always digits + dashes. If the VLM reads letters like I, O, l "
        "where digits 1, 0 would be expected, re-read at higher zoom. Common confusions: I↔1, O↔0, l↔1.\n"
        "- For labels or numbers adjacent to a specific schematic or view, crop tightly around that view rather "
        "than relying on a single full-page query — small text gets lost at thumbnail resolution.\n"
        "- LEADER LINES: when a label points to a part via a leader line, use `batch_look` on the label region "
        "and the pointed-to region in two separate crops to verify the connection — don't rely on a single "
        "full-page query that may mis-associate label and part by proximity.\n"
        "- DIMENSIONS: 'Width' typically refers to the shorter cross-sectional dimension (from a Section view), "
        "not the longest overall dimension (which is 'Length'). Dimensions tagged 'REF' (reference) are valid answers.\n"
    ),
    "business_report": (
        "- Crop tables at full resolution before reading numbers or labels — dense tables are hard to read at thumbnail zoom.\n"
        "- Multiple tables may contain similar-looking data. Verify the table's title/header matches the question's "
        "subject before extracting values.\n"
        "- For YoY / period-over-period calculations, extract raw values from the table first, then compute "
        "differences in Python — do not rely on the VLM for arithmetic.\n"
        "- CHART VALUES: VLM readings of bar/line chart values vary between calls. Use the first clear reading "
        "rather than re-querying the same chart to 'verify' — repeated reads add noise, not signal.\n"
        "- 'Broken down into' refers to immediate sub-categories only, not sub-sub-categories.\n"
        "- TEXT TRUNCATION: When a question asks for a phrase truncated at a punctuation boundary "
        "(first words before a punctuation mark, first sentence, etc.), read the full passage with "
        "`batch_look` and do the truncation in Python — the VLM over-shortens when asked to truncate directly.\n"
        "- PICTOGRAMS: When looking for a described pictogram among many, crop each icon individually and ask the "
        "VLM to describe it, rather than asking a single yes/no filtering question across all icons at once.\n"
        "- If a qualitative description (e.g., an adjective) does not appear in the table, look in the surrounding "
        "text paragraphs or footnotes.\n"
    ),
    "comics": (
        "- STORY MAP FIRST: For multi-story anthologies, build a story index before answering — scan each page "
        "to get (story title, page range, key characters). Match question keywords to the correct story.\n"
        "- COUNTING EVENTS: For 'how many times X happens', query panel-by-panel with HIGHLY SPECIFIC inclusion "
        "criteria — e.g., 'Is someone physically [exact action] in this panel? Exclude mentions in past-tense "
        "dialogue, near-misses, and aftermath.' Then count the positive panels in code.\n"
        "- VERIFY COUNTS: The VLM over-attributes actions in busy panels — it infers events from context clues "
        "(sound effects, weapons, postures) even when no action is depicted. After collecting candidates, "
        "re-examine each one with a tight crop and a disconfirming question ('Did this action ACTUALLY occur, or "
        "is it a near-miss / different action / aftermath?'). Expect many initial candidates to be false positives.\n"
        "- PANEL-BY-PANEL: When you need extractable events, ask 'what happens in each panel?' explicitly. "
        "Generic 'describe the page' queries miss the panel structure that makes events countable.\n"
        "- LITERAL VS FIGURATIVE: When a question contains qualifiers like 'in reality', 'actually', or 'truly', "
        "the answer likely contradicts the surface label/title shown in the panel — distinguish what something "
        "is called from what it factually is.\n"
        "- CHARACTER IDENTIFICATION: Use the exact term that appears in the speech bubbles. When the VLM gives "
        "conflicting answers about a small object or character, use narrative context (story setting, nearby "
        "objects, character role) to disambiguate.\n"
    ),
    "maps": (
        "- COARSE-TO-FINE: Start with a full-page view of the map for rough layout, then zoom into areas of "
        "interest (~800px crops), then tighter (~400px) for small text. Each step refines the previous.\n"
        "- COUNTING OBJECTS ON MAPS: For 'how many X are on the map', NEVER try to count from a full-page view — "
        "small objects are invisible at low resolution. Instead:\n"
        "  1. Estimate the object size relative to the map and pick a grid size so each tile shows individual "
        "objects clearly (large objects → 3x3; medium → 4x4 or 5x5; small dots/pins/symbols → 6x6 or more).\n"
        "  2. Split the map into tiles with ~15% overlap between adjacent tiles.\n"
        "  3. Per-tile, use `batch_look` to ask the VLM: 'List every [object] visible in this tile, with each "
        "one's relative position (top/bottom/left/right/center) and any distinguishing label nearby.'\n"
        "  4. In code, collect across tiles and deduplicate objects near tile boundaries by checking similar "
        "positions or matching labels.\n"
        "  5. Count the deduplicated list.\n"
        "- LOCATE INDEPENDENTLY: Find each landmark/feature with simple per-tile queries ('what labels are "
        "visible here?', 'is feature X present in this tile?'). Record approximate pixel positions using tile "
        "offset + relative position within the tile.\n"
        "- REASON WITH MATH: Compute spatial relationships in Python — distances, directions, relative "
        "positions — using the coordinates you collected. Basic vector math gives reliable answers with "
        "explicit error bounds.\n"
        "- LEGEND + ROAD TYPES: Crop the legend early. For road-type questions, crop the specific road segment "
        "at HIGH resolution alongside the legend and ask the VLM to compare the line style directly. Small "
        "differences (solid vs dashed, thin vs thick) are easy to misread at low resolution.\n"
        "- GRID COORDINATES: Cross-reference TWO sources — (1) crop the actual grid cell on the map to see "
        "what's there, and (2) look up the same coordinate in any feature index/legend that lists entries by "
        "grid coordinate. Disagreement between the two is usually an indexing-error trap.\n"
    ),
    "science_paper": (
        "- Papers can be long — locate the relevant section first (abstract, headings, figure/table captions) "
        "before reading in detail.\n"
        "- CITATION NUMBERS: For 'first/last citation on this page' style questions, treat citations as text "
        "patterns ([N], (Author, Year)) and enumerate them yourself in order rather than asking the VLM to "
        "identify them — VLM ordering of inline references is unreliable. Distinguish body-text citations from "
        "table headers and figure captions, which are often numbered separately.\n"
        "- CITED PAPER FINDINGS: To find what a cited work claims, first find its reference number in the "
        "bibliography (crop the references section), then locate the place(s) in body text where that number "
        "is discussed. If the cited paper's actual content isn't in this document, answer 'Unknown' rather "
        "than hallucinating from the title or context.\n"
        "- ABLATION STUDIES: Papers often contain multiple ablation studies on different components. Verify "
        "the section you're reading is about the specific component the question asks about, not a different "
        "subsystem.\n"
        "- If a question references a specific entity (layer number, model variant, dataset name) that does "
        "not appear anywhere in the document after thorough inspection, answer 'Unknown' — do not extrapolate "
        "from a similar-sounding entity.\n"
    ),
    "science_poster": (
        "- Posters are dense single-page documents. Crop specific sections at full resolution for precise values.\n"
        "- CHART ANNOTATIONS: If a chart has numeric labels printed directly on bars/lines, read those labels "
        "rather than estimating from bar heights — printed labels are exact, visual estimates are noisy.\n"
        "- For table values and percentages, crop the specific cell at full resolution before reading.\n"
        "- 'Percentage improvement' refers to the absolute difference in percentage points (e.g., 80% − 50% "
        "= 30 percentage points), not the relative change.\n"
        "- COLOR-CODED VALUES: For questions about colored numbers in a table (red, blue, highlighted), crop "
        "the table at maximum resolution and enumerate all candidates of that color before selecting — VLM "
        "color recall across an entire table is unreliable, but per-cell color checks are accurate.\n"
        "- GROUPED BAR CHARTS: A 'set of columns' / 'group of bars' refers to the bars at one x-axis position "
        "(one category, one benchmark), not all bars of one color across positions.\n"
    ),
    "infographics": (
        "- Infographics mix text, icons, and illustrations — a full-page view gives useful structural context "
        "before zooming in.\n"
        "- For precise numbers or dates, crop the specific data point at full resolution. For identifying "
        "broad visual elements (icons, sections, themes), a full-page view suffices.\n"
        "- SYSTEMATIC ENUMERATION: When a question asks for a first/last/only item that has or lacks some "
        "property, enumerate ALL items and their status before answering. Don't stop after finding two or "
        "three candidates — the answer hinges on which one is at the boundary.\n"
    ),
    "slide": (
        "- Slide decks can span many pages. Locate the relevant slide first by skimming titles/headers, then "
        "read in detail.\n"
        "- PAGE NAVIGATION: When a question refers to 'the page before X' or 'the page that contains Y', "
        "first locate X or Y, then verify the page index by cropping the page's header/title. Off-by-one "
        "errors on page indexing are common — double-check before submitting a page number or page-specific "
        "content.\n"
        "- For position-on-page questions (a specific word/bullet at the top/bottom/edge of a page), crop "
        "the relevant region at full resolution and read carefully.\n"
        "- Tables on slides are often small; crop at full resolution to read cell values.\n"
        "- EXACT ENTITY MATCHING: If a question references a specific column name, variable, or equation "
        "that does not appear anywhere in the document after thorough inspection, answer 'Unknown'. Do NOT "
        "substitute a similar-sounding name.\n"
        "- COMPUTATION: When a question says 'total', 'sum', or 'considering X and Y', extract all "
        "referenced values and compute in Python explicitly. Show the values entering the calculation "
        "before submitting.\n"
    ),
}


# Tool-routing overlay (M+OCR variant). Mirrors flat_solo's FLAT_SOLO_TOOL_HINTS
# for the three categories where BM25 + page_texts dominance materially helps,
# but every reference to `look()` is rewritten as `batch_look()` to match the
# leanest+OCR tool surface. Appended on top of CATEGORY_TIPS by
# `_get_category_tips`.

TOOL_HINTS: dict[str, str] = {
    "science_paper": (
        "- TOOL ROUTING: Papers can be very long — start with `search()` over the BM25 "
        "index and read `page_texts` to locate the relevant section before any visual "
        "tool calls. Use `batch_look()` only to verify or to read figures/tables.\n"
        "- CITATION NUMBERS: For 'first/last citation on this page' style questions, "
        "extract all `[N]` (or `(Author, Year)`) patterns from `page_texts` with a "
        "Python regex ordered by position. Do NOT ask the VLM to identify citation "
        "order — its inline ordering is unreliable.\n"
        "- CITED-WORK LOOKUP: To find what a cited work claims, find its reference "
        "number in the bibliography (via `search()` for the title), then `search()` "
        "the body text for that bracketed number.\n"
    ),
    "slide": (
        "- TOOL ROUTING: Slide decks span many pages — use `search()` and `page_texts` "
        "to find the relevant slide first, then crop / `batch_look()` for fine detail. "
        "Browsing slide-by-slide visually is wasteful.\n"
        "- PAGE NAVIGATION: For 'the page before X' or 'page where Y is mentioned', "
        "locate X / Y in `page_texts` first, then verify the page index by cropping "
        "the page's header/title — off-by-one errors on page indices are common.\n"
    ),
    "infographics": (
        "- TOOL ROUTING: A full-page `batch_look()` pass gives useful structural "
        "context before zooming. OCR (`page_texts`) on infographics often describes "
        "images instead of reading them, so prefer `batch_look()` for text that lives "
        "on icons or illustrations.\n"
    ),
}


def _get_category_tips(category: str) -> str:
    """Get per-category tips for leanest+OCR (CATEGORY_TIPS + TOOL_HINTS overlay)."""
    base = CATEGORY_TIPS.get(category, "")
    tool = TOOL_HINTS.get(category, "")
    if not base and not tool:
        return ""
    return f"## CATEGORY-SPECIFIC TIPS ({category})\n{base}{tool}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    page_dir: str
    num_pages: int
    search_index: Any = None
    page_texts: list[str] | None = None


def _format_page_texts(page_texts: list[str]) -> list[str]:
    return [t.strip() or "[No text extracted - use batch_look() for visual content]" for t in page_texts]


def _build_signature(instructions: str = TASK_INSTRUCTIONS) -> dspy.Signature:
    fields: dict = {
        "question": (
            str,
            dspy.InputField(desc="The question to answer about the document"),
        ),
        "doc_info": (str, dspy.InputField(desc="Document metadata: category and page count")),
        "page_texts": (
            list,
            dspy.InputField(desc="OCR-extracted text per page. List of strings, one per page (0-indexed)."),
        ),
        "answer": (
            str,
            dspy.OutputField(desc="The answer string for the question."),
        ),
    }
    return dspy.Signature(fields, instructions)


def _create_tools(vlm_predict: dspy.Predict, vlm_lm: dspy.LM, ctx: RunContext, batch_concurrency: int = 8) -> list:
    from PIL import Image as PILImage

    def _look_impl(image_path: str, query: str) -> str:
        """Internal: load image from path and send to VLM. Used only by _batch_look_impl —
        NOT registered as a sandbox-visible tool."""
        with logfire.span("look", image_path=image_path, query=query) as span:
            img = PILImage.open(image_path)
            with dspy.context(lm=vlm_lm):
                result = vlm_predict(image=dspy.Image(img), query=query)
                answer = result.answer or ""
                span.set_attribute("answer", answer[:2000])
                return answer

    def _batch_look_impl(requests_json: str) -> list[str]:
        """Internal: batch VLM calls in parallel. Input is JSON list of {path, query}."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import json as _json
        requests = _json.loads(requests_json)
        if not requests:
            return []
        results: list[str] = [""] * len(requests)

        def _do(idx: int, path: str, query: str) -> tuple[int, str]:
            return idx, _look_impl(path, query)

        is_vertex = "vertex_ai" in (vlm_lm.model if hasattr(vlm_lm, 'model') else str(vlm_lm))
        max_w = min(len(requests), 2 if is_vertex else batch_concurrency)
        with logfire.span("batch_look", num_requests=len(requests)):
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                futures = {
                    pool.submit(_do, i, r["path"], r["query"]): i
                    for i, r in enumerate(requests)
                }
                for future in as_completed(futures):
                    idx, answer = future.result()
                    results[idx] = answer
        return results

    def _search(query: str, k: int = 5) -> list[dict]:
        """Search document text using BM25. Returns list of {page, score, text} records."""
        if ctx.search_index is None:
            return [{"error": "No search index available"}]
        with logfire.span("search", query=query, k=k) as span:
            import bm25s
            import Stemmer
            chunks = ctx.search_index._chunk_meta
            query_tokens = bm25s.tokenize([query], stemmer=Stemmer.Stemmer("english"))
            n = min(k, len(chunks))
            results, scores = ctx.search_index.retrieve(query_tokens, k=n)
            records = []
            for idx, score in zip(results[0], scores[0]):
                if score <= 0:
                    continue
                chunk = chunks[idx]
                records.append({"page": chunk["page"], "score": round(float(score), 2), "text": chunk["text"]})
            span.set_attribute("num_results", len(records))
            return records

    # IMPORTANT: only _batch_look_impl and _search are registered as sandbox-visible
    # tool proxies. _look_impl is internal-only — the agent never sees a `look()`
    # symbol in its REPL. Single-image queries go through `batch_look([(img, q)])[0]`.
    return [_batch_look_impl, _search]


def _build_sandbox_code(page_dir: str, num_pages: int) -> str:
    """Build sandbox code that loads pages as PIL Images and defines
    `batch_look()` and `search()`. No `look()` is defined."""
    return f'''
import os
import tempfile
from PIL import Image

# Load all pages as PIL Images
Image.MAX_IMAGE_PIXELS = 500_000_000
pages = []
for i in range({num_pages}):
    path = os.path.join({page_dir!r}, f"page_{{i}}.png")
    assert os.path.exists(path), f"Page image not found: {{path}}"
    pages.append(Image.open(path))
assert len(pages) == {num_pages}, f"Expected {{num_pages}} pages, got {{len(pages)}}"

def search(query, k=5):
    """BM25 search over OCR text. Returns list of {{page, score, text}} dicts."""
    return _search(query, k)

def batch_look(requests):
    """Send multiple images to the VLM in parallel. Much faster than sequential calls.
    Input: list of (image, query) tuples. Returns: list of str answers (same order).
    Example: batch_look([(pages[0], "layout?"), (pages[1].crop((0,0,500,500)), "read text")])
    For a single query, use: batch_look([(image, query)])[0]"""
    import json as _json
    paths = []
    for image, query in requests:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp, format="PNG")
        tmp.close()
        paths.append({{"path": tmp.name, "query": query}})
    return _batch_look_impl(_json.dumps(paths))
'''


# ---------------------------------------------------------------------------
# LeanestOcrProgram
# ---------------------------------------------------------------------------

class LeanestOcrProgram:
    """Leanest+OCR solver — each question solved independently.

    Tool surface: `batch_look()` (from leanest) + `search()` + `page_texts`
    (from OCR). No `look()` is registered — that ergonomic wrapper is what
    distinguishes this solver from `flat_solo`, the prior M+OCR variant.
    """

    def __init__(
        self,
        vlm_lm: dspy.LM,
        max_iterations: int = 20,
        rlm_type: str = "lean",
        page_factor: float = 1.5,
        question_concurrency: int = 1,
        batch_concurrency: int = 8,
    ):
        self.vlm_lm = vlm_lm
        self.max_iterations = max_iterations
        self.rlm_type = rlm_type
        self.page_factor = page_factor
        self.question_concurrency = question_concurrency
        self.batch_concurrency = batch_concurrency

        self.vlm_predict = dspy.Predict(
            dspy.Signature(
                {
                    "image": (dspy.Image, dspy.InputField(desc="Page or cropped region image")),
                    "query": (str, dspy.InputField(desc="What to look for or describe")),
                    "answer": (str, dspy.OutputField(desc="Concise response")),
                },
                "Analyze the image content strictly to answer the query. "
                "Transcribe numbers and characters exactly. "
                "For technical drawings, trace leader lines and arrows to connect labels to their specific parts. "
                "Output ONLY the concise answer. If the information is missing, output 'Unknown'.",
            )
        )

    def solve_document(self, document: Document) -> tuple[dict[str, str], dict[str, list[dict]]]:
        """Solve all questions for a document, one question at a time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, img in enumerate(document.images):
                img.save(os.path.join(tmpdir, f"page_{i}.png"), format="PNG")

            search_index = None
            if document.page_texts:
                search_index = get_or_build_index(
                    document.doc_id,
                    document.page_texts,
                    bm25_dir=document.bm25_dir,
                )

            ctx = RunContext(
                page_dir=tmpdir,
                num_pages=len(document.images),
                search_index=search_index,
                page_texts=document.page_texts,
            )

            doc_info = f"Category: {document.doc_category}, Pages: {len(document.images)}"
            page_texts = _format_page_texts(document.page_texts) if document.page_texts else None
            if page_texts is None:
                page_texts = ["[No OCR text available]"]

            num_pages = len(document.images)
            page_bonus = min(10, self.page_factor * math.ceil(math.sqrt(max(0, num_pages - 9))))
            max_iter = self.max_iterations + int(page_bonus)

            tips = _get_category_tips(document.doc_category)
            instructions = TASK_INSTRUCTIONS + ("\n" + tips if tips else "")
            tools = _create_tools(self.vlm_predict, self.vlm_lm, ctx, self.batch_concurrency)
            sandbox_code = _build_sandbox_code(tmpdir, len(document.images))

            def _solve_question(q):
                """Solve a single question. Returns (question_id, answer, trajectory)."""
                with logfire.span(
                    "solve_leanest_ocr",
                    doc_id=document.doc_id,
                    question_id=q.question_id,
                    question=q.question[:200],
                ) as q_span:
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
                        "Leanest+OCR (%s) Q %s: max_iterations=%d (page_bonus=%d)",
                        self.rlm_type, q.question_id, max_iter, int(page_bonus),
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
                        return rlm(
                            question=q.question,
                            doc_info=doc_info,
                            page_texts=page_texts,
                        )

                    result = _solve_one()
                    answer = str(result.answer or "").strip()
                    trajectory = result.trajectory

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
                            "Solo Q %s: %s (GT=%s, PRED=%s)",
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
                logger.info("Leanest+OCR: running %d questions with concurrency=%d", len(document.questions), max_w)
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
                    "Leanest+OCR doc %s: %d/%d = %.1f%%",
                    document.doc_id, correct_count, scored_count,
                    100 * correct_count / scored_count,
                )

            return predictions, trajectories


# ---------------------------------------------------------------------------
# Factory for hydra instantiation
# ---------------------------------------------------------------------------

def create_leanest_ocr_program(
    max_iterations: int = 20,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    batch_concurrency: int = 8,
) -> LeanestOcrProgram:
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

    return LeanestOcrProgram(
        vlm_lm=vlm_lm,
        max_iterations=max_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
        batch_concurrency=batch_concurrency,
    )
