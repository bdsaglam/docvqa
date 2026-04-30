"""Flat solo solver — solves each question independently with direct VLM calls.

Like flat_batch_solver but each question gets its own RLM session.
The agent focuses on a single question at a time, submitting a single answer string.

Trade-off: no cross-question knowledge sharing, but each question gets
full iteration budget and no interference from other questions.
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
from docvqa.prompts import ANSWER_FORMATTING_RULES, get_category_tips
from docvqa.lean_rlm import LeanRLM
from docvqa.code_rlm import CodeRLM
from docvqa.thinking_rlm import ThinkingRLM
from docvqa.rlm import RLM
from docvqa.search import get_or_build_index
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

TASK_INSTRUCTIONS = (
    "You are a Document Visual Question Answering agent. You answer a question about a document by "
    "writing Python code, calling vision tools iteratively, and reasoning programmatically.\n\n"

    "## CORE PRINCIPLE: PERCEIVE WITH VLM, REASON IN PYTHON\n"
    "The VLM is a perception tool — it reports what it sees. It does NOT reason, compare, or compute.\n"
    "YOU do all reasoning, comparison, counting, and computation in Python code.\n\n"

    "## DATA\n"
    "- `question`: The question you must answer.\n"
    "- `page_texts`: OCR-extracted text per page. May be inaccurate — verify critical values visually.\n"
    "- `pages`: list of page images (PIL Images) (0-indexed).\n\n"

    "## TOOLS\n"
    "- search(query, k=5) -> list[dict]: BM25 search over OCR text. Returns [{page, score, text}]. "
    "Useful for multi-page documents to locate relevant pages. For single-page docs, read `page_texts` directly.\n"
    "- look(image, query) -> str: "
    "Send any PIL Image to the VLM with a PERCEPTION query. Returns a description of what is visible. "
    "`image` can be a page (`pages[0]`), a crop (`pages[0].crop((l,t,r,b))`), or any PIL Image. "
    "Full pages are downscaled — for fine details, crop first.\n"
    "- batch_look(requests) -> list[str]: Parallel VLM calls. "
    "Input: list of (image, query) tuples. Returns: list of descriptions in same order. "
    "MUCH faster than sequential look() — use whenever you have 2+ independent queries.\n\n"

    "## HOW TO USE `look`/`batch_look` — PERCEPTION QUERIES ONLY\n"
    "The VLM describes/transcribes what it sees. Never ask it to reason or answer your question directly.\n\n"
    "GOOD perception queries (ask the VLM to report what it sees):\n"
    '- look(crop, "Transcribe all text in this region, preserving layout")\n'
    '- look(crop, "List all numbers and their labels in this table region")\n'
    '- look(crop, "What roads, intersections, and labels are visible in this area?")\n'
    '- look(crop, "Describe each panel: characters, actions, speech bubbles")\n'
    '- look(crop, "What colors are used for each data series in this chart?")\n'
    '- look(crop, "List all items in this BOM table: item number, part number, quantity")\n\n'
    "BAD reasoning queries (never ask these):\n"
    '- "What is the largest value?" → Instead: transcribe all values, find max in Python\n'
    '- "Which road connects A to B?" → Instead: list all roads/labels in the area, trace in Python\n'
    '- "How many times does X happen?" → Instead: describe each panel, count matches in Python\n\n'

    "USE `batch_look` FOR EFFICIENCY:\n"
    "```python\n"
    "# Survey all pages at once (comics, slides, multi-page docs)\n"
    'descs = batch_look([(pages[i], "Describe the content, layout, and any text on this page") for i in range(len(pages))])\n'
    "for i, d in enumerate(descs): print(f'Page {i}: {d}')\n\n"
    "# Read a table in overlapping strips\n"
    "w, h = pages[0].size\n"
    "strip_h = h // 4\n"
    'strips = batch_look([(pages[0].crop((0, y, w, y+strip_h)), "Transcribe all rows in this table section")\n'
    "                     for y in range(0, h, strip_h - strip_h//4)])\n\n"
    "# Grid scan a map/poster for landmarks\n"
    'tiles = batch_look([(pages[0].crop((c*w//3, r*h//3, (c+1)*w//3, (r+1)*h//3)), "List all labels, roads, and landmarks visible")\n'
    "                    for r in range(3) for c in range(3)])\n\n"
    "# Cross-verify a critical value with redundant reads\n"
    "from collections import Counter\n"
    "crop = pages[0].crop((100, 200, 500, 350))\n"
    "reads = batch_look([\n"
    '    (crop, "Transcribe the number in this cell"),\n'
    '    (crop, "What digits are shown here?"),\n'
    '    (pages[0].crop((80, 180, 520, 370)), "Transcribe all text in this region"),  # slightly wider crop\n'
    "])\n"
    "print('Reads:', reads)  # compare in Python, take consensus\n"
    "```\n\n"

    "## APPROACH\n"
    "1. EXPLORE: Read `page_texts`, then batch_look full pages to understand layout.\n"
    "2. LOCATE: Use search() and page_texts to find relevant regions. Get page dimensions with `pages[i].size`.\n"
    "3. EXTRACT: Crop tightly to the region of interest, then `look`/`batch_look` to transcribe what's there. "
    "Store extracted data in Python variables (dicts, lists).\n"
    "4. REASON: Compute the answer in Python from extracted data — filter, count, compare, calculate.\n"
    "5. VERIFY: For critical values (numbers, names), use batch_look with redundant reads "
    "(same region with different phrasings, or slightly different crops). Compare in Python, take consensus.\n"
    "6. SUBMIT: Once confident, SUBMIT the answer.\n\n"

    "## GUIDELINES\n"
    "- CROP FIRST: For any detail, crop tightly around it before calling look. Don't rely on full-page reads for precision.\n"
    "- STRUCTURED EXTRACTION: When reading tables, store rows as dicts/lists in Python. When reading charts, store data points.\n"
    "- CONFLICT RESOLUTION: When repeated reads disagree, crop TIGHTER and re-read. Trust higher-resolution crops.\n"
    "- SUPERLATIVES ('largest', 'first', 'last'): Extract ALL candidates into a list, then select in Python.\n"
    "- MULTI-HOP QUESTIONS: If a question has nested references, decompose into numbered sub-steps in code comments. "
    "Solve each step one at a time, printing intermediate results.\n"
    "- UNKNOWN: Answer 'Unknown' only when a specific entity genuinely does not exist after thorough search. "
    "Do NOT substitute similar entities or extrapolate.\n"
    "- NEVER use outside/world knowledge. ALL answers MUST come from the document.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string.\n"
    '- Example: SUBMIT(answer="42")\n'
    "- The answer must follow these formatting rules:\n\n"

    + ANSWER_FORMATTING_RULES
)


# ---------------------------------------------------------------------------
# Helpers (reused from flat_batch_solver)
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    page_dir: str
    num_pages: int
    search_index: Any = None
    page_texts: list[str] | None = None


def _format_page_texts(page_texts: list[str]) -> list[str]:
    return [t.strip() or "[No text extracted - use look() for visual content]" for t in page_texts]


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


def _create_tools(vlm_predict: dspy.Predict, vlm_lm: dspy.LM, ctx: RunContext) -> list:
    from PIL import Image as PILImage

    def _look_impl(image_path: str, query: str) -> str:
        """Internal: load image from path and send to VLM."""
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
        max_w = min(len(requests), 2 if is_vertex else 8)
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

    return [_look_impl, _batch_look_impl, _search]


def _build_sandbox_code(page_dir: str, num_pages: int) -> str:
    """Build sandbox code that loads pages as PIL Images and defines `look()`."""
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

def look(image, query):
    """Send an image to the VLM with a query. `image` can be any PIL Image
    (a page from `pages`, a crop via `image.crop(...)`, or any processed image).
    Returns the VLM's text response."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp, format="PNG")
    tmp.close()
    return _look_impl(tmp.name, query)

def search(query, k=5):
    """BM25 search over OCR text. Returns list of {{page, score, text}} dicts."""
    return _search(query, k)

def batch_look(requests):
    """Send multiple images to the VLM in parallel. Much faster than sequential look() calls.
    Input: list of (image, query) tuples. Returns: list of str answers (same order).
    Example: batch_look([(pages[0], "layout?"), (pages[1].crop((0,0,500,500)), "read text")])"""
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
# FlatSoloProgram
# ---------------------------------------------------------------------------

class FlatSoloProgram:
    """Flat solo solver — each question solved independently, direct VLM calls."""

    def __init__(
        self,
        vlm_lm: dspy.LM,
        max_iterations: int = 20,
        rlm_type: str = "lean",
        page_factor: float = 1.5,
        question_concurrency: int = 1,
    ):
        self.vlm_lm = vlm_lm
        self.max_iterations = max_iterations
        self.rlm_type = rlm_type
        self.page_factor = page_factor
        self.question_concurrency = question_concurrency

        self.vlm_predict = dspy.Predict(
            dspy.Signature(
                {
                    "image": (dspy.Image, dspy.InputField(desc="Page or cropped region image")),
                    "query": (str, dspy.InputField(desc="What to describe or transcribe in the image")),
                    "answer": (str, dspy.OutputField(desc="Detailed description of what is visible")),
                },
                "You are a visual transcription tool. Describe and transcribe what you observe in the image "
                "related to the query. Be thorough and precise.\n\n"
                "Rules:\n"
                "- Transcribe ALL text, numbers, and labels exactly as written.\n"
                "- For tables: list each row with column values.\n"
                "- For charts: report axes, labels, data values, colors.\n"
                "- For diagrams: describe components, connections, labels, spatial positions.\n"
                "- For maps: list visible roads, landmarks, labels, and their relative positions.\n"
                "- For comics/illustrations: describe characters, actions, speech bubble text, panel layout.\n"
                "- Note colors, highlights, and emphasis where visible.\n"
                "- Do NOT answer questions. Do NOT reason, compare, or compute. Just describe what you see.\n"
                "- If nothing relevant is visible, say 'Nothing relevant visible'.",
            )
        )

    def solve_document(self, document: Document) -> tuple[dict[str, str], dict[str, list[dict]]]:
        """Solve all questions for a document, one question at a time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, img in enumerate(document.images):
                img.save(os.path.join(tmpdir, f"page_{i}.png"), format="PNG")

            search_index = None
            if document.page_texts:
                search_index = get_or_build_index(document.doc_id, document.page_texts)

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
            page_bonus = self.page_factor * math.ceil(math.sqrt(max(0, num_pages - 9)))
            max_iter = min(self.max_iterations + int(page_bonus), 40)

            tips = get_category_tips(document.doc_category)
            instructions = TASK_INSTRUCTIONS + ("\n" + tips if tips else "")
            tools = _create_tools(self.vlm_predict, self.vlm_lm, ctx)
            sandbox_code = _build_sandbox_code(tmpdir, len(document.images))

            def _solve_question(q):
                """Solve a single question. Returns (question_id, answer, trajectory)."""
                with logfire.span(
                    "solve_flat_solo",
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
                        "Flat solo (%s) Q %s: max_iterations=%d (page_bonus=%d)",
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

                    try:
                        result = _solve_one()
                        answer = str(result.answer or "").strip()
                        trajectory = result.trajectory
                    except Exception as e:
                        logger.warning("Flat solo failed for Q '%s': %s", q.question_id, e)
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
                # Sequential
                for q in document.questions:
                    qid, answer, trajectory = _solve_question(q)
                    predictions[qid] = answer
                    trajectories[qid] = trajectory
            else:
                # Parallel
                from concurrent.futures import ThreadPoolExecutor, as_completed
                max_w = min(self.question_concurrency, len(document.questions))
                logger.info("Flat solo: running %d questions with concurrency=%d", len(document.questions), max_w)
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
                    "Flat solo doc %s: %d/%d = %.1f%%",
                    document.doc_id, correct_count, scored_count,
                    100 * correct_count / scored_count,
                )

            return predictions, trajectories


# ---------------------------------------------------------------------------
# Factory for hydra instantiation
# ---------------------------------------------------------------------------

def create_flat_solo_program(
    max_iterations: int = 20,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
) -> FlatSoloProgram:
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

    return FlatSoloProgram(
        vlm_lm=vlm_lm,
        max_iterations=max_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
    )
