"""Flat batch solver — answers ALL questions with direct VLM calls, no sub-agents.

Unlike batch_rlm_solver which delegates to a page agent (sub-agent that can
zoom/crop autonomously), this solver gives the main RLM agent direct VLM
tools: `look` for full-page queries and `inspect` for cropped regions.
The main agent handles all reasoning — no nested agent hierarchy.

Trade-off: cheaper per VLM call (no sub-agent overhead), but the main agent
must be smarter about what to look at and how to crop.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
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
    "You are a Document Visual Question Answering agent. You answer questions about documents by "
    "writing Python code, calling vision tools iteratively, and reasoning programmatically.\n\n"

    "## DATA\n"
    "- `questions`: JSON list of {question_id, question} dicts. You must answer ALL of them.\n"
    "- `page_texts`: OCR-extracted text per page. May be inaccurate — verify critical values visually.\n\n"
    "- `pages`: list of page images (PIL Images) (0-indexed). Pass to `look()`, e.g. `look(pages[0], 'describe layout')`.\n\n"

    "## TOOLS\n"
    "- search(query, k=5) -> list[dict]: BM25 search over OCR text. Returns [{page, score, text}]. "
    "Useful for multi-page documents to locate relevant pages. For single-page docs, read `page_texts` directly.\n"
    "- look(image, query) -> str: "
    "Send any PIL Image to the VLM with a query. `image` can be a page from `pages` (e.g. `pages[0]`), "
    "a crop (e.g. `pages[0].crop((left, top, right, bottom))`), or any processed image. "
    "Full pages are downscaled — for fine details, crop first using PIL.\n"
    "- batch_look(requests) -> list[str]: Parallel VLM calls. "
    "Input: list of (image, query) tuples. Returns: list of answers in same order. "
    "Much faster than sequential look() calls — use when you have multiple independent queries.\n"
    "- ocr(image) -> str: OCR extraction with table structure recognition. "
    "Input: any PIL Image (page or crop). Returns structured Markdown.\n\n"

    "## look() vs ocr()\n"
    "Both accept a PIL Image but have different strengths:\n"
    "- look(image, query): Visual understanding — answers a question about the image. "
    "Good at interpreting charts, diagrams, maps, spatial relationships.\n"
    "- ocr(image): Text extraction — returns all text as structured Markdown. "
    "Good at preserving table row/column structure and reading dense or small text.\n"
    "They complement each other. You can use both on the same region: "
    "e.g. ocr() to get exact table values, then look() to clarify an ambiguous cell. "
    "If one tool gives a garbled or uncertain result, try the other.\n"
    
    "## APPROACH\n"
    "1. EXPLORE: Before answering any question, understand the document structure. "
    "Read `page_texts`, then use `look` to survey pages — "
    "'Describe the layout: what sections, tables, figures, and labels are present and where are they positioned?' "
    "Build a mental map of the document.\n"
    "2. PLAN: Look at all questions together. Group questions that need the same region or data source.\n"
    "3. SOLVE SEQUENTIALLY: Focus on one question (or a small group of related questions) at a time. "
    "Fully solve it before moving to the next. Knowledge you gain while solving one question "
    "(extracted tables, page layouts, values) carries over — use it for subsequent questions. "
    "Do NOT try to answer all questions in a single step.\n"
    "4. SUBMIT: Once all questions are answered, SUBMIT all answers together as a dict.\n\n"

    "## GUIDELINES\n"
    "- Full-page `look` gives a broad overview. For fine details, crop first: `look(pages[i].crop((l,t,r,b)), query)`.\n"
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
    "- Be efficient. Reuse observations across questions.\n"
    "- NEVER use outside/world knowledge. ALL answers MUST come from the document.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a dict mapping each question_id to its answer string.\n"
    '- Example: SUBMIT(answers={"q1": "42", "q2": "Tokyo"})\n'
    "- Each answer must follow these formatting rules:\n\n"

    + ANSWER_FORMATTING_RULES
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_page_texts(page_texts: list[str]) -> list[str]:
    return [t.strip() or "[No text extracted - use look() for visual content]" for t in page_texts]


def _build_signature(instructions: str = TASK_INSTRUCTIONS) -> dspy.Signature:
    fields: dict = {
        "questions": (
            str,
            dspy.InputField(desc="JSON list of {question_id, question} dicts — answer ALL of them"),
        ),
        "doc_info": (str, dspy.InputField(desc="Document metadata: category and page count")),
        "page_texts": (
            list,
            dspy.InputField(desc="OCR-extracted text per page. List of strings, one per page (0-indexed)."),
        ),
        "answers": (
            str,
            dspy.OutputField(desc="Dict mapping question_id to answer string. Must include ALL question_ids."),
        ),
    }
    return dspy.Signature(fields, instructions)


@dataclass
class RunContext:
    page_dir: str
    num_pages: int
    search_index: Any = None
    page_texts: list[str] | None = None



def _ocr_via_docling(image_path: str, docling_url: str = "http://localhost:5001") -> str:
    """Send an image to docling-serve for OCR + table structure extraction.

    Returns structured markdown text.  Skips picture_description (slow)
    since the agent already has VLM for visual understanding.
    """
    import base64
    from io import BytesIO
    from PIL import Image as PILImage

    img = PILImage.open(image_path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "options": {
            "to_formats": ["md"],
            "from_formats": ["image"],
            "do_ocr": True,
            "do_table_structure": True,
            "do_picture_description": False,
            "image_export_mode": "placeholder",
        },
        "sources": [{"kind": "file", "base64_string": b64, "filename": "crop.png"}],
    }

    import requests as http_requests

    # Submit async task
    resp = http_requests.post(
        f"{docling_url}/v1/convert/source/async", json=payload, timeout=60,
    )
    resp.raise_for_status()
    task_id = resp.json()["task_id"]

    # Poll until done (timeout after 120s)
    import time as _time

    deadline = _time.time() + 120
    while _time.time() < deadline:
        poll = http_requests.get(
            f"{docling_url}/v1/status/poll/{task_id}?wait=2", timeout=10,
        )
        status = poll.json().get("task_status", "")
        if status == "success":
            result = http_requests.get(
                f"{docling_url}/v1/result/{task_id}", timeout=30,
            )
            result.raise_for_status()
            md = result.json()["document"]["md_content"]
            return md.replace("\n\n<!-- image -->", "").replace("<!-- image -->", "")
        if status == "failure":
            error = poll.json().get("error_message", "unknown")
            return f"[OCR failed: {error}]"
        _time.sleep(1)

    return "[OCR timed out after 120s]"


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

        # Lower concurrency for Vertex AI models to avoid rate limits
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

    def _ocr_impl(image_path: str) -> str:
        """Internal: send image to docling-serve for structured OCR extraction."""
        with logfire.span("ocr", image_path=image_path) as span:
            result = _ocr_via_docling(image_path)
            span.set_attribute("result_length", len(result))
            span.set_attribute("result_preview", result[:2000])
            return result

    return [_look_impl, _batch_look_impl, _search, _ocr_impl]


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

def ocr(image):
    """Extract text from an image using OCR with table structure recognition.
    Returns structured Markdown (with proper table formatting, headers, etc.).
    Best for: tables, dense text, small text that VLM struggles with.
    Use look() instead when you need visual understanding (charts, diagrams, spatial reasoning).
    Input: any PIL Image (a page or a crop)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp, format="PNG")
    tmp.close()
    return _ocr_impl(tmp.name)
'''


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def _parse_answers(raw: str, expected_ids: set[str]) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}

    raw = str(raw).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except (json.JSONDecodeError, ValueError):
        pass

    json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError):
            pass

    result = {}
    for qid in expected_ids:
        pattern = re.compile(rf'["\']?{re.escape(qid)}["\']?\s*[:=]\s*["\']([^"\']+)["\']', re.IGNORECASE)
        m = pattern.search(raw)
        if m:
            result[qid] = m.group(1).strip()

    if result:
        return result

    if len(expected_ids) == 1:
        qid = next(iter(expected_ids))
        return {qid: raw}

    logger.warning("Could not parse batch answers: %s", raw[:200])
    return {}


# ---------------------------------------------------------------------------
# FlatBatchProgram
# ---------------------------------------------------------------------------

class FlatBatchProgram:
    """Flat batch solver — direct VLM calls, no sub-agent hierarchy."""

    def __init__(
        self,
        vlm_lm: dspy.LM,
        iterations_per_question: int = 4,
        base_iterations: int = 6,
        rlm_type: str = "standard",
        page_factor: float = 1.5,
        max_iterations: int = 40,
    ):
        self.vlm_lm = vlm_lm
        self.iterations_per_question = iterations_per_question
        self.base_iterations = base_iterations
        self.rlm_type = rlm_type
        self.page_factor = page_factor
        self.max_iterations = max_iterations

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
        """Solve all questions for a document in a single flat batch session."""
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

            # Scale limits by number of questions and pages
            num_q = len(document.questions)
            num_pages = len(document.images)
            page_bonus = self.page_factor * math.ceil(math.sqrt(max(0, num_pages - 9)))
            max_iter = min(
                self.base_iterations + self.iterations_per_question * num_q + int(page_bonus),
                self.max_iterations,
            )

            # Build RLM with category-specific instructions
            tips = get_category_tips(document.doc_category)
            instructions = TASK_INSTRUCTIONS + ("\n" + tips if tips else "")
            tools = _create_tools(self.vlm_predict, self.vlm_lm, ctx)

            RLMClass = {"code": CodeRLM, "lean": LeanRLM, "thinking": ThinkingRLM}.get(self.rlm_type, RLM)
            rlm = RLMClass(
                signature=_build_signature(instructions),
                max_iterations=max_iter,
                max_llm_calls=max_iter * 3,
                tools=tools,
                verbose=True,
                sandbox_code=_build_sandbox_code(tmpdir, len(document.images)),
            )
            logger.info(
                "Flat batch (%s) limits for %d questions, %d pages: max_iterations=%d (page_bonus=%d)",
                self.rlm_type, num_q, num_pages, max_iter, int(page_bonus),
            )

            # Use short keys (q1, q2, ...) to save tokens
            short_to_full = {}
            questions_list = []
            for i, q in enumerate(document.questions):
                short_id = f"q{i + 1}"
                short_to_full[short_id] = q.question_id
                questions_list.append({"question_id": short_id, "question": q.question})
            questions_json = json.dumps(questions_list)
            short_ids = set(short_to_full.keys())

            if page_texts is None:
                page_texts = ["[No OCR text available]"]

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
            def _solve_batch():
                return rlm(
                    questions=questions_json,
                    doc_info=doc_info,
                    page_texts=page_texts,
                )

            with logfire.span(
                "solve_flat_batch",
                doc_id=document.doc_id,
                doc_category=document.doc_category,
                num_questions=len(questions_list),
                num_pages=len(document.images),
            ) as batch_span:
                try:
                    result = _solve_batch()
                    raw_answers = result.answers
                    trajectory = result.trajectory
                except Exception as e:
                    logger.warning("Flat batch RLM failed for doc '%s': %s", document.doc_id, e)
                    raw_answers = "{}"
                    trajectory = []

                short_answers = _parse_answers(raw_answers, short_ids)
                answers_dict = {
                    short_to_full[k]: v for k, v in short_answers.items() if k in short_to_full
                }
                batch_span.set_attribute("num_iterations", len(trajectory))
                batch_span.set_attribute("num_parsed_answers", len(answers_dict))

                predictions = {}
                trajectories = {}
                correct_count = 0
                scored_count = 0

                for q in document.questions:
                    answer = answers_dict.get(q.question_id, "Unknown")
                    if not answer or answer.strip() == "":
                        answer = "Unknown"
                    predictions[q.question_id] = answer
                    trajectories[q.question_id] = trajectory

                    if q.answer is not None:
                        scored_count += 1
                        is_correct, extracted = evaluate_prediction(answer, q.answer)
                        if is_correct:
                            correct_count += 1
                        with logfire.span(
                            "flat_batch_question_result",
                            question_id=q.question_id,
                            question=q.question[:200],
                            is_correct=is_correct,
                            prediction=answer[:200],
                            ground_truth=q.answer[:200],
                            extracted_answer=extracted[:200],
                        ):
                            pass
                        logger.info(
                            "Flat Q %s: %s (GT=%s, PRED=%s)",
                            q.question_id,
                            "CORRECT" if is_correct else "WRONG",
                            q.answer[:40],
                            extracted[:40],
                        )

                if scored_count > 0:
                    batch_span.set_attribute("accuracy", correct_count / scored_count)
                    batch_span.set_attribute("correct", correct_count)
                    batch_span.set_attribute("scored_questions", scored_count)

            return predictions, trajectories


# ---------------------------------------------------------------------------
# Factory for hydra instantiation
# ---------------------------------------------------------------------------

def create_flat_batch_ocr_program(
    iterations_per_question: int = 4,
    base_iterations: int = 6,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "standard",
    page_factor: float = 1.5,
    max_iterations: int = 40,
) -> FlatBatchProgram:
    vlm_config = LMConfig(
        model=vlm["model"],
        api_base=vlm.get("api_base"),
        api_key=vlm.get("api_key"),
        max_tokens=vlm.get("max_tokens", 65536),
        temperature=vlm.get("temperature", 1.0),
        vertex_location=vlm.get("vertex_location"),
    ) if vlm and vlm.get("model") else LMConfig()

    vlm_lm = vlm_config.to_dspy_lm()

    return FlatBatchProgram(
        vlm_lm=vlm_lm,
        iterations_per_question=iterations_per_question,
        base_iterations=base_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        max_iterations=max_iterations,
    )
