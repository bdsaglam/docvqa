"""Flat parallel solver — batch_look is the sole vision tool.

All VLM queries go through batch_look, which runs them in parallel.
Single queries are just batch_look with one element. This encourages
the agent to think in batches: overlapping sweeps, grid scans,
redundant reads with majority voting, multi-scale crops — all in
one parallel call.
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
    "- `page_texts`: OCR-extracted text per page. May be inaccurate — verify critical values visually.\n"
    "- `pages`: list of page images (PIL Images) (0-indexed). Crop with `pages[i].crop((l,t,r,b))`.\n\n"

    "## TOOLS\n"
    "- search(query, k=5) -> list[dict]: BM25 search over OCR text. Returns [{page, score, text}]. "
    "Useful for multi-page documents to locate relevant pages. For single-page docs, read `page_texts` directly.\n"
    "- batch_look(requests) -> list[str]: Send images to the VLM in parallel. "
    "Input: list of (image, query) tuples. Returns: list of answers in same order. "
    "All queries run simultaneously — use large batches (8-16 queries is efficient). "
    "Works with any PIL Image: full pages, crops, processed images.\n\n"

    "## PARALLEL VISION STRATEGIES\n"
    "batch_look runs all queries simultaneously. Design batches that maximize information per call.\n\n"

    "**Strategies** (combine freely):\n"
    "- **Overlapping sweep**: Split a table into overlapping horizontal strips and read each in parallel. "
    "Stitch results in code. Overlap ensures no row falls on a boundary.\n"
    "- **Grid scan**: Split a page into a grid (e.g. 3x3) and batch_look all 9 crops "
    "with 'what labels/landmarks are here?'. Build a spatial map in code.\n"
    "- **Multi-scale**: Same query at different crop sizes — full page for context, tight crop for precision.\n"
    "- **Multi-phrasing**: Same region, different question wordings — catches VLM blind spots.\n"
    "- **Redundant reads**: For critical values, send 2-3 identical (image, query) pairs and majority-vote.\n"
    "- **Cross-region verification**: Read a value from the table AND from a chart showing the same data.\n"
    "After receiving results, analyze in Python: `Counter(results).most_common(1)`, detect outliers, "
    "stitch strips, or cross-validate. Trust consensus, not any single read.\n\n"

    "## APPROACH\n"
    "1. EXPLORE: Read `page_texts`, then batch_look full pages to understand layout — "
    "e.g. `batch_look([(pages[i], 'Describe the layout: sections, tables, figures, labels, positions') for i in range(len(pages))])`.\n"
    "2. PLAN: Group questions by region. Design batch queries that serve multiple questions at once.\n"
    "3. SOLVE SEQUENTIALLY: Focus on one question (or a small group of related questions) at a time. "
    "For each, design a batch of parallel queries — sweeps, multi-scale, redundant reads. "
    "Send a LARGE batch at once. Analyze results in Python — find consensus, flag disagreements. "
    "Only do another batch if results are truly ambiguous. "
    "Knowledge you gain while solving one question carries over — use it for subsequent questions. "
    "Do NOT try to answer all questions in a single step.\n"
    "4. SUBMIT: Once all questions are answered, SUBMIT all answers together as a dict.\n\n"

    "## GUIDELINES\n"
    "- Full-page batch_look gives a broad overview. For fine details, CROP FIRST: "
    "`pages[i].crop((l,t,r,b))`.\n"
    "- Use `pages[i].size` to get dimensions for cropping.\n"
    "- Ask the VLM ONE simple factual question per query. Extract raw facts, then compute in Python.\n"
    "- A single VLM read can be noisy. For critical values, send 2-3 redundant reads and majority-vote.\n"
    "- CONFLICT RESOLUTION: When batch reads disagree, give more weight to the TIGHTER crop. "
    "Never blindly adopt a number — compare across reads and take the consensus.\n"
    "- SUPERLATIVES: For 'largest', 'first', 'last', 'only' questions — enumerate ALL candidates first, "
    "then select programmatically. Do NOT stop at the first match.\n"
    "- UNKNOWN RULES: Answer 'Unknown' when:\n"
    "  (a) A specific named entity (column name, layer number, variable) does not exist after thorough search.\n"
    "  (b) A chart/table explicitly shows N/A or missing data for the requested item.\n"
    "  (c) Multiple independent reads all fail to find the requested information — that IS the signal.\n"
    "  Do NOT substitute a similar-sounding entity or extrapolate from nearby data.\n"
    "  Do NOT use narrative/descriptive text when a chart explicitly shows N/A.\n"
    "- COMPUTATION: When a question says 'total' or 'considering X and Y', it may require arithmetic. "
    "Extract all referenced values and compute explicitly in Python.\n"
    "- Be efficient. Reuse observations across questions. Aim for large, well-designed batches.\n"
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
    return [t.strip() or "[No text extracted - use batch_look() for visual content]" for t in page_texts]


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


def _create_tools(vlm_predict: dspy.Predict, vlm_lm: dspy.LM, ctx: RunContext) -> list:
    from PIL import Image as PILImage

    def _look_one(image_path: str, query: str) -> str:
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
            return idx, _look_one(path, query)

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

    return [_batch_look_impl, _search]


def _build_sandbox_code(page_dir: str, num_pages: int) -> str:
    """Build sandbox code that loads pages as PIL Images and defines batch_look()."""
    return f'''
import os
import tempfile
from PIL import Image
from collections import Counter

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
    """Send images to the VLM in parallel. Returns list of str answers (same order).
    Input: list of (image, query) tuples. All queries run simultaneously.

    Example — survey all pages:
        results = batch_look([(pages[i], 'Describe layout: sections, tables, figures') for i in range(len(pages))])

    Example — read a value with redundant verification:
        crop = pages[0].crop((100, 200, 400, 350))
        results = batch_look([(crop, "What number is shown?")] * 3)
        answer = Counter(results).most_common(1)[0][0]

    Example — overlapping strip sweep of a table:
        w, h = pages[0].size
        strip_h = h // 4
        overlap = strip_h // 4
        strips = [(pages[0].crop((0, y, w, min(y + strip_h, h))),
                    "Read all rows in this table section")
                   for y in range(0, h - strip_h + 1, strip_h - overlap)]
        results = batch_look(strips)

    Example — grid scan to locate landmarks:
        w, h = pages[0].size
        results = batch_look([
            (pages[0].crop((c*w//3, r*h//3, (c+1)*w//3, (r+1)*h//3)),
             "What labels or landmarks are visible?")
            for r in range(3) for c in range(3)
        ])

    Example — multi-scale read:
        w, h = pages[0].size
        results = batch_look([
            (pages[0], "What is the title of the chart?"),
            (pages[0].crop((0, 0, w//2, h//3)), "What is the title of the chart?"),
        ])
    """
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
# FlatParallelProgram
# ---------------------------------------------------------------------------

class FlatParallelProgram:
    """Flat parallel solver — batch_look is the sole vision tool."""

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
        """Solve all questions for a document using batch_look as the sole vision tool."""
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
                "Flat parallel (%s) limits for %d questions, %d pages: max_iterations=%d (page_bonus=%d)",
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
                "solve_flat_parallel",
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
                    logger.warning("Flat parallel RLM failed for doc '%s': %s", document.doc_id, e)
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
                            "flat_parallel_question_result",
                            question_id=q.question_id,
                            question=q.question[:200],
                            is_correct=is_correct,
                            prediction=answer[:200],
                            ground_truth=q.answer[:200],
                            extracted_answer=extracted[:200],
                        ):
                            pass
                        logger.info(
                            "FlatPar Q %s: %s (GT=%s, PRED=%s)",
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

def create_flat_parallel_program(
    iterations_per_question: int = 4,
    base_iterations: int = 6,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "standard",
    page_factor: float = 1.5,
    max_iterations: int = 40,
) -> FlatParallelProgram:
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

    return FlatParallelProgram(
        vlm_lm=vlm_lm,
        iterations_per_question=iterations_per_question,
        base_iterations=base_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        max_iterations=max_iterations,
    )
