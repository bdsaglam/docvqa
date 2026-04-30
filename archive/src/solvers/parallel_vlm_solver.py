"""Parallel VLM solver — batch_look only, no single look().

Key insight: local VLM (Qwen) is fast but noisy per-call. Instead of
the agent burning iterations on sequential verification, flood the VLM
with parallel reads and let the agent reason over variance in code.

Only tool is `batch_look` — forces the agent to think in batches.
The agent can send the same query with different crops/scales, or the
same crop with different phrasings, to get multiple independent readings.
Majority vote, outlier detection, and confidence estimation happen in
Python code, not via sequential re-querying.
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
from docvqa.prompts import ANSWER_FORMATTING_RULES
from docvqa.code_rlm import CodeRLM
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
    "- batch_look(requests) -> list[str]: Send multiple images to the VLM IN PARALLEL. "
    "Input: list of (image, query) tuples. Returns: list of answers in same order. "
    "Always batch your queries — even a single query goes through batch_look.\n\n"

    "## KEY PRINCIPLE: PARALLEL READS FOR RELIABILITY\n"
    "The VLM is fast but noisy — the same query on the same crop can return different answers. "
    "Instead of re-querying sequentially, exploit parallelism. Think of `batch_look` as a sensor array "
    "you can point at the document however you want — all queries run simultaneously.\n\n"

    "**Strategies** (combine freely):\n"
    "- **Overlapping sweep**: To read a table, split it into overlapping horizontal strips and read each strip "
    "in parallel. Stitch results in code. Overlap ensures no row falls on a boundary.\n"
    "- **Grid scan**: To find something on a page, split into a grid (e.g. 3x3) and batch_look all 9 crops "
    "with 'what labels/landmarks are here?'. Build a spatial map in code.\n"
    "- **Multi-scale**: Same query at different crop sizes — full page for context, tight crop for precision.\n"
    "- **Multi-phrasing**: Same region, different question wordings — catches VLM blind spots.\n"
    "- **Redundant reads**: For critical values, send 2-3 identical pairs and majority-vote.\n"
    "- **Cross-region verification**: Read a value from the table AND from a chart that shows the same data.\n"
    "After receiving results, analyze in Python: `Counter(results).most_common(1)`, detect outliers, "
    "stitch strips, or cross-validate. Trust consensus, not any single read.\n\n"

    "## APPROACH\n"
    "1. EXPLORE: Use targeted tools to locate relevant content: `search()` for keyword lookup across pages, "
    "Python `re` for pattern matching (section headers, citation numbers), "
    "and `batch_look` on pages for semantic understanding. "
    "Page texts can be very long — only print short, relevant excerpts after you've pinpointed the section you need.\n"
    "2. PLAN: Group questions by region. Design batch queries that serve multiple questions at once.\n"
    "3. SOLVE: For each question, design a batch of parallel queries — sweeps, multi-scale, redundant reads. "
    "Send a LARGE batch at once (8-16 queries is fine). Analyze results in Python — find consensus, "
    "flag disagreements. Only do another batch if results are truly ambiguous.\n"
    "4. SUBMIT: Once all questions are answered, SUBMIT all answers together.\n\n"

    "## GUIDELINES\n"
    "- Full-page batch_look gives a broad overview. For fine details, CROP FIRST: `pages[i].crop((l,t,r,b))`.\n"
    "- Use `pages[i].size` to get dimensions for cropping.\n"
    "- Ask the VLM ONE simple factual question per query. Extract raw facts, then compute in Python.\n"
    "- A single VLM read is UNRELIABLE. Always design for redundancy — send 2-3 reads per critical value.\n"
    "- CONFLICT RESOLUTION: When batch reads disagree, give more weight to the TIGHTER crop. "
    "Never blindly adopt a number — compare across reads and take the consensus.\n"
    "- For tables: sweep overlapping strips from top to bottom, read each strip's rows.\n"
    "- For spatial questions: grid-scan to locate landmarks, then compute relationships in code.\n"
    "- SUPERLATIVES: For 'largest', 'first', 'last', 'only' — enumerate ALL candidates, don't stop at first match.\n"
    "- UNKNOWN RULES: Redundancy helps find answers, but do NOT let it prevent you from answering 'Unknown'. "
    "If a specific entity genuinely does not exist, more reads won't find it. Answer 'Unknown' when:\n"
    "  (a) A specific named entity (column name, layer number, variable) does not exist after thorough search.\n"
    "  (b) A chart/table explicitly shows N/A or missing data for the requested item.\n"
    "  (c) Multiple independent reads all fail to find the requested information — that IS the signal.\n"
    "  Do NOT substitute a similar-sounding entity or extrapolate from nearby data.\n"
    "  Do NOT use narrative/descriptive text when a chart explicitly shows N/A.\n"
    "- COMPUTATION: When a question says 'total' or 'considering X and Y', it may require arithmetic. "
    "Extract all referenced values and compute explicitly in Python.\n"
    "- Be efficient. Reuse observations. Aim for large, well-designed batches over many small ones.\n"
    "- NEVER use outside/world knowledge. ALL answers MUST come from the document.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a dict mapping each question_id to its answer string.\n"
    '- Example: SUBMIT(answers={"q1": "42", "q2": "Tokyo"})\n'
    "- Each answer must follow these formatting rules:\n\n"

    + ANSWER_FORMATTING_RULES
)


# ---------------------------------------------------------------------------
# Category-specific tips (adapted for batch_look-only workflow)
# ---------------------------------------------------------------------------

CATEGORY_TIPS: dict[str, str] = {
    "engineering_drawing": (
        "- PRECISION IS CRITICAL: Always crop tables and labels at full resolution. Use batch_look to read "
        "multiple BOM sections in parallel — overlapping strips ensure no row is missed.\n"
        "- The BOM/parts list uses ITEM NUMBERS (e.g., 71, 164) and PART/IDENTIFYING NUMBERS (e.g., 1901060-021, AN 910-2). "
        "Questions asking for 'identifying number' or 'part number' want the PART NUMBER, not the item number.\n"
        "- 'VIEW IN DIRECTION X' labels indicate viewing angles. Answer with just the letter (e.g., 'D'), not 'Direction D'.\n"
        "- For counting parts (e.g., clamps), batch_look the BOM table in overlapping strips at full resolution. "
        "Sum the QTY column in code — don't estimate from visual inspection.\n"
        "- VLM OCR CONFUSION: Part numbers are almost always numeric (digits 0-9 plus dashes). "
        "Send redundant reads of the same crop and majority-vote to resolve ambiguities. "
        "Common confusions: I↔1, O↔0, l↔1.\n"
        "- DIMENSIONS: 'Width' typically refers to the shorter cross-sectional dimension (from a Section view), "
        "not the longest overall dimension (which is 'length'). Dimensions marked 'REF' are valid answers.\n"
    ),
    "business_report": (
        "- Crop tables at full resolution before reading any numbers or labels.\n"
        "- Multiple tables may contain similar-looking data. Verify you're reading the correct table for the question.\n"
        "- For YoY calculations, extract raw values from the table first, then compute differences in Python. "
        "Do not rely on the VLM for arithmetic.\n"
        "- CHART VALUES: Send 2-3 redundant reads of the same chart crop and majority-vote. "
        "VLM readings of chart values vary — trust consensus, not any single read.\n"
        "- 'Broken down into' means immediate sub-categories only, not sub-sub-categories.\n"
        "- TEXT EXTRACTION: When a question asks for 'first words up to the first comma', read the full bullet point text "
        "yourself and extract in Python. Do not ask the VLM to truncate — it over-shortens.\n"
        "- PICTOGRAMS: When looking for a described pictogram, batch_look each individual icon with 'describe this icon' "
        "rather than asking a yes/no filtering question across all icons.\n"
    ),
    "comics": (
        "- OCR is unreliable for comics — use batch_look to read speech bubbles and panel actions, not OCR text.\n"
        "- STORY MAP FIRST: For multi-story anthologies, batch_look all pages with 'What is the story title on this page? "
        "List panel actions and speech bubble text.' to build a story index. Match question keywords to the correct story.\n"
        "- For COUNTING EVENTS (e.g., 'how many times X happens'): batch_look all pages with HIGHLY SPECIFIC queries. "
        "Ask 'Is someone physically [exact action] in this panel? Exclude mentions of past events, near-misses, and aftermath.' "
        "Then count matches in code. Use strict inclusion criteria to avoid over-counting.\n"
        "- VERIFY COUNTS: The VLM hallucinates actions in busy panels — it infers events from context clues "
        "even when no action is depicted. After collecting candidates, RE-EXAMINE each: "
        "crop the specific panel tightly and batch_look with a disconfirming question "
        "('Did this action ACTUALLY occur, or is it a near-miss/different action/aftermath?'). "
        "Expect many initial candidates to be false positives.\n"
        "- PANEL-BY-PANEL: batch_look panels with 'what happens in each panel?' — not generic page descriptions.\n"
        "- LITERAL VS FIGURATIVE: When a question says 'in reality', 'actually', or 'truly', the answer likely "
        "contradicts the surface description. Read carefully for the distinction between a title/alias and the factual answer.\n"
        "- CHARACTER IDENTIFICATION: Answer with the exact term used in speech bubbles. "
        "Use narrative context to disambiguate when batch reads disagree about a small object.\n"
    ),
    "maps": (
        "- COARSE-TO-FINE: First batch_look the full page with 'Describe layout and landmarks', then batch_look "
        "medium crops (~800px) of areas of interest, then tight crops (~400px) for small text.\n"
        "- GRID SCAN: Split the map into tiles (e.g. 3x3), batch_look all tiles with 'What labels are visible here? "
        "Where in this tile is [landmark]?', then stitch positions in code using tile offsets.\n"
        "- REASON WITH MATH: Compute spatial relationships in Python using collected positions. "
        "Basic vector math gives reliable answers and error bounds.\n"
        "- LEGEND + ROAD TYPES: Crop the legend alongside the specific road segment at HIGH resolution. "
        "Batch_look both crops and compare line styles in code.\n"
        "- GRID COORDINATES: Batch_look the actual grid cell AND search text for entries with that grid coordinate. Cross-reference.\n"
    ),
    "science_paper": (
        "- Papers have many pages — use search() and page_texts to locate relevant sections first.\n"
        "- CITATION NUMBERS: Never ask the VLM 'what is the first/last citation on this page'. Instead: "
        "(1) Use page_texts to find all [N] patterns with Python regex, ordered by position. "
        "(2) For ambiguous cases, batch_look the specific paragraphs at full resolution to verify.\n"
        "- CITED PAPER FINDINGS: To find what a cited work claims, first find its reference number "
        "in the bibliography, then search body text for that number to find where it's discussed.\n"
        "- ABLATION STUDIES: Papers often have multiple ablation studies. Verify the section is about "
        "the specific component the question asks about, not a different subsystem.\n"
        "- If a question references a specific entity not found after thorough search, answer 'Unknown'. "
        "Do not extrapolate from similar entities.\n"
    ),
    "science_poster": (
        "- Posters are dense single-page documents. Batch_look specific sections for precise values.\n"
        "- CHART ANNOTATIONS: If a chart has percentage labels on bars/lines, "
        "read those labels rather than computing from raw bar heights.\n"
        "- For table values and percentages, batch_look the specific table cells at full resolution with redundant reads.\n"
        "- 'Percentage improvement' = absolute difference in percentage points (e.g., 80% - 50% = 30%).\n"
        "- COLOR-CODED VALUES: Batch_look individual table cells at MAXIMUM resolution. "
        "Enumerate all candidates of that color before selecting.\n"
        "- GROUPED BAR CHARTS: A 'set of columns' refers to the group of bars at one x-axis position "
        "(e.g., one benchmark), not bars of one color across all positions.\n"
    ),
    "infographics": (
        "- Infographics mix text, icons, and illustrations — batch_look the full page for context.\n"
        "- OCR often describes images rather than reading text — use batch_look to read actual text.\n"
        "- For precise numbers or dates, crop the specific data point. For identifying visual elements, full-page batch_look is fine.\n"
        "- SYSTEMATIC ENUMERATION: When a question asks 'which item is the last/first to have/lack X', "
        "enumerate ALL items and their X status in a list before answering. Don't stop after finding a few.\n"
    ),
    "slide": (
        "- Slides span many pages. Use search() and page_texts to find the relevant slide first.\n"
        "- PAGE NAVIGATION: When a question refers to 'the page before X' or 'the page where Y is mentioned', "
        "first locate X/Y by searching page_texts, then verify by cropping the page header/title. "
        "Off-by-one errors are common — double-check page indices.\n"
        "- For 'last word on page X', batch_look the bottom portion of that page.\n"
        "- Tables in slides may be small — crop at full resolution and use redundant reads.\n"
        "- EXACT ENTITY MATCHING: If a question references a specific column name, variable, or equation "
        "that does not exist after thorough search, answer 'Unknown'. Do NOT substitute a similar-sounding name.\n"
        "- COMPUTATION: When a question says 'total' or 'considering X and Y', extract all referenced values "
        "and compute explicitly in Python. Show the calculation before submitting.\n"
    ),
}


def _get_category_tips(category: str) -> str:
    tips = CATEGORY_TIPS.get(category, "")
    if tips:
        return f"\n## CATEGORY-SPECIFIC TIPS ({category})\n{tips}"
    return ""


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

        # High concurrency for local models, lower for Vertex AI
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

    # _batch_look_impl must be in tools list so sandbox can call it, but we don't want the LLM to call it directly.
    # The sandbox wrapper batch_look() handles image saving; direct _batch_look_impl calls fail.
    return [_batch_look_impl, _search]


def _build_sandbox_code(page_dir: str, num_pages: int) -> str:
    """Build sandbox code — only batch_look, no single look()."""
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
    """Send multiple images to the VLM in parallel. This is your ONLY vision tool.
    Input: list of (image, query) tuples. Returns: list of str answers (same order).
    All queries run simultaneously — use LARGE batches (8-16 is fine).

    Example — overlapping strip sweep of a table:
        w, h = pages[0].size
        strip_h = h // 4
        overlap = strip_h // 4
        strips = []
        for y in range(0, h - strip_h + 1, strip_h - overlap):
            strips.append(pages[0].crop((0, y, w, min(y + strip_h, h))))
        results = batch_look([(s, "Read all rows in this table section") for s in strips])

    Example — grid scan to locate landmarks:
        w, h = pages[0].size
        results = batch_look([
            (pages[0].crop((c*w//3, r*h//3, (c+1)*w//3, (r+1)*h//3)),
             "What labels or landmarks are visible?")
            for r in range(3) for c in range(3)
        ])

    Example — redundant reads with majority vote:
        from collections import Counter
        crop = pages[0].crop((100, 200, 400, 350))
        results = batch_look([(crop, "Read the number")] * 3)
        answer = Counter(results).most_common(1)[0][0]
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
# Answer parsing (same as flat_batch)
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
# ParallelVLMProgram
# ---------------------------------------------------------------------------

class ParallelVLMProgram:
    """Parallel VLM solver — batch_look only, flood-and-analyze pattern."""

    def __init__(
        self,
        vlm_lm: dspy.LM,
        iterations_per_question: int = 4,
        base_iterations: int = 6,
        page_factor: float = 0.5,
        max_iterations: int = 30,
    ):
        self.vlm_lm = vlm_lm
        self.iterations_per_question = iterations_per_question
        self.base_iterations = base_iterations
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
        """Solve all questions for a document."""
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

            num_q = len(document.questions)
            num_pages = len(document.images)
            page_bonus = self.page_factor * math.ceil(math.sqrt(max(0, num_pages - 9)))
            max_iter = min(
                self.base_iterations + self.iterations_per_question * num_q + int(page_bonus),
                self.max_iterations,
            )

            tips = _get_category_tips(document.doc_category)
            instructions = TASK_INSTRUCTIONS + ("\n" + tips if tips else "")
            tools = _create_tools(self.vlm_predict, self.vlm_lm, ctx)

            rlm = CodeRLM(
                signature=_build_signature(instructions),
                max_iterations=max_iter,
                max_llm_calls=max_iter * 3,
                tools=tools,
                verbose=True,
                sandbox_code=_build_sandbox_code(tmpdir, len(document.images)),
            )
            logger.info(
                "Parallel VLM limits for %d questions, %d pages: max_iterations=%d (page_bonus=%d)",
                num_q, num_pages, max_iter, int(page_bonus),
            )

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
                    "Rate limit, retry %d in %.0fs", rs.attempt_number, rs.next_action.sleep
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
                "solve_parallel_vlm",
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
                    logger.warning("Parallel VLM RLM failed for doc '%s': %s", document.doc_id, e)
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
                            "parallel_vlm_question_result",
                            question_id=q.question_id,
                            question=q.question[:200],
                            is_correct=is_correct,
                            prediction=answer[:200],
                            ground_truth=q.answer[:200],
                            extracted_answer=extracted[:200],
                        ):
                            pass
                        logger.info(
                            "ParVLM Q %s: %s (GT=%s, PRED=%s)",
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

def create_parallel_vlm_program(
    iterations_per_question: int = 4,
    base_iterations: int = 6,
    page_factor: float = 0.5,
    max_iterations: int = 30,
    vlm: dict[str, Any] | None = None,
) -> ParallelVLMProgram:
    vlm_config = LMConfig(
        model=vlm["model"],
        api_base=vlm.get("api_base"),
        api_key=vlm.get("api_key"),
        max_tokens=vlm.get("max_tokens", 65536),
        temperature=vlm.get("temperature", 1.0),
        vertex_location=vlm.get("vertex_location"),
    ) if vlm and vlm.get("model") else LMConfig()

    vlm_lm = vlm_config.to_dspy_lm()

    return ParallelVLMProgram(
        vlm_lm=vlm_lm,
        iterations_per_question=iterations_per_question,
        base_iterations=base_iterations,
        page_factor=page_factor,
        max_iterations=max_iterations,
    )
