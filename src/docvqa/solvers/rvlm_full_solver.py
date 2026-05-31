"""RVLM-full solver — kitchen-sink: rvlm + look() ergonomic + OCR.

Tool surface = ``look(image, query)`` single-image wrapper + ``batch_look``
(parallel) + ``search`` + ``page_texts``. Distinct from
:mod:`docvqa.solvers.rvlm_ocr_solver` only in the addition of the
single-image ``look()`` ergonomic wrapper — per D-006, that ergonomic
extra is *confounding* if you want to attribute lift to OCR alone.

Engineering name per D-010 (paper-facing name TBD). This solver retains
the D-004 page-only cropping ablation (``vlm_cropping=False``) and the
search-on/off ablation (``use_search=False``).

Per D-007/D-009 (docs/paper/decisions.md), this solver uses a MINIMAL
prompt for parity with ``rvlm_minimal``:
- This solver owns its tool-surface documentation (``TASK_INSTRUCTIONS``).
- No per-category tips and no per-category tool-routing overlay: an n=8
  ablation showed the per-category content was not load-bearing, so it
  was stripped to reach prompt parity with the proposed method.
- ``ANSWER_FORMATTING_RULES`` is read from
  ``profile.answer_formatting_rules`` — not imported from
  :mod:`docvqa.prompts`.
"""

from __future__ import annotations

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

from docvqa.data import Document, Question
from docvqa.datasets.profile import DatasetProfile, get_profile
from docvqa.rlm import LeanRLM, CodeRLM, ThinkingRLM, RLM
from docvqa.search import get_or_build_index
from docvqa.types import LMConfig
from docvqa.retry_utils import is_retryable_lm_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (formatting rules are substituted from the profile)
# ---------------------------------------------------------------------------

_CROPPING_BODY = (
    "You are a Document Visual Question Answering agent. You answer a question about a document by "
    "writing Python code, calling vision tools iteratively, and reasoning programmatically.\n\n"

    "## DATA\n"
    "- `question`: The question you must answer.\n"
    "- `page_texts`: OCR-extracted text per page. May be inaccurate — verify critical values visually.\n"
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
    "Much faster than sequential look() calls — use it for efficiently processing multiple images or queries or cross-checks.\n"

    "## APPROACH\n"
    "1. EXPLORE: Before answering, understand the document structure. "
    "Read `page_texts`, then use `look` to survey pages — "
    "'Describe the layout: what sections, tables, figures, and labels are present and where are they positioned?' "
    "Build a mental map of the document.\n"
    "2. LOCATE: Find the specific region(s) relevant to the question.\n"
    "3. EXTRACT: Use `look` with tight crops to read exact values. "
    "For fine details, crop first: `look(pages[i].crop((l,t,r,b)), query)`.\n"
    "4. VERIFY: Cross-check extracted values if ambiguous.\n"
    "5. SUBMIT: Once you have the answer, SUBMIT it.\n\n"

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
    "- COMPUTATION: When a question says 'total' or 'considering X and Y', it may require arithmetic. "
    "Extract all referenced values and compute explicitly in Python.\n"
    "- NEVER use outside/world knowledge. ALL answers MUST come from the document.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string.\n"
    '- Example: SUBMIT(answer="42")\n'
    "- The answer must follow these formatting rules:\n\n"
)

_PAGE_ONLY_BODY = (
    "You are a Document Visual Question Answering agent. You answer a question about a document by "
    "writing Python code, calling vision tools iteratively, and reasoning programmatically.\n\n"

    "## DATA\n"
    "- `question`: The question you must answer.\n"
    "- `page_texts`: OCR-extracted text per page. May be inaccurate — verify critical values visually.\n"
    "- `num_pages`: total number of pages (0-indexed).\n\n"

    "## TOOLS\n"
    "- search(query, k=5) -> list[dict]: BM25 search over OCR text. Returns [{page, score, text}]. "
    "Useful for multi-page documents to locate relevant pages. For single-page docs, read `page_texts` directly.\n"
    "- look(page_idx, query) -> str: "
    "Send the page at index `page_idx` (int, 0-indexed) to the VLM with a query. "
    "Whole pages only — no cropping is available.\n"
    "- batch_look(requests) -> list[str]: Parallel VLM calls. "
    "Input: list of (page_idx, query) tuples. Returns: list of answers in same order. "
    "Much faster than sequential look() calls — use it for efficiently processing multiple pages or queries.\n"

    "## APPROACH\n"
    "1. EXPLORE: Read `page_texts`, then use `look(page_idx, ...)` to survey pages and build a mental map.\n"
    "2. LOCATE: Identify the page(s) relevant to the question.\n"
    "3. EXTRACT: Re-look at relevant pages with targeted queries to read exact values.\n"
    "4. VERIFY: Cross-check extracted values by re-querying the same page if ambiguous.\n"
    "5. SUBMIT: Once you have the answer, SUBMIT it.\n\n"

    "## GUIDELINES\n"
    "- Ask the VLM ONE simple factual question per call. Do NOT combine multiple questions or ask it to reason. "
    "Extract raw facts, then count/compare/compute in Python.\n"
    "- VLM CONFLICT RESOLUTION: When readings conflict on the same page, do ONE tie-breaking read with a "
    "more specific question. Never silently adopt a new number from a 'verification' pass.\n"
    "- SUPERLATIVES: For 'largest', 'first', 'last', 'only' questions — enumerate ALL candidates first, "
    "then select programmatically. Do NOT stop at the first match.\n"
    "- COMPUTATION: When a question says 'total' or 'considering X and Y', it may require arithmetic. "
    "Extract all referenced values and compute explicitly in Python.\n"
    "- NEVER use outside/world knowledge. ALL answers MUST come from the document.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string.\n"
    '- Example: SUBMIT(answer="42")\n'
    "- The answer must follow these formatting rules:\n\n"
)

def _build_task_instructions(profile: DatasetProfile, vlm_cropping: bool) -> str:
    body = _CROPPING_BODY if vlm_cropping else _PAGE_ONLY_BODY
    return body + profile.answer_formatting_rules

# Back-compat export for shelved solvers (flat_solo_gepa) that imported
# ``TASK_INSTRUCTIONS`` from the old flat_solo_solver. They want the default
# (DocVQA-2026 + cropping) prompt seed for GEPA optimization.
TASK_INSTRUCTIONS = _build_task_instructions(get_profile("VLR-CVC/DocVQA-2026"), vlm_cropping=True)

# ---------------------------------------------------------------------------
# Helpers (inlined from the former flat_solo_solver — Phase 2D merge)
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    page_dir: str
    num_pages: int
    search_index: Any = None
    page_texts: list[str] | None = None

def _format_page_texts(page_texts: list[str]) -> list[str]:
    return [t.strip() or "[No text extracted - use look() for visual content]" for t in page_texts]

def _build_signature(instructions: str) -> dspy.Signature:
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

def _strip_search_tool(instructions: str) -> str:
    """Remove the search() tool line for the use_search=False ablation."""
    return re.sub(r'- search\(query, k=5\)[^\n]*\n', '', instructions)

def _create_tools(vlm_predict: dspy.Predict, vlm_lm: dspy.LM, ctx: RunContext, *, use_search: bool = True) -> list:
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

    tools = [_look_impl, _batch_look_impl]
    if use_search:
        tools.append(_search)
    return tools

def _build_sandbox_code(page_dir: str, num_pages: int, use_search: bool = True) -> str:
    """Build sandbox code that loads pages as PIL Images and defines `look()`."""
    search_def = '''
def search(query, k=5):
    """BM25 search over OCR text. Returns list of {page, score, text} dicts."""
    return _search(query, k)
''' if use_search else ''
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
{search_def}
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

def _build_sandbox_code_page_only(page_dir: str, num_pages: int, use_search: bool = True) -> str:
    """Page-only sandbox for the D-004 cropping ablation. `look(page_idx, query)`
    accepts only an integer page index — no PIL Images, no crops."""
    search_def = '''
def search(query, k=5):
    """BM25 search over OCR text. Returns list of {page, score, text} dicts."""
    return _search(query, k)
''' if use_search else ''
    return f'''
import os
num_pages = {num_pages}
_page_paths = [os.path.join({page_dir!r}, f"page_{{i}}.png") for i in range({num_pages})]
for _p in _page_paths:
    assert os.path.exists(_p), f"Page image not found: {{_p}}"

def look(page_idx, query):
    """Send page `page_idx` (int, 0-indexed) to the VLM with a query.
    No cropping is available — whole pages only."""
    if not isinstance(page_idx, int):
        raise TypeError(f"look() expects an int page index, got {{type(page_idx).__name__}}")
    if not (0 <= page_idx < num_pages):
        raise IndexError(f"page_idx {{page_idx}} out of range [0, {{num_pages}})")
    return _look_impl(_page_paths[page_idx], query)
{search_def}
def batch_look(requests):
    """Send multiple pages to the VLM in parallel. Whole pages only.
    Input: list of (page_idx, query) tuples. Returns: list of str answers (same order)."""
    import json as _json
    paths = []
    for page_idx, query in requests:
        if not isinstance(page_idx, int):
            raise TypeError(f"batch_look() expects int page indices, got {{type(page_idx).__name__}}")
        if not (0 <= page_idx < num_pages):
            raise IndexError(f"page_idx {{page_idx}} out of range [0, {{num_pages}})")
        paths.append({{"path": _page_paths[page_idx], "query": query}})
    return _batch_look_impl(_json.dumps(paths))
'''

# ---------------------------------------------------------------------------
# RvlmFullProgram
# ---------------------------------------------------------------------------

class RvlmFullProgram:
    """RVLM-full solver. See module docstring."""

    def __init__(
        self,
        vlm_lm: dspy.LM,
        profile: DatasetProfile,
        max_iterations: int = 20,
        rlm_type: str = "lean",
        page_factor: float = 1.5,
        question_concurrency: int = 1,
        vlm_cropping: bool = True,
        use_search: bool = True,
    ):
        self.vlm_lm = vlm_lm
        self.profile = profile
        self.max_iterations = max_iterations
        self.rlm_type = rlm_type
        self.page_factor = page_factor
        self.question_concurrency = question_concurrency
        self.vlm_cropping = vlm_cropping
        self.use_search = use_search

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

    def _per_question_prefix(self, q: Question) -> str:
        """Optional hint string prepended to the per-question prompt."""
        if self.profile.question_format_hint_fn is None:
            return ""
        hint = self.profile.question_format_hint_fn(q)
        return f"\n{hint}\n" if hint else ""

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

            instructions = _build_task_instructions(self.profile, self.vlm_cropping)
            if not self.use_search:
                instructions = _strip_search_tool(instructions)
            tools = _create_tools(self.vlm_predict, self.vlm_lm, ctx, use_search=self.use_search)
            if self.vlm_cropping:
                sandbox_code = _build_sandbox_code(tmpdir, len(document.images), use_search=self.use_search)
            else:
                sandbox_code = _build_sandbox_code_page_only(tmpdir, len(document.images), use_search=self.use_search)

            def _solve_question(q: Question):
                """Solve a single question. Returns (question_id, answer, trajectory)."""
                with logfire.span(
                    "solve_rvlm_full",
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
                        "RVLM-full [%s] (%s) Q %s: max_iterations=%d (page_bonus=%d)",
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
                        is_correct, extracted = self.profile.score_fn(answer, q.answer, q)
                        q_span.set_attribute("is_correct", is_correct)
                        q_span.set_attribute("ground_truth", q.answer[:200])
                        q_span.set_attribute("extracted_answer", extracted[:200])
                        logger.info(
                            "RVLM-full[%s] Q %s: %s (GT=%s, PRED=%s)",
                            self.profile.name,
                            q.question_id,
                            "CORRECT" if is_correct else "WRONG",
                            q.answer[:40],
                            extracted[:40],
                        )

                    return q.question_id, answer, trajectory

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
                logger.info("RVLM-full: running %d questions with concurrency=%d", len(document.questions), max_w)
                with ThreadPoolExecutor(max_workers=max_w) as pool:
                    futures = {pool.submit(_solve_question, q): q for q in document.questions}
                    for future in as_completed(futures):
                        qid, answer, trajectory = future.result()
                        predictions[qid] = answer
                        trajectories[qid] = trajectory

            correct = 0
            scored = 0
            for q in document.questions:
                if q.answer is not None:
                    scored += 1
                    is_correct, _ = self.profile.score_fn(predictions[q.question_id], q.answer, q)
                    if is_correct:
                        correct += 1
            if scored > 0:
                logger.info(
                    "RVLM-full [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name, document.doc_id, correct, scored,
                    100 * correct / scored,
                )

            return predictions, trajectories

# ---------------------------------------------------------------------------
# Factory for Hydra instantiation
# ---------------------------------------------------------------------------

def create_rvlm_full_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 20,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    vlm_cropping: bool = True,
    use_search: bool = True,
) -> RvlmFullProgram:
    """Hydra factory.

    Profile resolution order:
        1. ``profile_name`` if given — look up by registered name slug.
        2. ``dataset`` if given — look up by HF dataset id.
        3. Default to DocVQA-2026.

    Pass ``solver.dataset=${data.dataset}`` from the top-level config so
    the profile picks itself up automatically per Hydra invocation.
    """
    from docvqa.datasets.profile import _PROFILES  # type: ignore[attr-defined]

    if profile_name is not None:
        # Allow lookup by either the dataset id or the profile.name slug.
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

    return RvlmFullProgram(
        vlm_lm=vlm_lm,
        profile=profile,
        max_iterations=max_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
        vlm_cropping=vlm_cropping,
        use_search=use_search,
    )
