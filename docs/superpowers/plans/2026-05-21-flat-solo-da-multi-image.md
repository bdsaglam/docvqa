# Flat Solo DA Multi-Image Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sibling of `flat_solo_da_solver.py` where `look()` and `batch_look()` accept either a single image/page-index OR a list — letting the VLM see multiple images in one call.

**Architecture:** New module `src/docvqa/solvers/flat_solo_da_mi_solver.py` cloned from `flat_solo_da_solver.py`. The VLM `dspy.Predict` signature changes from `image: dspy.Image` to `images: list[dspy.Image]`. The two sandbox builders (`cropping`, `page_only`) are forked to expose overloaded tools that normalize singletons to a list internally. Tool prompt text is rewritten to describe the multi-image capability neutrally (no enumerated list of uses). Hydra config `configs/solver/flat_solo_da_mi.yaml` mirrors `flat_solo_da.yaml`.

**Tech Stack:** Python, DSPy, Hydra, vLLM (local Qwen 3.5 27B at `localhost:8927`).

**Testing note:** This codebase has no unit tests for solvers — verification is import-check + Hydra-instantiation + a 2-doc smoke run + a full val eval. The spec explicitly scoped automated tests out.

---

## File Structure

**Create:**
- `src/docvqa/solvers/flat_solo_da_mi_solver.py` — solver module (≈260 lines)
- `configs/solver/flat_solo_da_mi.yaml` — Hydra config (11 lines)

**Modify:** none.

**Imported as-is (not re-implemented):** `RunContext`, `_format_page_texts`, `_build_signature`, `_strip_search_tool` from `docvqa.solvers.flat_solo_solver`.

---

## Task 1: Implement the solver module

**Files:**
- Create: `src/docvqa/solvers/flat_solo_da_mi_solver.py`

- [ ] **Step 1: Create the full module**

Write the file with module docstring, imports, prompt bodies, helpers, tool factory, sandbox builders, class, and factory. Full code:

```python
"""Dataset-aware flat solo solver with multi-image VLM calls.

A sibling of :mod:`docvqa.solvers.flat_solo_da_solver`. The only
substantive difference is that the VLM tool can receive multiple
images in a single call, not just one. ``look(images, query)`` and
``batch_look(requests)`` accept either a single PIL Image (cropping
mode) / page index (page-only mode) or a list — singletons remain
backward compatible.

This unlocks cross-image queries (e.g. comparing two crops, locating
a patch within a larger page, inspecting a sequence) without the
agent having to stitch the answer together from several independent
single-image lookups in Python.

Profile-driven prompt formatting / per-category tips / per-question
hint / scoring are unchanged. Dataset-aware behavior is identical to
``flat_solo_da_solver`` — see that module's docstring.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from typing import Any

import dspy
import logfire
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from docvqa.data import Document, Question
from docvqa.datasets.profile import DatasetProfile, get_profile
from docvqa.rlm import LeanRLM, CodeRLM, ThinkingRLM, RLM
from docvqa.search import get_or_build_index
from docvqa.solvers.flat_solo_solver import (
    RunContext,
    _build_signature,
    _format_page_texts,
    _strip_search_tool,
)
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template (formatting rules are substituted from the profile)
# ---------------------------------------------------------------------------

_CROPPING_BODY_MI = (
    "You are a Document Visual Question Answering agent. You answer a question about a document by "
    "writing Python code, calling vision tools iteratively, and reasoning programmatically.\n\n"

    "## DATA\n"
    "- `question`: The question you must answer.\n"
    "- `page_texts`: OCR-extracted text per page. May be inaccurate — verify critical values visually.\n"
    "- `pages`: list of page images (PIL Images) (0-indexed). Pass to `look()`, e.g. `look(pages[0], 'describe layout')`.\n\n"

    "## TOOLS\n"
    "- search(query, k=5) -> list[dict]: BM25 search over OCR text. Returns [{page, score, text}]. "
    "Useful for multi-page documents to locate relevant pages. For single-page docs, read `page_texts` directly.\n"
    "- look(images, query) -> str: "
    "Send one or more PIL Images to the VLM with a single query. `images` is either a single PIL Image "
    "or a list of PIL Images. When a list is passed, the VLM sees all of them together in one call and "
    "can reason across them (the agent decides what to ask: cross-reference, comparison, locate one "
    "within another, follow a sequence, etc). Each image can be a page (e.g. `pages[0]`), a crop "
    "(e.g. `pages[0].crop((left, top, right, bottom))`), or any processed PIL image. "
    "Full pages are downscaled — for fine details, crop first using PIL.\n"
    "- batch_look(requests) -> list[str]: Parallel VLM calls. "
    "Input: list of (images, query) tuples, where `images` is a single PIL Image or a list. "
    "Returns: list of answers in same order. "
    "Use it to run multiple independent queries (single- or multi-image) in parallel.\n"

    "## APPROACH\n"
    "1. EXPLORE: Before answering, understand the document structure. "
    "Read `page_texts`, then use `look` to survey pages — "
    "'Describe the layout: what sections, tables, figures, and labels are present and where are they positioned?' "
    "Build a mental map of the document.\n"
    "2. LOCATE: Find the specific region(s) relevant to the question.\n"
    "3. EXTRACT: Use `look` with tight crops to read exact values. "
    "For fine details, crop first: `look(pages[i].crop((l,t,r,b)), query)`. "
    "When a question naturally spans multiple regions, pass them together as a list.\n"
    "4. VERIFY: Cross-check extracted values if ambiguous.\n"
    "5. SUBMIT: Once you have the answer, SUBMIT it.\n\n"

    "## GUIDELINES\n"
    "- Full-page `look` gives a broad overview. For fine details, crop first: `look(pages[i].crop((l,t,r,b)), query)`.\n"
    "- Use `pages[i].size` to get dimensions for cropping.\n"
    "- Ask the VLM ONE focused question per call. For single-image calls, extract a raw fact and "
    "count/compare/compute in Python. For multi-image calls, the question should be about the "
    "relationship across the images (e.g. comparison, identification across, sequence) — not "
    "several unrelated facts stacked into one prompt.\n"
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

_PAGE_ONLY_BODY_MI = (
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
    "Send one or more pages to the VLM with a single query. `page_idx` is either an int (0-indexed) "
    "or a list of ints. When a list is passed, the VLM sees all of those pages together in one call "
    "and can reason across them (cross-reference, comparison, follow a sequence, etc). "
    "Whole pages only — no cropping is available.\n"
    "- batch_look(requests) -> list[str]: Parallel VLM calls. "
    "Input: list of (page_idx, query) tuples, where `page_idx` is an int or a list of ints. "
    "Returns: list of answers in same order. "
    "Use it to run multiple independent queries (single- or multi-page) in parallel.\n"

    "## APPROACH\n"
    "1. EXPLORE: Read `page_texts`, then use `look(page_idx, ...)` to survey pages and build a mental map.\n"
    "2. LOCATE: Identify the page(s) relevant to the question.\n"
    "3. EXTRACT: Re-look at relevant pages with targeted queries to read exact values. "
    "When a question naturally spans multiple pages, pass them together as a list.\n"
    "4. VERIFY: Cross-check extracted values by re-querying the same page if ambiguous.\n"
    "5. SUBMIT: Once you have the answer, SUBMIT it.\n\n"

    "## GUIDELINES\n"
    "- Ask the VLM ONE focused question per call. For single-page calls, extract a raw fact and "
    "count/compare/compute in Python. For multi-page calls, the question should be about the "
    "relationship across the pages (e.g. comparison, identification across, sequence) — not "
    "several unrelated facts stacked into one prompt.\n"
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
    body = _CROPPING_BODY_MI if vlm_cropping else _PAGE_ONLY_BODY_MI
    return body + profile.answer_formatting_rules


# ---------------------------------------------------------------------------
# Multi-image VLM tools
# ---------------------------------------------------------------------------


def _create_tools(vlm_predict: dspy.Predict, vlm_lm: dspy.LM, ctx: RunContext, *, use_search: bool = True) -> list:
    from PIL import Image as PILImage

    def _look_impl(image_paths: list[str], query: str) -> str:
        """Internal: load N images from disk and send to VLM as a single multi-image call."""
        with logfire.span("look", num_images=len(image_paths), query=query) as span:
            imgs = [PILImage.open(p) for p in image_paths]
            with dspy.context(lm=vlm_lm):
                result = vlm_predict(
                    images=[dspy.Image(img) for img in imgs],
                    query=query,
                )
                answer = result.answer or ""
                span.set_attribute("answer", answer[:2000])
                return answer

    def _batch_look_impl(requests_json: str) -> list[str]:
        """Internal: batch multi-image VLM calls in parallel.

        Input is JSON list of ``{"paths": [str, ...], "query": str}``.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import json as _json
        requests = _json.loads(requests_json)
        if not requests:
            return []
        results: list[str] = [""] * len(requests)

        def _do(idx: int, paths: list[str], query: str) -> tuple[int, str]:
            return idx, _look_impl(paths, query)

        is_vertex = "vertex_ai" in (vlm_lm.model if hasattr(vlm_lm, 'model') else str(vlm_lm))
        max_w = min(len(requests), 2 if is_vertex else 8)
        with logfire.span("batch_look", num_requests=len(requests)):
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                futures = {
                    pool.submit(_do, i, r["paths"], r["query"]): i
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


# ---------------------------------------------------------------------------
# Sandbox builders (multi-image; singletons accepted via normalization)
# ---------------------------------------------------------------------------


def _build_sandbox_code_mi(page_dir: str, num_pages: int, use_search: bool = True) -> str:
    """Cropping-mode sandbox. `look(images, query)` accepts a PIL Image or list of PIL Images."""
    search_def = '''
def search(query, k=5):
    """BM25 search over OCR text. Returns list of {page, score, text} dicts."""
    return _search(query, k)
''' if use_search else ''
    return f'''
import os
import tempfile
from PIL import Image as _PILImg

Image = _PILImg
Image.MAX_IMAGE_PIXELS = 500_000_000
pages = []
for i in range({num_pages}):
    path = os.path.join({page_dir!r}, f"page_{{i}}.png")
    assert os.path.exists(path), f"Page image not found: {{path}}"
    pages.append(_PILImg.open(path))
assert len(pages) == {num_pages}, f"Expected {{num_pages}} pages, got {{len(pages)}}"

def _normalize_images(images):
    if isinstance(images, _PILImg.Image):
        return [images]
    return list(images)

def _save_images(images):
    paths = []
    for image in images:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp, format="PNG")
        tmp.close()
        paths.append(tmp.name)
    return paths

def look(images, query):
    """Send one or more images to the VLM with a single query.

    `images` is either a single PIL.Image or a list of PIL.Images. With a list,
    the VLM sees them all in one call and can reason across them (compare, locate
    one within another, follow a sequence, etc).
    """
    paths = _save_images(_normalize_images(images))
    return _look_impl(paths, query)
{search_def}
def batch_look(requests):
    """Run multiple VLM calls in parallel.

    Input: list of (images, query) tuples where `images` is a single PIL.Image
    or a list of PIL.Images. Returns: list of str answers (same order).
    """
    import json as _json
    out = []
    for images, query in requests:
        paths = _save_images(_normalize_images(images))
        out.append({{"paths": paths, "query": query}})
    return _batch_look_impl(_json.dumps(out))
'''


def _build_sandbox_code_page_only_mi(page_dir: str, num_pages: int, use_search: bool = True) -> str:
    """Page-only sandbox. `look(page_idx, query)` accepts an int or list of ints — no cropping."""
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

def _normalize_idx(page_idx):
    if isinstance(page_idx, int):
        page_idx = [page_idx]
    paths = []
    for idx in page_idx:
        if not isinstance(idx, int):
            raise TypeError(f"look() expects int page indices, got {{type(idx).__name__}}")
        if not (0 <= idx < num_pages):
            raise IndexError(f"page_idx {{idx}} out of range [0, {{num_pages}})")
        paths.append(_page_paths[idx])
    return paths

def look(page_idx, query):
    """Send one or more pages to the VLM with a single query.

    `page_idx` is an int (0-indexed) or a list of ints. With a list, the VLM
    sees all those pages in one call.
    """
    return _look_impl(_normalize_idx(page_idx), query)
{search_def}
def batch_look(requests):
    """Run multiple VLM calls in parallel.

    Input: list of (page_idx, query) tuples where `page_idx` is an int or list of ints.
    Returns: list of str answers (same order).
    """
    import json as _json
    out = []
    for page_idx, query in requests:
        out.append({{"paths": _normalize_idx(page_idx), "query": query}})
    return _batch_look_impl(_json.dumps(out))
'''


# ---------------------------------------------------------------------------
# FlatSoloDAMIProgram (dataset-aware, multi-image VLM)
# ---------------------------------------------------------------------------


class FlatSoloDAMIProgram:
    """Dataset-aware flat_solo solver with multi-image VLM calls. See module docstring."""

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
                    "images": (
                        list[dspy.Image],
                        dspy.InputField(desc="One or more images (pages or crops) to analyze together"),
                    ),
                    "query": (str, dspy.InputField(desc="What to look for or describe")),
                    "answer": (str, dspy.OutputField(desc="Concise response")),
                },
                "Analyze the image(s) strictly to answer the query. "
                "When multiple images are provided, treat them together (e.g. comparison, "
                "identification across, sequence) and answer with respect to all of them. "
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

            base_instructions = _build_task_instructions(self.profile, self.vlm_cropping)
            if not self.use_search:
                base_instructions = _strip_search_tool(base_instructions)
            tips = self.profile.category_tips_fn(document.doc_category)
            instructions = base_instructions + ("\n" + tips if tips else "")
            tools = _create_tools(self.vlm_predict, self.vlm_lm, ctx, use_search=self.use_search)
            if self.vlm_cropping:
                sandbox_code = _build_sandbox_code_mi(tmpdir, len(document.images), use_search=self.use_search)
            else:
                sandbox_code = _build_sandbox_code_page_only_mi(tmpdir, len(document.images), use_search=self.use_search)

            def _solve_question(q: Question):
                """Solve a single question. Returns (question_id, answer, trajectory)."""
                with logfire.span(
                    "solve_flat_solo_da_mi",
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
                        "Flat solo DA-MI [%s] (%s) Q %s: max_iterations=%d (page_bonus=%d)",
                        self.profile.name, self.rlm_type, q.question_id, max_iter, int(page_bonus),
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
                            "Solo-MI[%s] Q %s: %s (GT=%s, PRED=%s)",
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
                logger.info("Flat solo DA-MI: running %d questions with concurrency=%d", len(document.questions), max_w)
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
                    "Flat solo DA-MI [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name, document.doc_id, correct, scored,
                    100 * correct / scored,
                )

            return predictions, trajectories


# ---------------------------------------------------------------------------
# Factory for Hydra instantiation
# ---------------------------------------------------------------------------


def create_flat_solo_da_mi_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 20,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    vlm_cropping: bool = True,
    use_search: bool = True,
) -> FlatSoloDAMIProgram:
    """Hydra factory. Profile resolution order matches flat_solo_da_solver."""
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

    return FlatSoloDAMIProgram(
        vlm_lm=vlm_lm,
        profile=profile,
        max_iterations=max_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
        vlm_cropping=vlm_cropping,
        use_search=use_search,
    )
```

- [ ] **Step 2: Import-check the module**

```bash
uv run python -c "from docvqa.solvers.flat_solo_da_mi_solver import create_flat_solo_da_mi_program, FlatSoloDAMIProgram; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/docvqa/solvers/flat_solo_da_mi_solver.py
git commit -m "feat: add flat_solo_da_mi_solver (multi-image VLM calls)"
```

---

## Task 2: Hydra config

**Files:**
- Create: `configs/solver/flat_solo_da_mi.yaml`

- [ ] **Step 1: Write the config**

```yaml
_target_: docvqa.solvers.flat_solo_da_mi_solver.create_flat_solo_da_mi_program
# Profile auto-derived from the dataset id; override with profile_name if needed.
dataset: ${data.dataset}
profile_name: null
max_iterations: 30
vlm: ${vlm}
rlm_type: lean
page_factor: 1.5
question_concurrency: 4
vlm_cropping: true
use_search: true
```

- [ ] **Step 2: Verify Hydra can compose with this solver**

```bash
uv run python -c "
from hydra import compose, initialize_config_dir
import os
with initialize_config_dir(config_dir=os.path.abspath('configs'), version_base=None):
    cfg = compose(config_name='config', overrides=['solver=flat_solo_da_mi', 'lm=qwen-3_5-27b-vllm-local', 'vlm=qwen-3_5-27b-vllm-local'])
    print('target:', cfg.solver._target_)
"
```
Expected: `target: docvqa.solvers.flat_solo_da_mi_solver.create_flat_solo_da_mi_program`

- [ ] **Step 3: Commit**

```bash
git add configs/solver/flat_solo_da_mi.yaml
git commit -m "feat: add Hydra config for flat_solo_da_mi solver"
```

---

## Task 3: 2-doc smoke run

**Files:** none modified.

- [ ] **Step 1: Run on 2 val docs (cropping mode)**

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=flat_solo_da_mi solver.rlm_type=lean \
  data.split=val data.num_samples=2 max_concurrency=2 \
  run_id=mi-smoke-crop
```
Expected: completes without error; logfire trace for the doc shows at least one `look` span (with `num_images=1` or higher).

- [ ] **Step 2: Confirm a trajectory references the new tool**

```bash
ls runs/mi-smoke-crop/trajectories/ | head -3
uv run python -c "
import json, glob
for f in sorted(glob.glob('runs/mi-smoke-crop/trajectories/*.json'))[:1]:
    data = json.load(open(f))
    print('doc:', f)
    print('keys:', list(data.keys())[:3])
"
```
Expected: trajectory JSONs exist and contain entries (the trajectory shape per-question varies — just confirm files were produced).

- [ ] **Step 3: Run on 2 val docs (page-only mode)**

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=flat_solo_da_mi solver.rlm_type=lean solver.vlm_cropping=false \
  data.split=val data.num_samples=2 max_concurrency=2 \
  run_id=mi-smoke-pageonly
```
Expected: completes without error.

- [ ] **Step 4: No commit (smoke runs leave no source changes).**

If a smoke run fails: stop, diagnose, fix the solver, re-run. Don't proceed to the full eval.

---

## Task 4: Full val eval (background)

**Files:** none modified.

- [ ] **Step 1: Launch the full val eval in a dedicated tmux session**

```bash
mkdir -p runs
tmux new-session -d -s flat-solo-da-mi-eval -n eval \
  "cd /home/baris/repos/docvqa && \
   uv run python evals.py \
     lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
     solver=flat_solo_da_mi solver.rlm_type=lean \
     data.split=val data.num_samples=null max_concurrency=16 \
     run_id=flat-solo-da-mi-val 2>&1 | tee runs/flat-solo-da-mi-val.log"
```

Expected: `tmux ls` shows `flat-solo-da-mi-eval` session.

- [ ] **Step 2: Schedule heartbeat to check progress**

Schedule a wakeup ~25 min out — Qwen 3.5 27B on the val split typically takes a couple of hours; an early check confirms it's progressing.

- [ ] **Step 3: When done, generate report**

```bash
python scripts/report.py --run-id flat-solo-da-mi-val
```
Expected: prints overall accuracy. Compare against baseline `flat_solo_da` at 48.8% val (per CLAUDE.md).

- [ ] **Step 4: Append result to CLAUDE.md "Best Results" table only if it beats baseline.**

If it doesn't beat baseline, the comparison still matters — record in the run notes regardless, but skip the headline table edit.
