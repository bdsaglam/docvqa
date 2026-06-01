"""RVLM+OCR solver — RVLM with an OCR retrieval channel (M+OCR extension per D-006).

Tool surface = ``batch_look`` (the RVLM recursive sub-call) plus ``search``
(BM25 over OCR text) and ``page_texts`` in scope. **No** single-image
``look()`` registration: callers wanting a single visual query use
``batch_look([(image, query)])[0]`` (same idiom as rvlm).

This is a clean fork of :mod:`docvqa.solvers.rvlm_solver`. Distinct from
:mod:`docvqa.solvers.rvlm_full_ablation_solver`, which confounds the OCR extension
with a single-image ``look()`` ergonomic wrapper.

Engineering name per D-010 (paper-facing name TBD).

Prompt parity (D-007). Per an n=8 ablation the hand-crafted per-category
tips are not load-bearing, so this solver carries a MINIMAL prompt that
matches ``rvlm_minimal``: a category-agnostic ``_TASK_BODY`` plus the
profile's answer-formatting rules, with no per-category overlay or
per-category tips dispatch. The generic search / OCR /
``page_texts`` / ``batch_look`` tool documentation lives in ``_TASK_BODY``
and is retained — only the per-category specializations are removed.
``ANSWER_FORMATTING_RULES`` is read from
``profile.answer_formatting_rules`` — not imported from
:mod:`docvqa.prompts`.
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

from docvqa.data import Document, Question
from docvqa.datasets.profile import DatasetProfile, get_profile
from docvqa.rlm import LeanRLM, CodeRLM, ThinkingRLM, RLM
from docvqa.search import get_or_build_index
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt body (formatting rules substituted from the profile)
# ---------------------------------------------------------------------------

_TASK_BODY = (
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
    "- NEVER use outside/world knowledge. ALL answers MUST come from the document.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string.\n"
    '- Example: SUBMIT(answer="42")\n'
    "- The answer must follow these formatting rules:\n\n"
)

def _build_task_instructions(profile: DatasetProfile) -> str:
    return _TASK_BODY + profile.answer_formatting_rules

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
# RvlmOcrAblationProgram
# ---------------------------------------------------------------------------

class RvlmOcrAblationProgram:
    """RVLM+OCR solver — each question solved independently.

    Tool surface: ``batch_look()`` (from rvlm) + ``search()`` + ``page_texts``
    (from OCR). No ``look()`` is registered — that ergonomic wrapper is what
    distinguishes this solver from ``rvlm_full``, the kitchen-sink variant.
    """

    def __init__(
        self,
        vlm_lm: dspy.LM,
        profile: DatasetProfile,
        max_iterations: int = 20,
        rlm_type: str = "lean",
        page_factor: float = 1.5,
        question_concurrency: int = 1,
        batch_concurrency: int = 8,
    ):
        self.vlm_lm = vlm_lm
        self.profile = profile
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

    def _per_question_prefix(self, q: Question) -> str:
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

            # No category tips, no per-document dispatch — the body is the body.
            instructions = _build_task_instructions(self.profile)
            tools = _create_tools(self.vlm_predict, self.vlm_lm, ctx, self.batch_concurrency)
            sandbox_code = _build_sandbox_code(tmpdir, len(document.images))

            def _solve_question(q: Question):
                """Solve a single question. Returns (question_id, answer, trajectory)."""
                with logfire.span(
                    "solve_rvlm_ocr",
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
                        "RVLM+OCR [%s] (%s) Q %s: max_iterations=%d (page_bonus=%d)",
                        self.profile.name, self.rlm_type, q.question_id, max_iter, int(page_bonus),
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
                            "RVLM+OCR[%s] Q %s: %s (GT=%s, PRED=%s)",
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
                logger.info("RVLM+OCR: running %d questions with concurrency=%d", len(document.questions), max_w)
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
                    "RVLM+OCR [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name, document.doc_id, correct, scored,
                    100 * correct / scored,
                )

            return predictions, trajectories

# ---------------------------------------------------------------------------
# Factory for hydra instantiation
# ---------------------------------------------------------------------------

def create_rvlm_ocr_ablation_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 20,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    batch_concurrency: int = 8,
) -> RvlmOcrAblationProgram:
    """Hydra factory. See ``rvlm_solver.create_rvlm_program``."""
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

    return RvlmOcrAblationProgram(
        vlm_lm=vlm_lm,
        profile=profile,
        max_iterations=max_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
        batch_concurrency=batch_concurrency,
    )
