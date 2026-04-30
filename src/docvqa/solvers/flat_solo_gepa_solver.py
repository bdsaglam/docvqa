"""Flat solo solver — GEPA-friendly variant with candidate-injectable prompts.

This is a near-exact copy of `flat_solo_solver.py` with three differences:

1. `task_instructions` and `tips_overrides` are passed in (no module-level
   import of TASK_INSTRUCTIONS / CATEGORY_TIPS at construction time).
2. `solve_document` accepts an optional `precomputed` dict so the GEPA
   evaluator can skip per-call image saving / BM25 building.
3. `create_flat_solo_gepa_program` factory loads candidates from a JSON path
   so the same Hydra config can run seed (null) or any optimized candidate.

The production `flat_solo_solver.py` is intentionally left untouched.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy
import logfire
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from docvqa.data import Document
from docvqa.metrics import evaluate_prediction
from docvqa.prompts import CATEGORY_TIPS
from docvqa.rlm import LeanRLM, CodeRLM, ThinkingRLM, RLM
from docvqa.search import get_or_build_index
from docvqa.solvers.flat_solo_solver import TASK_INSTRUCTIONS as SEED_TASK_INSTRUCTIONS
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)


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


def _category_tips_block(category: str, tips_overrides: dict[str, str]) -> str:
    """Return the CATEGORY-SPECIFIC TIPS block for `category`, using overrides if present."""
    tips = tips_overrides.get(category, "")
    if tips:
        return f"## CATEGORY-SPECIFIC TIPS ({category})\n{tips}"
    return ""


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
# FlatSoloGepaProgram
# ---------------------------------------------------------------------------

DEFAULT_VLM_PROMPT = (
    "Analyze the image content strictly to answer the query. "
    "Transcribe numbers and characters exactly. "
    "For technical drawings, trace leader lines and arrows to connect labels to their specific parts. "
    "Output ONLY the concise answer. If the information is missing, output 'Unknown'."
)


class FlatSoloGepaProgram:
    """Flat solo solver with injectable task_instructions and per-category tips.

    Behaviorally identical to FlatSoloProgram when constructed with the seed
    candidate (TASK_INSTRUCTIONS + CATEGORY_TIPS).
    """

    def __init__(
        self,
        vlm_lm: dspy.LM,
        task_instructions: str,
        tips_overrides: dict[str, str],
        max_iterations: int = 20,
        rlm_type: str = "lean",
        page_factor: float = 1.5,
        question_concurrency: int = 1,
        vlm_prompt: str | None = None,
    ):
        self.vlm_lm = vlm_lm
        self.task_instructions = task_instructions
        self.tips_overrides = dict(tips_overrides)
        self.max_iterations = max_iterations
        self.rlm_type = rlm_type
        self.page_factor = page_factor
        self.question_concurrency = question_concurrency
        self.vlm_prompt = vlm_prompt or DEFAULT_VLM_PROMPT

        self.vlm_predict = dspy.Predict(
            dspy.Signature(
                {
                    "image": (dspy.Image, dspy.InputField(desc="Page or cropped region image")),
                    "query": (str, dspy.InputField(desc="What to look for or describe")),
                    "answer": (str, dspy.OutputField(desc="Concise response")),
                },
                self.vlm_prompt,
            )
        )

    def solve_document(
        self,
        document: Document,
        precomputed: dict | None = None,
    ) -> tuple[dict[str, str], dict[str, list[dict]]]:
        """Solve all questions for a document, one question at a time.

        If `precomputed` is provided (dict with `page_dir`, `search_index`,
        `page_texts_formatted`), skips per-call image saving / BM25 building.
        """
        if precomputed is not None:
            page_dir = precomputed["page_dir"]
            search_index = precomputed["search_index"]
            page_texts = precomputed["page_texts_formatted"]
            return self._solve_with_context(document, page_dir, search_index, page_texts)

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, img in enumerate(document.images):
                img.save(os.path.join(tmpdir, f"page_{i}.png"), format="PNG")

            search_index = None
            if document.page_texts:
                search_index = get_or_build_index(document.doc_id, document.page_texts)

            page_texts = _format_page_texts(document.page_texts) if document.page_texts else None
            if page_texts is None:
                page_texts = ["[No OCR text available]"]

            return self._solve_with_context(document, tmpdir, search_index, page_texts)

    def _solve_with_context(
        self,
        document: Document,
        page_dir: str,
        search_index: Any,
        page_texts: list[str],
    ) -> tuple[dict[str, str], dict[str, list[dict]]]:
        ctx = RunContext(
            page_dir=page_dir,
            num_pages=len(document.images),
            search_index=search_index,
            page_texts=document.page_texts,
        )

        doc_info = f"Category: {document.doc_category}, Pages: {len(document.images)}"

        num_pages = len(document.images)
        page_bonus = min(10, self.page_factor * math.ceil(math.sqrt(max(0, num_pages - 9))))
        max_iter = self.max_iterations + int(page_bonus)

        tips_block = _category_tips_block(document.doc_category, self.tips_overrides)
        instructions = self.task_instructions + ("\n" + tips_block if tips_block else "")
        tools = _create_tools(self.vlm_predict, self.vlm_lm, ctx)
        sandbox_code = _build_sandbox_code(page_dir, len(document.images))

        def _solve_question(q):
            """Solve a single question. Returns (question_id, answer, trajectory)."""
            with logfire.span(
                "solve_flat_solo_gepa",
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
                    "Flat solo gepa (%s) Q %s: max_iterations=%d (page_bonus=%d)",
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
                    logger.warning("Flat solo gepa failed for Q '%s': %s", q.question_id, e)
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
                        "Solo gepa Q %s: %s (GT=%s, PRED=%s)",
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
            logger.info(
                "Flat solo gepa: running %d questions with concurrency=%d",
                len(document.questions), max_w,
            )
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                futures = {pool.submit(_solve_question, q): q for q in document.questions}
                for future in as_completed(futures):
                    qid, answer, trajectory = future.result()
                    predictions[qid] = answer
                    trajectories[qid] = trajectory

        for q in document.questions:
            if q.answer is not None:
                scored_count += 1
                is_correct, _ = evaluate_prediction(predictions[q.question_id], q.answer)
                if is_correct:
                    correct_count += 1

        if scored_count > 0:
            logger.info(
                "Flat solo gepa doc %s: %d/%d = %.1f%%",
                document.doc_id, correct_count, scored_count,
                100 * correct_count / scored_count,
            )

        return predictions, trajectories


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------

CATEGORIES = (
    "business_report",
    "comics",
    "engineering_drawing",
    "infographics",
    "maps",
    "science_paper",
    "science_poster",
    "slide",
)


def build_seed_candidate() -> dict[str, str]:
    """Build the seed candidate from the current production prompts."""
    candidate: dict[str, str] = {"task_instructions": SEED_TASK_INSTRUCTIONS}
    for cat in CATEGORIES:
        candidate[f"tip_{cat}"] = CATEGORY_TIPS.get(cat, "")
    return candidate


def load_candidate(candidate_path: str | Path | None) -> dict[str, str]:
    """Load a candidate JSON, or return seed if path is None."""
    if candidate_path is None:
        return build_seed_candidate()
    data = json.loads(Path(candidate_path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Candidate at {candidate_path} is not a dict")
    # Guard: ensure all keys are present; fall back to seed for any missing
    seed = build_seed_candidate()
    merged = dict(seed)
    for k, v in data.items():
        if not isinstance(v, str):
            raise ValueError(f"Candidate key '{k}' is not a string (got {type(v).__name__})")
        merged[k] = v
    if "task_instructions" not in merged:
        raise ValueError("Candidate missing 'task_instructions'")
    return merged


def candidate_to_overrides(candidate: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Split a candidate dict into (task_instructions, tips_overrides)."""
    task_instructions = candidate["task_instructions"]
    tips_overrides = {cat: candidate.get(f"tip_{cat}", "") for cat in CATEGORIES}
    return task_instructions, tips_overrides


# ---------------------------------------------------------------------------
# Factory for hydra instantiation
# ---------------------------------------------------------------------------

def create_flat_solo_gepa_program(
    candidate_path: str | None = None,
    max_iterations: int = 20,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
) -> FlatSoloGepaProgram:
    candidate = load_candidate(candidate_path)
    task_instructions, tips_overrides = candidate_to_overrides(candidate)

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

    return FlatSoloGepaProgram(
        vlm_lm=vlm_lm,
        task_instructions=task_instructions,
        tips_overrides=tips_overrides,
        max_iterations=max_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
    )
