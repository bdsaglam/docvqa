"""Dataset-aware flat solo solver.

A clone of :mod:`docvqa.solvers.flat_solo_solver` with one extra knob:
a :class:`docvqa.datasets.profile.DatasetProfile` injected via the
constructor. The profile drives:

- **Answer-formatting rules** appended to the agent's task instructions
  (replaces the DocVQA-2026-specific :data:`ANSWER_FORMATTING_RULES`).
- **Per-category tips** via ``profile.category_tips_fn`` (a no-op for
  benchmarks with a single category like MP-DocVQA / MMLongBench-Doc).
- **Per-question format hint** via ``profile.question_format_hint_fn``
  — surfaces MMLongBench-Doc's ``answer_format`` inline so the agent
  picks the right formatter.

Scoring is **not** done inside the solver. The runner (or eval-loop)
should call ``profile.score_fn(pred, gt, question)`` to compute
correctness, so this same solver can ship Qwen-judge-scored predictions
for MMLongBench-Doc and ANLS-scored predictions for MP-DocVQA without
forking the solver.

The flat_solo helpers (``_format_page_texts``, ``_build_signature``,
``_strip_search_tool``, ``_create_tools``, ``_build_sandbox_code``,
``_build_sandbox_code_page_only``, ``RunContext``) are reused as-is.
The class also reuses the per-question retry / iteration-budget logic
from flat_solo.
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
    _build_sandbox_code,
    _build_sandbox_code_page_only,
    _build_signature,
    _create_tools,
    _format_page_texts,
    _strip_search_tool,
)
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template (formatting rules are substituted from the profile)
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


# ---------------------------------------------------------------------------
# FlatSoloDAProgram (dataset-aware)
# ---------------------------------------------------------------------------


class FlatSoloDAProgram:
    """Dataset-aware flat_solo solver. See module docstring."""

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

            base_instructions = _build_task_instructions(self.profile, self.vlm_cropping)
            if not self.use_search:
                base_instructions = _strip_search_tool(base_instructions)
            tips = self.profile.category_tips_fn(document.doc_category)
            instructions = base_instructions + ("\n" + tips if tips else "")
            tools = _create_tools(self.vlm_predict, self.vlm_lm, ctx, use_search=self.use_search)
            if self.vlm_cropping:
                sandbox_code = _build_sandbox_code(tmpdir, len(document.images), use_search=self.use_search)
            else:
                sandbox_code = _build_sandbox_code_page_only(tmpdir, len(document.images), use_search=self.use_search)

            def _solve_question(q: Question):
                """Solve a single question. Returns (question_id, answer, trajectory)."""
                with logfire.span(
                    "solve_flat_solo_da",
                    doc_id=document.doc_id,
                    question_id=q.question_id,
                    question=q.question[:200],
                    profile=self.profile.name,
                ) as q_span:
                    # The format hint (if any) is prepended to the question text so the
                    # agent sees it as part of the user task — it survives RLM round-trips
                    # without needing a new signature slot.
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
                        "Flat solo DA [%s] (%s) Q %s: max_iterations=%d (page_bonus=%d)",
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
                            "Solo[%s] Q %s: %s (GT=%s, PRED=%s)",
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
                logger.info("Flat solo DA: running %d questions with concurrency=%d", len(document.questions), max_w)
                with ThreadPoolExecutor(max_workers=max_w) as pool:
                    futures = {pool.submit(_solve_question, q): q for q in document.questions}
                    for future in as_completed(futures):
                        qid, answer, trajectory = future.result()
                        predictions[qid] = answer
                        trajectories[qid] = trajectory

            # Per-doc log line; final scoring happens in the runner via
            # profile.score_fn (this same function is run there).
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
                    "Flat solo DA [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name, document.doc_id, correct, scored,
                    100 * correct / scored,
                )

            return predictions, trajectories


# ---------------------------------------------------------------------------
# Factory for Hydra instantiation
# ---------------------------------------------------------------------------


def create_flat_solo_da_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 20,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    vlm_cropping: bool = True,
    use_search: bool = True,
) -> FlatSoloDAProgram:
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

    return FlatSoloDAProgram(
        vlm_lm=vlm_lm,
        profile=profile,
        max_iterations=max_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
        vlm_cropping=vlm_cropping,
        use_search=use_search,
    )
