"""ReAct baseline: ``dspy.ReAct`` with VLM-only tools, no REPL.

The point of this solver, vs ``rvlm_solver``, is to test whether the
code-REPL (LeanRLM + Python sandbox) is doing real work or whether plain
ReAct-style iterative tool use reaches the same accuracy. Same VLM tools
in spirit (single + parallel page-level perception), but **no Python
execution**, so no crops, no arithmetic on retrieved values, no
intermediate variables.

Tool surface (JSON-arg-friendly so ``dspy.ReAct`` can format calls):
- ``look(page_index: int, query: str) -> str``: single page → VLM query.
- ``look_many(page_indices: list[int], query: str) -> list[str]``: same
  query across many pages, in parallel.

What's intentionally lost vs ``rvlm``:
- No ``PIL.crop`` on retrieved page images — ReAct can't construct PIL
  ops, so fine-detail extraction degrades.
- No arithmetic / list comprehensions / counting in Python — superlative
  and compound-answer questions must be assembled by the LM itself from
  the trajectory.

Dataset-aware via the injected :class:`DatasetProfile` (same pattern as
``rvlm_solver``).
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from typing import Any

import dspy
import logfire
from PIL import Image as PILImage
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from docvqa.data import Document, Question
from docvqa.datasets.profile import DatasetProfile, get_profile
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt body
# ---------------------------------------------------------------------------

_TASK_BODY = (
    "You are a Document Visual Question Answering agent. You answer a "
    "question about a document by iteratively calling vision tools to "
    "examine document pages and reasoning over the observations.\n\n"

    "## TOOLS\n"
    "- look(page_index, query) -> str: Send a single page image to the "
    "VLM with a query. Returns the VLM's answer.\n"
    "- look_many(page_indices, query) -> list[str]: Send the SAME query "
    "to many pages in parallel. Returns answers in the same order as the "
    "indices. Use this to survey many pages at once.\n"
    "- finish: Signals that you have all the information you need and are "
    "ready to extract the final answer.\n\n"

    "## APPROACH\n"
    "1. EXPLORE: Use look_many on a sample of pages (or all pages, if "
    "few) with a broad query like 'Describe layout and main content' to "
    "build a mental map of the document.\n"
    "2. LOCATE: Identify which page(s) hold the evidence for the question.\n"
    "3. EXTRACT: Use look on those specific pages with a tightly-focused "
    "query to read exact values.\n"
    "4. VERIFY: If a critical reading is ambiguous, re-ask with a more "
    "specific query.\n"
    "5. FINISH: When you have the answer, call finish.\n\n"

    "## GUIDELINES\n"
    "- Ask the VLM ONE simple factual question per call. Do NOT combine "
    "multiple questions into one query.\n"
    "- VLM CONFLICT RESOLUTION: The VLM can give different answers on "
    "repeated calls. When readings conflict, ask a more specific "
    "question. Do not silently adopt a new number from a later 'verify' "
    "pass.\n"
    "- SUPERLATIVES: For 'largest', 'first', 'last', 'only' questions — "
    "enumerate ALL candidates with look_many first, then pick.\n"
    "- COMPUTATION: When a question implies arithmetic (totals, "
    "differences), gather the raw values first, then compute in your "
    "reasoning.\n"
    "- NEVER use outside/world knowledge. ALL answers MUST come from the "
    "document.\n\n"

    "## OUTPUT FORMAT\n"
    "- The final ``answer`` field must follow these formatting rules:\n\n"
)


def _build_task_instructions(profile: DatasetProfile) -> str:
    return _TASK_BODY + profile.answer_formatting_rules


# ---------------------------------------------------------------------------
# Signature + tools
# ---------------------------------------------------------------------------


def _build_signature(instructions: str) -> dspy.Signature:
    fields: dict = {
        "question": (
            str,
            dspy.InputField(desc="The question to answer about the document"),
        ),
        "doc_info": (str, dspy.InputField(desc="Document metadata: category and page count")),
        "answer": (
            str,
            dspy.OutputField(desc="The answer string for the question."),
        ),
    }
    return dspy.Signature(fields, instructions)


def _create_tools(
    vlm_predict: dspy.Predict,
    vlm_lm: dspy.LM,
    page_dir: str,
    num_pages: int,
    batch_concurrency: int,
) -> list:
    """Build the look / look_many tools bound to the per-document page dir."""

    PILImage.MAX_IMAGE_PIXELS = 500_000_000

    def _page_path(idx: int) -> str:
        return os.path.join(page_dir, f"page_{idx}.png")

    def _query_one(idx: int, query: str) -> str:
        if not (0 <= idx < num_pages):
            return f"[Error] page_index out of range: {idx} (document has {num_pages} pages, indices 0..{num_pages - 1})"
        path = _page_path(idx)
        if not os.path.exists(path):
            return f"[Error] page image missing: {path}"
        with logfire.span("react.look", page_index=idx, query=query[:200]) as span:
            img = PILImage.open(path)
            with dspy.context(lm=vlm_lm):
                result = vlm_predict(image=dspy.Image(img), query=query)
                answer = (result.answer or "")
                span.set_attribute("answer", answer[:2000])
                return answer

    def look(page_index: int, query: str) -> str:
        """Send a single page image to the VLM with a query. Returns the VLM's answer."""
        return _query_one(int(page_index), str(query))

    def look_many(page_indices: list[int], query: str) -> list[str]:
        """Send the same query to many pages in parallel. Returns answers in the same order as the indices."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        idxs = [int(i) for i in page_indices]
        if not idxs:
            return []
        results: list[str] = [""] * len(idxs)
        max_w = min(len(idxs), batch_concurrency)

        with logfire.span("react.look_many", num_pages=len(idxs), query=str(query)[:200]):
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                futures = {pool.submit(_query_one, idxs[i], str(query)): i for i in range(len(idxs))}
                for fut in as_completed(futures):
                    i = futures[fut]
                    results[i] = fut.result()
        return results

    return [look, look_many]


# ---------------------------------------------------------------------------
# ReactProgram
# ---------------------------------------------------------------------------


def _trajectory_to_list(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert dspy.ReAct's flat trajectory dict to a list of per-iter dicts.

    Input keys look like ``thought_0, tool_name_0, tool_args_0, observation_0,
    thought_1, ...``. Each group of 4 becomes one entry. The trajectory may
    be truncated, so we group whatever is present.
    """
    if not traj:
        return []
    # Group keys by iteration index parsed from the suffix.
    by_idx: dict[int, dict[str, Any]] = {}
    for k, v in traj.items():
        parts = k.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        field, idx_s = parts[0], int(parts[1])
        by_idx.setdefault(idx_s, {})[field] = v
    return [by_idx[i] for i in sorted(by_idx.keys())]


class ReactProgram:
    """ReAct baseline: ``dspy.ReAct`` over VLM tools, no Python REPL."""

    def __init__(
        self,
        vlm_lm: dspy.LM,
        profile: DatasetProfile,
        max_iterations: int = 25,
        page_factor: float = 1.5,
        question_concurrency: int = 4,
        batch_concurrency: int = 8,
    ):
        self.vlm_lm = vlm_lm
        self.profile = profile
        self.max_iterations = max_iterations
        self.page_factor = page_factor
        self.question_concurrency = question_concurrency
        self.batch_concurrency = batch_concurrency

        self.vlm_predict = dspy.Predict(
            dspy.Signature(
                {
                    "image": (dspy.Image, dspy.InputField(desc="Page image")),
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
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, img in enumerate(document.images):
                img.save(os.path.join(tmpdir, f"page_{i}.png"), format="PNG")

            doc_info = f"Category: {document.doc_category}, Pages: {len(document.images)}"

            num_pages = len(document.images)
            page_bonus = min(10, self.page_factor * math.ceil(math.sqrt(max(0, num_pages - 9))))
            max_iter = self.max_iterations + int(page_bonus)

            base_instructions = _build_task_instructions(self.profile)
            tips = self.profile.category_tips_fn(document.doc_category)
            instructions = base_instructions + ("\n" + tips if tips else "")
            tools = _create_tools(
                self.vlm_predict, self.vlm_lm, tmpdir, num_pages, self.batch_concurrency
            )
            signature = _build_signature(instructions)

            def _solve_question(q: Question):
                with logfire.span(
                    "solve_react",
                    doc_id=document.doc_id,
                    question_id=q.question_id,
                    question=q.question[:200],
                    profile=self.profile.name,
                ) as q_span:
                    question_text = q.question + self._per_question_prefix(q)
                    react = dspy.ReAct(signature=signature, tools=tools, max_iters=max_iter)
                    logger.info(
                        "REACT [%s] Q %s: max_iters=%d (page_bonus=%d)",
                        self.profile.name, q.question_id, max_iter, int(page_bonus),
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
                        return react(question=question_text, doc_info=doc_info)

                    result = _solve_one()
                    answer = str(getattr(result, "answer", "") or "").strip()
                    trajectory_list = _trajectory_to_list(getattr(result, "trajectory", {}) or {})

                    if not answer:
                        answer = "Unknown"

                    q_span.set_attribute("num_iterations", len(trajectory_list))
                    q_span.set_attribute("prediction", answer[:200])

                    if q.answer is not None:
                        is_correct, extracted = self.profile.score_fn(answer, q.answer, q)
                        q_span.set_attribute("is_correct", is_correct)
                        q_span.set_attribute("ground_truth", q.answer[:200])
                        q_span.set_attribute("extracted_answer", extracted[:200])
                        logger.info(
                            "REACT[%s] Q %s: %s (GT=%s, PRED=%s)",
                            self.profile.name,
                            q.question_id,
                            "CORRECT" if is_correct else "WRONG",
                            q.answer[:40],
                            extracted[:40],
                        )

                    return q.question_id, answer, trajectory_list

            predictions: dict[str, str] = {}
            trajectories: dict[str, list[dict]] = {}

            if self.question_concurrency <= 1:
                for q in document.questions:
                    qid, answer, traj = _solve_question(q)
                    predictions[qid] = answer
                    trajectories[qid] = traj
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                max_w = min(self.question_concurrency, len(document.questions))
                logger.info("REACT: running %d questions with concurrency=%d", len(document.questions), max_w)
                with ThreadPoolExecutor(max_workers=max_w) as pool:
                    futures = {pool.submit(_solve_question, q): q for q in document.questions}
                    for future in as_completed(futures):
                        qid, answer, traj = future.result()
                        predictions[qid] = answer
                        trajectories[qid] = traj

            correct = 0
            scored = 0
            for q in document.questions:
                if q.answer is not None:
                    scored += 1
                    if self.profile.score_fn(predictions[q.question_id], q.answer, q)[0]:
                        correct += 1
            if scored > 0:
                logger.info(
                    "REACT [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name, document.doc_id, correct, scored,
                    100 * correct / scored,
                )

            return predictions, trajectories


def create_react_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 25,
    vlm: dict[str, Any] | None = None,
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    batch_concurrency: int = 8,
) -> ReactProgram:
    """Hydra factory — mirrors ``create_rvlm_program``."""

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

    return ReactProgram(
        vlm_lm=vlm_lm,
        profile=profile,
        max_iterations=max_iterations,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
        batch_concurrency=batch_concurrency,
    )
