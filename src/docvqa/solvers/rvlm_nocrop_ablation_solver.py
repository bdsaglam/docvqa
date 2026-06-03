"""RVLM no-crop ablation — `batch_look` operates on WHOLE pages by index.

Clean fork of :mod:`docvqa.solvers.rvlm_solver` (the proposed method) that
removes exactly one affordance: **cropping / zooming**. In `rvlm`,
`batch_look(requests)` takes `(image, query)` tuples where the image may be
a full page *or* an arbitrary crop (`pages[i].crop((l, t, r, b))`) — the
agent can zoom into a tight region to read fine print. Here,
`batch_look(requests)` takes `(page_index, query)` tuples: the agent can
only query a **whole page** by its 0-indexed number. There are no PIL page
objects in scope, so there is no way to crop.

Everything else — the LeanRLM REPL scaffold, the recursive VLM sub-call,
the prompt body, the profile/answer rules — is identical to `rvlm`. The
paired Δ vs `rvlm` therefore isolates the contribution of **cropping** to
the active-perception loop: how much of the score comes from the agent's
ability to zoom into a region vs reading whole pages.

Hypothesis: cropping matters most on detail-dense categories
(engineering_drawing, business_report, maps) where a whole-page read can't
resolve fine print — there the no-crop ablation should drop the most.

Reuses `rvlm`'s shared helpers (`_create_tools`, `_build_signature`,
`vlm_predict`) unchanged; overrides only the sandbox `batch_look` wrapper
and the parts of the prompt that reference cropping.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from typing import Any

import dspy
import logfire

from docvqa.data import Document, Question
from docvqa.datasets.profile import DatasetProfile, get_profile
from docvqa.rlm import LeanRLM, CodeRLM, ThinkingRLM, RLM
from docvqa.types import LMConfig
from docvqa.solvers.rvlm_solver import _create_tools, _build_signature

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sandbox: batch_look(requests) where each request is (page_index, query).
# No PIL `pages` objects → no crop is possible. The page PNGs already live
# on disk (page_dir/page_<i>.png), so we map index → path directly.
# ---------------------------------------------------------------------------

def _build_sandbox_code(page_dir: str, num_pages: int) -> str:
    return f'''
# No-crop ablation: batch_look reads WHOLE pages by 0-indexed number.
# There are no PIL page objects in scope and no way to crop a region.
NUM_PAGES = {num_pages}

def batch_look(requests):
    """Send one or more pages to the VLM in parallel.
    Input: list of (page_index, query) tuples, where page_index is a 0-indexed
    page number in [0, NUM_PAGES-1). Returns: list of str answers (same order).
    Example: batch_look([(0, "what is the title?"), (2, "read the data table")])
    For a single query: batch_look([(page_index, query)])[0]."""
    import json as _json, os as _os
    paths = []
    for page_index, query in requests:
        page_index = int(page_index)
        assert 0 <= page_index < {num_pages}, f"page_index out of range (0..{num_pages - 1}): {{page_index}}"
        paths.append({{"path": _os.path.join({page_dir!r}, f"page_{{page_index}}.png"), "query": query}})
    return _batch_look_impl(_json.dumps(paths))
'''


# ---------------------------------------------------------------------------
# Task body: rvlm's body with every crop affordance removed. Whole-page
# reads only; `batch_look` is addressed by page index.
# ---------------------------------------------------------------------------

_TASK_BODY = (
    "You are a Document Visual Question Answering agent. You answer a question about a document by "
    "writing Python code, calling vision tools iteratively, and reasoning programmatically.\n\n"

    "## DATA\n"
    "- `question`: The question you must answer.\n"
    "- `doc_info`: Document metadata, including the page count. Pages are addressed by their "
    "0-indexed number (0 .. page_count-1); `NUM_PAGES` holds the count.\n\n"

    "## TOOLS\n"
    "- batch_look(requests) -> list[str]\n"
    "  What: send one or more WHOLE pages to a VLM in parallel.\n"
    "  When: any visual question — full-page survey, locating content, reading a value.\n"
    "  How: list of (page_index, query) tuples. `page_index` is a 0-indexed page number. "
    "Returns answers in the same order. For a single query: "
    "`batch_look([(page_index, query)])[0]`.\n"
    "- SUBMIT(answer=\"...\")\n"
    "  What: deliver the final answer and terminate.\n"
    "  When: you have the answer and have verified it.\n\n"

    "## APPROACH\n"
    "1. SURVEY — read the document at a coarse level to build a mental map. "
    "Use full-page `batch_look` queries; for many-page docs, batch a sample "
    "of pages in one call.\n"
    "2. LOCATE — identify the page(s) that contain the answer.\n"
    "3. EXTRACT — get the values out of the relevant page with `batch_look`. "
    "Ask ONE simple, specific factual question per VLM call (a focused query "
    "pulls a precise value out of a whole page better than a broad one).\n"
    "4. VERIFY — for any precise value (numbers, fine text, small labels), "
    "do not commit a reading you've only seen once. Design a check: "
    "re-read the page with a differently-phrased query, look for consistency "
    "across reads, or cross-reference an adjacent label. See the verification "
    "guidance below.\n"
    "5. SUBMIT — call `SUBMIT(answer=\"...\")` once you have the answer.\n\n"

    "Never use outside or world knowledge. Every answer must come from the "
    "document.\n\n"

    "## DOCUMENT-SHAPE GUIDANCE\n"
    "Apply the patterns below that match the document at hand.\n\n"

    "- **The VLM is unreliable; reliability is your job.** The underlying "
    "VLM is non-deterministic — the same page and query can return "
    "different answers across calls, especially for precise values "
    "(numbers, fine text, small labels) and high-density pages. A "
    "single read is not trustworthy. Build a reading procedure that "
    "compensates: read the same page multiple times and look for the "
    "consistent answer; rephrase the query to target the specific value; "
    "ask several narrow questions about a page instead of one broad one; "
    "cross-check against an adjacent label or value. Silently swapping a "
    "value after one re-read with no evidence is just noise.\n\n"

    "- **High-density single page** (large image, lots of detail per "
    "page): a single broad `batch_look` query can miss fine detail. Ask "
    "the VLM ONE narrow, specific question at a time (\"what is the value "
    "in the third row of the revenue column?\" rather than \"read the "
    "table\"), and re-read the page with different phrasings to pin down a "
    "value.\n\n"

    "- **Many-page document** (slides, papers, reports): you do NOT need to "
    "read every page. Survey in batches "
    "(`batch_look([(i, 'summarize') for i in sample])`) to build a "
    "table-of-contents in your head. Then drill into the relevant page.\n\n"

    "- **Counting / superlatives / 'all of'** questions (\"how many...\", "
    "\"which is largest...\", \"list all...\"): enumerate ALL candidates "
    "first by surveying the document. Do NOT stop at the first match. "
    "Once you have the candidate set, compare or count in Python.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string: `SUBMIT(answer=\"42\")`.\n"
    "- The answer must follow these formatting rules:\n\n"
)


def _build_task_instructions(profile: DatasetProfile) -> str:
    return _TASK_BODY + profile.answer_formatting_rules


class RvlmNocropAblationProgram:
    """RVLM no-crop ablation — whole-page `batch_look` by index. See module docstring."""

    def __init__(
        self,
        vlm_lm: dspy.LM,
        profile: DatasetProfile,
        max_iterations: int = 25,
        rlm_type: str = "lean",
        page_factor: float = 1.5,
        question_concurrency: int = 4,
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
                "When a label is separated from the item it identifies, trace any visual connector (leader line, arrow, callout, alignment) to determine which item it refers to. "
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

            instructions = _build_task_instructions(self.profile)
            tools = _create_tools(self.vlm_predict, self.vlm_lm, self.batch_concurrency)
            sandbox_code = _build_sandbox_code(tmpdir, len(document.images))

            def _solve_question(q: Question):
                with logfire.span(
                    "solve_rvlm_nocrop",
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
                        "RVLM-NOCROP [%s] (%s) Q %s: max_iterations=%d (page_bonus=%d)",
                        self.profile.name, self.rlm_type, q.question_id, max_iter, int(page_bonus),
                    )

                    result = rlm(question=question_text, doc_info=doc_info)
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
                            "RVLM-NOCROP[%s] Q %s: %s (GT=%s, PRED=%s)",
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
                logger.info("RVLM-NOCROP: running %d questions with concurrency=%d", len(document.questions), max_w)
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
                    if self.profile.score_fn(predictions[q.question_id], q.answer, q)[0]:
                        correct += 1
            if scored > 0:
                logger.info(
                    "RVLM-NOCROP [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name, document.doc_id, correct, scored,
                    100 * correct / scored,
                )

            return predictions, trajectories


def create_rvlm_nocrop_ablation_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 25,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    batch_concurrency: int = 8,
) -> RvlmNocropAblationProgram:
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

    return RvlmNocropAblationProgram(
        vlm_lm=vlm_lm,
        profile=profile,
        max_iterations=max_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
        batch_concurrency=batch_concurrency,
    )
