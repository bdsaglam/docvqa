"""Hybrid RVLM — agent has both `display()` and `ask_vlm()`, picks per call.

Background. The rvlm-family solvers expose only a sub-VLM tool (the
agent never sees images itself; it delegates every visual query). The
direct_vlm-family solvers expose only `display()` (the agent is itself
a multimodal LM and sees images inline). Both families peg on
DocVQA-2026 in different ways.

Question. Given the *choice*, what does the agent pick? If it mostly
delegates via `ask_vlm`, that's positive evidence the recursive-
perception pattern is the right one — the rvlm architecture is not a
forced detour. If it mostly looks itself via `display`, the rvlm
family's always-delegate design is paying a tax it doesn't need to. If
it mixes by question shape, that's interesting and worth a section.

Implementation. Uses :class:`docvqa.rlm.multimodal.MultimodalRLM` (the
multimodal-capable RLM that pipes `display(image)` back to the next
turn as an inline `image_url` block) and provides `ask_vlm(image,
query) -> str` as a tool (a single-call wrapper over the same sub-VLM
predict path the rvlm family uses).

Body is dataset-agnostic — DocVQA-specific answer-format rules come
from ``profile.answer_formatting_rules`` exactly like in
``rvlm_minimal_solver`` and ``direct_vlm_minimal_solver``.
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
from docvqa.rlm import MultimodalRLM
from docvqa.solvers.rvlm_unified_solver import _build_signature
from docvqa.types import LMConfig
from docvqa.retry_utils import is_retryable_lm_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sub-VLM tool: ask_vlm(image, query) -> str. Single-call (no batch) so the
# tool API parallels display() in shape; the agent can issue multiple calls
# in one code block if it wants parallelism by writing the loop itself.
# ---------------------------------------------------------------------------

def _create_ask_vlm_tool(vlm_predict: dspy.Predict, vlm_lm: dspy.LM) -> list:
    from PIL import Image as PILImage

    def _ask_vlm_impl(image_path: str, query: str) -> str:
        """Internal: load image from path and send to a fresh sub-VLM call."""
        with logfire.span("ask_vlm", image_path=image_path, query=query) as span:
            img = PILImage.open(image_path)
            with dspy.context(lm=vlm_lm):
                result = vlm_predict(image=dspy.Image(img), query=query)
                answer = result.answer or ""
                span.set_attribute("answer", answer[:2000])
                return answer

    return [_ask_vlm_impl]

def _build_sandbox_code(page_dir: str, num_pages: int) -> str:
    """Sandbox with pages + a Python wrapper for the ask_vlm tool."""
    return f'''
import os
import tempfile
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000
pages = []
for i in range({num_pages}):
    path = os.path.join({page_dir!r}, f"page_{{i}}.png")
    assert os.path.exists(path), f"Page image not found: {{path}}"
    pages.append(Image.open(path))
assert len(pages) == {num_pages}, f"Expected {num_pages} pages, got {{len(pages)}}"

def ask_vlm(image, query):
    """Send image+query to a fresh sub-VLM call. Returns its text answer.
    image: any PIL Image (a page like pages[i], or a crop).
    query: a focused natural-language question.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp, format="PNG")
    tmp.close()
    return _ask_vlm_impl(tmp.name, query)
'''

# ---------------------------------------------------------------------------
# Body. Two perception tools (display + ask_vlm) presented symmetrically.
# Dataset-specific answer rules appended from profile at runtime.
# ---------------------------------------------------------------------------

_TASK_BODY = (
    "You are a Document Visual Question Answering agent. You answer a question about a document by "
    "writing Python code, calling vision tools iteratively, and reasoning programmatically. You are "
    "yourself a multimodal model — when you `display(image)`, you will SEE that image in your next "
    "turn. You can also delegate focused visual queries to a fresh sub-VLM via `ask_vlm(image, query)`. "
    "Choose per call.\n\n"

    "## DATA\n"
    "- `question`: The question you must answer.\n"
    "- `pages`: list of page images (PIL Images, 0-indexed). Pass them to tool calls.\n\n"

    "## TOOLS\n"
    "- display(image)\n"
    "  What: show the image to yourself; you will see it inline in your next turn.\n"
    "  When: when you want to look directly — coarse layout, reading a region you can interpret yourself, anything where you'd benefit from seeing context the same time you reason about it.\n"
    "  How: pass any PIL Image — a page (`pages[i]`) or a crop (`pages[i].crop((left, top, right, bottom))`). Full pages are downscaled — for fine details, crop first.\n"
    "- ask_vlm(image, query) -> str\n"
    "  What: send the image and a focused query to a fresh, stateless sub-VLM call; returns its text answer.\n"
    "  When: when you want to delegate a focused factual extraction — reading a specific value, scanning for a label, OCR-style reads. The sub-VLM has no memory of your conversation; ask one thing at a time.\n"
    "  How: any PIL Image plus a short, specific query. For multiple queries, just call multiple times.\n"
    "- SUBMIT(answer=\"...\")\n"
    "  What: deliver the final answer and terminate.\n"
    "  When: you have the answer and have verified it.\n\n"

    "## APPROACH\n"
    "1. SURVEY — read the document at a coarse level to build a mental map. "
    "`display(pages[i])` to see layout; for many-page docs, `ask_vlm(pages[i], 'summarize')` over a sample.\n"
    "2. LOCATE — identify the page(s) and region(s) that contain the answer.\n"
    "3. EXTRACT — get the values out. Either `display()` a tight crop and read yourself, or "
    "`ask_vlm()` with a focused query. Whichever fits the question shape better.\n"
    "4. VERIFY — for any precise value (numbers, fine text, small labels), do not commit a reading "
    "you've only seen once. Design a check: re-read with a different crop or query, look for consistency "
    "across reads, or cross-reference an adjacent label.\n"
    "5. SUBMIT — call `SUBMIT(answer=\"...\")` once you have the answer.\n\n"

    "Never use outside or world knowledge. Every answer must come from the document.\n\n"

    "## VERIFICATION UNDER PERCEPTION STOCHASTICITY\n"
    "Both your own visual perception and the sub-VLM are non-deterministic — the same image and query "
    "can return different readings, especially for precise values (numbers, fine text, small labels) and "
    "high-density images. A single read is not trustworthy. Build a reading procedure that compensates. "
    "You have a broad palette of strategies and can combine them as the situation calls: re-read the same "
    "region multiple times and look for the consistent answer; switch between `display` and `ask_vlm`; "
    "read at multiple crop sizes or framings; rephrase the query; tile-scan a region too large for one "
    "read; cross-check against an adjacent label or value. Be aware of pitfalls — a tighter crop reads "
    "more precisely but can occlude context (a label may sit just outside the box); silently swapping a "
    "value after one re-read with no evidence is just noise.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string: `SUBMIT(answer=\"42\")`.\n"
    "- The answer must follow these formatting rules:\n\n"
)

def _build_task_instructions(profile: DatasetProfile) -> str:
    return _TASK_BODY + profile.answer_formatting_rules

SEED_TASK_INSTRUCTIONS_LENGTH = len(_TASK_BODY)

class RvlmHybridProgram:
    """Hybrid solver — MultimodalRLM with both display() and ask_vlm()."""

    def __init__(
        self,
        vlm_lm: dspy.LM,
        profile: DatasetProfile,
        max_iterations: int = 25,
        page_factor: float = 1.5,
        question_concurrency: int = 4,
        max_messages: int = 8,
        max_image_pixels: int = 8_000_000,
    ):
        self.vlm_lm = vlm_lm
        self.profile = profile
        self.max_iterations = max_iterations
        self.page_factor = page_factor
        self.question_concurrency = question_concurrency
        self.max_messages = max_messages
        self.max_image_pixels = max_image_pixels

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
            tools = _create_ask_vlm_tool(self.vlm_predict, self.vlm_lm)
            sandbox_code = _build_sandbox_code(tmpdir, len(document.images))

            def _solve_question(q: Question):
                with logfire.span(
                    "solve_rvlm_hybrid",
                    doc_id=document.doc_id,
                    question_id=q.question_id,
                    question=q.question[:200],
                    profile=self.profile.name,
                ) as q_span:
                    question_text = q.question + self._per_question_prefix(q)
                    mm_rlm = MultimodalRLM(
                        signature=_build_signature(instructions),
                        max_iterations=max_iter,
                        max_llm_calls=max_iter * 3,
                        tools=tools,
                        verbose=True,
                        sandbox_code=sandbox_code,
                        max_messages=self.max_messages,
                        max_image_pixels=self.max_image_pixels,
                    )
                    logger.info(
                        "RVLM-HYB [%s] Q %s: max_iterations=%d (page_bonus=%d)",
                        self.profile.name, q.question_id, max_iter, int(page_bonus),
                    )

                    @retry(
                        retry=retry_if_exception(is_retryable_lm_error),
                        stop=stop_after_attempt(4),
                        wait=wait_exponential(multiplier=30, min=30, max=120),
                        before_sleep=lambda rs: logger.warning(
                            "Retryable error, retry %d in %.0fs: %s",
                            rs.attempt_number,
                            rs.next_action.sleep,  # type: ignore[union-attr]
                            rs.outcome.exception() if rs.outcome else "?",  # type: ignore[union-attr]
                        ),
                        reraise=True,
                    )
                    def _solve_one():
                        return mm_rlm(question=question_text, doc_info=doc_info)

                    try:
                        result = _solve_one()
                        answer = str(result.answer or "").strip()
                        trajectory = result.trajectory
                    except Exception as e:
                        logger.warning("RVLM-HYB failed for Q '%s': %s", q.question_id, e)
                        answer = "Unknown"
                        trajectory = []

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
                            "RVLM-HYB[%s] Q %s: %s (GT=%s, PRED=%s)",
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
                logger.info("RVLM-HYB: running %d questions with concurrency=%d", len(document.questions), max_w)
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
                    "RVLM-HYB [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name, document.doc_id, correct, scored,
                    100 * correct / scored,
                )

            return predictions, trajectories

def create_rvlm_hybrid_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 25,
    vlm: dict[str, Any] | None = None,
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    max_messages: int = 8,
    max_image_pixels: int = 8_000_000,
) -> RvlmHybridProgram:
    """Hydra factory. Profile resolution: same as ``rvlm_minimal_solver``."""
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

    return RvlmHybridProgram(
        vlm_lm=vlm_lm,
        profile=profile,
        max_iterations=max_iterations,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
        max_messages=max_messages,
        max_image_pixels=max_image_pixels,
    )
