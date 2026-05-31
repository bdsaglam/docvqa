"""Proposed method: recursive VLM (RVLM) with a minimal, dataset-agnostic prompt.

A code-LM drives a Python REPL and calls a VLM recursively via
``batch_look`` to perceive document pages. The solver body carries NO
benchmark-specific content — earlier ``rvlm`` variants shipped a
DocVQA-2026-specific category-tips block in their system prompt (8
hand-crafted blocks: business_report, comics, engineering_drawing,
infographics, maps, science_paper, science_poster, slide), giving a
reviewer a legitimate threat to the paper's generality claim — *"the
scaffold only works because the prompt was tuned for this benchmark."*

This solver strips that benchmark-specific content from the solver
body. What remains is:

1. **Tool docs** (brief): what ``batch_look`` does, when to call it,
   how to call it; the ``SUBMIT(answer=...)`` terminal action.
2. **Approach** (universal): SURVEY → LOCATE → EXTRACT → VERIFY → SUBMIT.
3. **Document-shape guidance** (4 generic patterns, no benchmark
   category names): high-density single page, many-page document,
   counting / superlatives, VLM disagreement resolution.

Dataset-specific content (answer format rules, the "Unknown" sentinel,
the percentage-difference convention) stays in
``profile.answer_formatting_rules`` and is appended unchanged. The
solver body itself is byte-identical across DocVQA-2026, MP-DocVQA,
and MMLongBench-Doc — only the profile changes.

Scoring within trial noise of the heavy-prompt variants was direct
evidence for the D-006 hypothesis: recursive perception is the
load-bearing mechanism; the per-category tips are procedural, not
foundational.
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared RVLM helpers: signature, recursive-VLM tool, sandbox bootstrap.
# This module is the canonical home for these (formerly in the deleted
# heavy-prompt unified solver); the baseline/ablation rvlm variants
# (``rvlm_naked``, ``rvlm_skeletal``, ``rvlm_hybrid``, ``rvlm_gepa``)
# import them from here.
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


def _create_tools(vlm_predict: dspy.Predict, vlm_lm: dspy.LM, batch_concurrency: int = 8) -> list:
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

    return [_batch_look_impl]


def _build_sandbox_code(page_dir: str, num_pages: int) -> str:
    """Build sandbox code that loads pages as PIL Images and defines `batch_look()`."""
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

def batch_look(requests):
    """Send multiple images to the VLM in parallel. Much faster than sequential calls.
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
# Minimal task body: tool docs + approach + document-shape patterns.
# Zero benchmark-category names. Dataset-specific answer rules are
# appended by ``profile.answer_formatting_rules`` at runtime.
# ---------------------------------------------------------------------------

_TASK_BODY = (
    "You are a Document Visual Question Answering agent. You answer a question about a document by "
    "writing Python code, calling vision tools iteratively, and reasoning programmatically.\n\n"

    "## DATA\n"
    "- `question`: The question you must answer.\n"
    "- `pages`: list of page images (PIL Images, 0-indexed). Pass them to tool calls.\n\n"

    "## TOOLS\n"
    "- batch_look(requests) -> list[str]\n"
    "  What: send one or more images to a VLM in parallel.\n"
    "  When: any visual question — full-page survey, region crop, value read.\n"
    "  How: list of (image, query) tuples. Image is any PIL Image — a page "
    "(`pages[i]`) or a crop (`pages[i].crop((left, top, right, bottom))`). "
    "Returns answers in the same order. For a single query: "
    "`batch_look([(image, query)])[0]`.\n"
    "- SUBMIT(answer=\"...\")\n"
    "  What: deliver the final answer and terminate.\n"
    "  When: you have the answer and have verified it.\n\n"

    "## APPROACH\n"
    "1. SURVEY — read the document at a coarse level to build a mental map. "
    "Use full-page `batch_look` queries; for many-page docs, batch a sample "
    "of pages in one call.\n"
    "2. LOCATE — identify the page(s) and region(s) that contain the answer.\n"
    "3. EXTRACT — get the values out of the relevant region with `batch_look`. "
    "Ask ONE simple factual question per VLM call.\n"
    "4. VERIFY — for any precise value (numbers, fine text, small labels), "
    "do not commit a reading you've only seen once. Design a check: "
    "re-read with a different crop or query, look for consistency across "
    "reads, or cross-reference an adjacent label. See the verification "
    "guidance below.\n"
    "5. SUBMIT — call `SUBMIT(answer=\"...\")` once you have the answer.\n\n"

    "Never use outside or world knowledge. Every answer must come from the "
    "document.\n\n"

    "## DOCUMENT-SHAPE GUIDANCE\n"
    "Apply the patterns below that match the document at hand.\n\n"

    "- **The VLM is unreliable; reliability is your job.** The underlying "
    "VLM is non-deterministic — the same image and query can return "
    "different answers across calls, especially for precise values "
    "(numbers, fine text, small labels) and high-density images. A "
    "single read is not trustworthy. Build a reading procedure that "
    "compensates. You have a broad palette of strategies and can combine "
    "them as the situation calls: read the same region multiple times "
    "and look for the consistent answer; read at multiple crop sizes or "
    "framings; rephrase the query; tile-scan a region too large for one "
    "read; cross-check against an adjacent label or value. Be aware of "
    "pitfalls — a tighter crop reads more precisely but can occlude "
    "context (a label may sit just outside the box); silently swapping "
    "a value after one re-read with no evidence is just noise.\n\n"

    "- **High-density single page** (large image, lots of detail per "
    "page): a single full-page `batch_look` will miss fine detail. Survey "
    "to locate regions of interest, then crop tight (~200-600px on a side) "
    "and read each crop with one focused query. Use `pages[i].size` to "
    "compute crop coordinates.\n\n"

    "- **Many-page document** (slides, papers, reports): you do NOT need to "
    "read every page. Survey in batches "
    "(`batch_look([(pages[i], 'summarize') for i in sample])`) to build a "
    "table-of-contents in your head. Then drill into the relevant section.\n\n"

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

SEED_TASK_INSTRUCTIONS_LENGTH = len(_TASK_BODY)  # for paper-quotable sizing

class RvlmProgram:
    """Minimal-prompt RVLM solver. See module docstring."""

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

            # No category tips, no per-document dispatch — the body is the body.
            instructions = _build_task_instructions(self.profile)
            tools = _create_tools(self.vlm_predict, self.vlm_lm, self.batch_concurrency)
            sandbox_code = _build_sandbox_code(tmpdir, len(document.images))

            def _solve_question(q: Question):
                with logfire.span(
                    "solve_rvlm",
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
                        "RVLM-MIN [%s] (%s) Q %s: max_iterations=%d (page_bonus=%d)",
                        self.profile.name, self.rlm_type, q.question_id, max_iter, int(page_bonus),
                    )

                    # No agent-level retry. dspy.LM retries each LLM/VLM call
                    # (lm.num_retries / vlm.num_retries) on transient errors.
                    # If a call still fails after those, the exception
                    # propagates → the question raises → the doc fails (runner
                    # returns None, not persisted → re-run on next launch).
                    # We deliberately do NOT restart the whole agent, which
                    # would discard all completed iterations and their reads.
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
                            "RVLM-MIN[%s] Q %s: %s (GT=%s, PRED=%s)",
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
                logger.info("RVLM-MIN: running %d questions with concurrency=%d", len(document.questions), max_w)
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
                    "RVLM-MIN [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name, document.doc_id, correct, scored,
                    100 * correct / scored,
                )

            return predictions, trajectories

def create_rvlm_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 25,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    batch_concurrency: int = 8,
) -> RvlmProgram:
    """Hydra factory. Profile resolution: explicit ``profile_name`` wins, else ``dataset``, else DocVQA-2026."""
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

    return RvlmProgram(
        vlm_lm=vlm_lm,
        profile=profile,
        max_iterations=max_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
        batch_concurrency=batch_concurrency,
    )
