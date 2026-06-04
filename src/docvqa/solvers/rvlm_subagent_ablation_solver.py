"""RVLM general-sub-agent ablation — `batch_look` → `batch_subagent`.

Fork of :mod:`docvqa.solvers.rvlm_solver` (the proposed method) that
generalizes the recursive sub-call from a narrow **perception** tool into a
general **delegation** tool. In `rvlm`, `batch_look((image, query))` sends
an image + "what to look for" to a VLM — it is framed purely as visual
perception. Here, `batch_subagent((task, image))` delegates an arbitrary
**subtask** to a capable **multimodal sub-agent** (still a single
multimodal-model call, like rvlm's VLM): the sub-agent can see any image
the main agent passes it *and* reason over text, and the image is
**optional** (`None` for a non-visual subtask). The tool description makes
the main agent explicitly aware it can delegate any well-scoped subtask —
read/describe an image, extract & structure data, summarize a span,
perform a focused reasoning/extraction step — not only perception.

The paired Δ vs `rvlm` tests whether a **general task-decomposition /
delegation** tool helps the main agent solve more (by decomposing the
problem and delegating subtasks) than the perception-specific `batch_look`
— or whether the general framing just adds overhead and the narrow
perception sub-call is what carries `rvlm`. Same LeanRLM REPL scaffold,
same underlying model; only the sub-call's interface + framing change.
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
from docvqa.solvers.rvlm_solver import _build_signature

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool: batch_subagent — delegate subtasks to a multimodal sub-agent.
# Each request is (task, image|None): task is an instruction string, image is
# optional visual context. Image present -> multimodal call; absent -> text call.
# ---------------------------------------------------------------------------

_SUBAGENT_INSTR = (
    "You are a capable multimodal sub-agent invoked to complete ONE delegated subtask. "
    "Use only the information provided — the image(s) below (if any) and the task. "
    "Transcribe numbers and characters exactly. When a label is separated from the item it "
    "identifies, trace any visual connector (leader line, arrow, callout, alignment) to "
    "determine which item it refers to. Return ONLY the result of the task. If the information "
    "needed is missing, say so."
)


def _create_tools(sub_lm: dspy.LM, batch_concurrency: int = 8) -> list:
    """One tool: batch_subagent. Each request delegates a task to the multimodal sub-agent
    with 0..N images (a single image, a list — e.g. a page range — or none)."""
    from PIL import Image as PILImage

    def _one(task: str, paths: list[str]) -> str:
        with logfire.span("subagent", task=task[:300], n_images=len(paths)) as span:
            parts: list[dict] = [{"type": "text", "text": _SUBAGENT_INSTR + "\n"}]
            for i, pth in enumerate(paths):
                if len(paths) > 1:
                    parts.append({"type": "text", "text": f"\n[Image {i}]"})
                formatted = dspy.Image(PILImage.open(pth)).format()
                if isinstance(formatted, list):
                    parts.extend(formatted)
                else:
                    parts.append({"type": "image_url", "image_url": {"url": formatted}})
            parts.append({"type": "text", "text": f"\nTask: {task}\n\nResult:"})
            messages = [{"role": "user", "content": parts}]
            with dspy.context(lm=sub_lm):
                response = sub_lm.forward(messages=messages)
            msg = response.choices[0].message
            text = msg.content or getattr(msg, "reasoning_content", "") or ""
            out = str(text or "").strip()
            span.set_attribute("result", out[:2000])
            return out

    def _batch_subagent_impl(requests_json: str) -> list[str]:
        """Internal: parallel sub-agent calls. Input JSON list of {task, paths:[...]}."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import json as _json
        requests = _json.loads(requests_json)
        if not requests:
            return []
        results: list[str] = [""] * len(requests)
        is_vertex = "vertex_ai" in (sub_lm.model if hasattr(sub_lm, "model") else str(sub_lm))
        max_w = min(len(requests), 2 if is_vertex else batch_concurrency)
        with logfire.span("batch_subagent", num_requests=len(requests)):
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                futures = {
                    pool.submit(_one, r["task"], r.get("paths", [])): i
                    for i, r in enumerate(requests)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
        return results

    return [_batch_subagent_impl]


def _build_sandbox_code(page_dir: str, num_pages: int) -> str:
    """Sandbox loads pages and defines `batch_subagent()` (no `look`)."""
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
assert len(pages) == {num_pages}, f"Expected {{num_pages}} pages, got {{len(pages)}}"

def batch_subagent(requests):
    """Delegate one or more subtasks to a multimodal sub-agent, in parallel.
    Each request is (task, images): `task` is a natural-language instruction (str);
    `images` is the visual context the subtask needs and may be:
      - a single PIL Image  — one page (pages[i]) or a crop (pages[i].crop((l,t,r,b)))
      - a LIST of PIL Images — e.g. a range of pages: pages[3:8], or [pages[0], pages[4]]
      - None                — for a non-visual subtask (text / reasoning / computation)
    The sub-agent is a multimodal model: it sees all images you pass and can also reason
    over text. Returns: list of str results, same order. Single task: batch_subagent([(task, imgs)])[0].
    Examples:
      batch_subagent([("read the total revenue", pages[3])])
      batch_subagent([("which of these pages contains the revenue table? give the page index", pages[3:8])])
      batch_subagent([("normalize this date to YYYY-MM-DD: 'Jan 1st 24'", None)])"""
    import json as _json
    items = []
    for req in requests:
        if isinstance(req, (tuple, list)):
            task = req[0]
            images = req[1] if len(req) > 1 else None
        else:
            task = req
            images = None
        if images is None:
            images = []
        elif not isinstance(images, (list, tuple)):
            images = [images]
        paths = []
        for im in images:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            im.save(tmp, format="PNG")
            tmp.close()
            paths.append(tmp.name)
        items.append({{"task": str(task), "paths": paths}})
    return _batch_subagent_impl(_json.dumps(items))
'''


# ---------------------------------------------------------------------------
# Task body: rvlm's, with batch_look reframed as general delegation.
# ---------------------------------------------------------------------------

_TASK_BODY = (
    "You are a Document Visual Question Answering agent. You answer a question about a document by "
    "writing Python code, delegating subtasks to a sub-agent, and reasoning programmatically.\n\n"

    "## DATA\n"
    "- `question`: The question you must answer.\n"
    "- `pages`: list of page images (PIL Images, 0-indexed). Pass any of them (or a crop) to the "
    "sub-agent when a subtask needs visual context.\n\n"

    "## TOOLS\n"
    "- batch_subagent(requests) -> list[str]\n"
    "  What: delegate one or more SUBTASKS to a capable multimodal sub-agent, run in parallel. "
    "The sub-agent is a multimodal model: it can see an image when you provide one, and it can "
    "reason over text when you do not. A subtask can be visual or not — whatever decomposing the "
    "problem calls for. Some subtasks need image(s) (e.g. read a value from a region, describe a "
    "figure, find which of a range of pages contains something); others do not (e.g. compare or "
    "normalize values you already have, reformat an answer, work through a calculation, summarize "
    "text you pass in). Delegate whatever is well-scoped, and combine the results in Python.\n"
    "  How: list of (task, images) tuples. `task` is a natural-language instruction (string); "
    "`images` is the visual context the subtask needs — a single image (a page `pages[i]` or a "
    "crop `pages[i].crop((left, top, right, bottom))`), a LIST of images (e.g. a page range "
    "`pages[3:8]`), or `None` if the subtask is non-visual. "
    "Returns results in the same order. For a single subtask: `batch_subagent([(task, images)])[0]`.\n"
    "- SUBMIT(answer=\"...\")\n"
    "  What: deliver the final answer and terminate.\n"
    "  When: you have the answer and have verified it.\n\n"

    "## APPROACH\n"
    "1. UNDERSTAND — read the question and work out which subtasks would answer it.\n"
    "2. DECOMPOSE & DELEGATE — break the problem into well-scoped subtasks and delegate each to "
    "the sub-agent. Give a subtask an image when it needs to look at something; leave the image "
    "out when it is a text or reasoning step. Batch independent subtasks into one call.\n"
    "3. COMBINE — assemble the sub-agent results in Python (count, compare, compute, select), or "
    "delegate the combination itself as a subtask.\n"
    "4. VERIFY — for any result you are unsure of, do not commit it after a single delegation. "
    "Re-delegate with a different framing (or a different image/region), look for consistency, or "
    "cross-check against related information. The sub-agent is non-deterministic, so a lone result "
    "is not trustworthy.\n"
    "5. SUBMIT — call `SUBMIT(answer=\"...\")` once you have a verified answer.\n\n"

    "Never use outside or world knowledge. Every answer must come from the document.\n\n"

    "## GUIDANCE\n"
    "- **Scope each subtask narrowly.** A focused subtask — visual or textual — yields a more "
    "reliable result than a broad one. If a result looks off, re-scope and re-delegate.\n"
    "- **Verify precise values.** Numbers, fine text, and small labels are where results are least "
    "reliable; confirm them with a second, differently-framed delegation before committing.\n"
    "- **Many-part questions** (\"how many...\", \"which is largest...\", \"list all...\"): gather "
    "ALL the parts first (do not stop at the first one), then combine or compare them — in Python "
    "or via a delegated reasoning subtask.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string: `SUBMIT(answer=\"42\")`.\n"
    "- The answer must follow these formatting rules:\n\n"
)


def _build_task_instructions(profile: DatasetProfile) -> str:
    return _TASK_BODY + profile.answer_formatting_rules


class RvlmSubagentAblationProgram:
    """RVLM general-sub-agent ablation — `batch_subagent`. See module docstring."""

    def __init__(
        self,
        sub_lm: dspy.LM,
        profile: DatasetProfile,
        max_iterations: int = 25,
        rlm_type: str = "lean",
        page_factor: float = 1.5,
        question_concurrency: int = 4,
        batch_concurrency: int = 8,
    ):
        self.sub_lm = sub_lm
        self.profile = profile
        self.max_iterations = max_iterations
        self.rlm_type = rlm_type
        self.page_factor = page_factor
        self.question_concurrency = question_concurrency
        self.batch_concurrency = batch_concurrency
        # The sub-agent is a raw multimodal `sub_lm.forward(messages=...)` call (built in
        # `_create_tools`), so it can take 0..N images per delegated subtask.

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
            tools = _create_tools(self.sub_lm, self.batch_concurrency)
            sandbox_code = _build_sandbox_code(tmpdir, len(document.images))

            def _solve_question(q: Question):
                with logfire.span(
                    "solve_rvlm_subagent",
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
                        "RVLM-SUBAGENT [%s] (%s) Q %s: max_iterations=%d (page_bonus=%d)",
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
                            "RVLM-SUBAGENT[%s] Q %s: %s (GT=%s, PRED=%s)",
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
                logger.info("RVLM-SUBAGENT: running %d questions with concurrency=%d", len(document.questions), max_w)
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
                    "RVLM-SUBAGENT [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name, document.doc_id, correct, scored,
                    100 * correct / scored,
                )

            return predictions, trajectories


def create_rvlm_subagent_ablation_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 25,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    batch_concurrency: int = 8,
) -> RvlmSubagentAblationProgram:
    """Hydra factory. The sub-agent runs on the `vlm` model (a multimodal model)."""
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

    sub_lm = vlm_config.to_dspy_lm()

    return RvlmSubagentAblationProgram(
        sub_lm=sub_lm,
        profile=profile,
        max_iterations=max_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
        batch_concurrency=batch_concurrency,
    )
