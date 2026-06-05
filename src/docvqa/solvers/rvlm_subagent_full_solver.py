"""RVLM full-agent sub-agent variant — `batch_subagent` delegates to a FULL agent.

Fork of :mod:`docvqa.solvers.rvlm_subagent_ablation_solver`. There the
sub-call is a **single multimodal-model forward pass** (`sub_lm.forward(...)`).
Here each delegated subtask is handed to a **full-fledged LeanRLM sub-agent**
that has its *own* Python REPL and its *own* recursive `batch_look` over the
images the main agent passes it — so the sub-agent can crop/zoom/re-read/
compute on its subtask, not just answer in one shot.

This climbs one rung on the sub-call-complexity ladder:

    rvlm           : sub-call = single VLM forward (perception only)
    subagent       : sub-call = single multimodal forward (any task, 1 shot)
    subagent_full  : sub-call = a full agent loop (own batch_look, own REPL)  <-- this

**Recursion depth is hard-capped at 1 by construction:** the sub-agent's
toolset is `batch_look` ONLY (rvlm's perception tool) — it has no
`batch_subagent`, so it cannot spawn sub-sub-agents. The sub-agent runs on a
small iteration budget (`subagent_max_iterations`, default 6) to bound cost,
since the main agent issues many delegations and each is now a whole loop.

Paired Δ vs `rvlm` / `subagent` asks: does giving the sub-call full agency
(iterative perception on its subtask) beat a single focused VLM forward — or
does a single forward already capture the benefit (bounding the necessary
sub-call complexity, which would strengthen minimal-`rvlm`-as-method)? On the
DocVQA-2026 val set the main agent delegates almost exclusively *perception*
subtasks, so in practice this measures **recursive perception depth-2**, not
hierarchical reasoning decomposition.
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
from docvqa.solvers.rvlm_solver import (
    _build_signature,
    _create_tools as _rvlm_batch_look_tools,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inner sub-agent: a full LeanRLM scoped to ONE delegated subtask, with its
# own batch_look over the images the main agent passed. batch_look ONLY ->
# recursion depth is 1 (no batch_subagent in scope).
# ---------------------------------------------------------------------------

_SUBAGENT_RLM_INSTR = (
    "You are a sub-agent completing ONE focused subtask delegated by a main agent. "
    "You answer by writing Python, perceiving images with `batch_look`, and reasoning "
    "programmatically.\n\n"
    "## DATA\n"
    "- `question`: the subtask you must complete.\n"
    "- `pages`: list of page images (PIL Images, 0-indexed) — the visual context for this "
    "subtask. May be empty for a non-visual subtask.\n\n"
    "## TOOLS\n"
    "- batch_look(requests) -> list[str]\n"
    "  What: send one or more images to a VLM in parallel.\n"
    "  How: list of (image, query) tuples. Image is any PIL Image — a page (`pages[i]`) or a "
    "crop (`pages[i].crop((left, top, right, bottom))`). Returns answers in the same order. "
    "For a single query: `batch_look([(image, query)])[0]`.\n"
    "- SUBMIT(answer=\"...\")\n"
    "  What: return the result of the subtask and terminate.\n\n"
    "## APPROACH\n"
    "Perceive what the subtask needs — survey, then crop tight for fine detail. Transcribe "
    "numbers and characters exactly. The VLM is non-deterministic: for any precise value "
    "(numbers, fine text, small labels), do not trust a single read — re-read with a different "
    "crop/query and look for consistency before committing. Reason over the reads in Python, "
    "then SUBMIT. If the needed information is missing, SUBMIT 'Unknown'. Use ONLY the provided "
    "images — never outside or world knowledge.\n\n"
    "## OUTPUT\n"
    "SUBMIT the result of the subtask as a concise string (the main agent will combine it): "
    "`SUBMIT(answer=\"...\")`."
)


def _build_inner_sandbox(paths: list[str]) -> str:
    """Sandbox for the sub-agent: load the passed images as `pages`, define `batch_look`."""
    return f'''
import os
import tempfile
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000
_paths = {paths!r}
pages = [Image.open(p) for p in _paths]

def batch_look(requests):
    """Send multiple images to the VLM in parallel.
    Input: list of (image, query) tuples. Returns: list of str answers (same order).
    Example: batch_look([(pages[0], "layout?"), (pages[0].crop((0,0,500,500)), "read text")])"""
    import json as _json
    paths = []
    for image, query in requests:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp, format="PNG")
        tmp.close()
        paths.append({{"path": tmp.name, "query": query}})
    return _batch_look_impl(_json.dumps(paths))
'''


def _create_tools(
    vlm_predict: dspy.Predict,
    vlm_lm: dspy.LM,
    subagent_max_iterations: int = 6,
    rlm_type: str = "lean",
    batch_concurrency: int = 8,
) -> list:
    """One tool: batch_subagent. Each request spins up a FULL LeanRLM sub-agent (own
    batch_look over the passed images), depth-capped at 1 (no nested batch_subagent)."""
    # Shared batch_look impl for every inner sub-agent (one VLM predict, one LM).
    batch_look_tools = _rvlm_batch_look_tools(vlm_predict, vlm_lm, batch_concurrency)
    RLMClass = {"code": CodeRLM, "lean": LeanRLM, "thinking": ThinkingRLM}.get(rlm_type, LeanRLM)

    def _one(task: str, paths: list[str]) -> str:
        with logfire.span("subagent_full", task=task[:300], n_images=len(paths)) as span:
            rlm = RLMClass(
                signature=_build_signature(_SUBAGENT_RLM_INSTR),
                max_iterations=subagent_max_iterations,
                max_llm_calls=subagent_max_iterations * 3,
                tools=batch_look_tools,
                verbose=False,
                sandbox_code=_build_inner_sandbox(paths),
            )
            doc_info = f"{len(paths)} image(s) provided" if paths else "No images (text-only subtask)"
            result = rlm(question=str(task), doc_info=doc_info)
            out = str(result.answer or "").strip()
            span.set_attribute("sub_iterations", len(result.trajectory))
            span.set_attribute("result", out[:2000])
            return out or "Unknown"

    def _batch_subagent_impl(requests_json: str) -> list[str]:
        """Internal: parallel full-agent sub-calls. Input JSON list of {task, paths:[...]}."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import json as _json
        requests = _json.loads(requests_json)
        if not requests:
            return []
        results: list[str] = [""] * len(requests)
        is_vertex = "vertex_ai" in (vlm_lm.model if hasattr(vlm_lm, "model") else str(vlm_lm))
        max_w = min(len(requests), 2 if is_vertex else batch_concurrency)
        with logfire.span("batch_subagent_full", num_requests=len(requests)):
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
    """Main-agent sandbox: load pages, define `batch_subagent` (delegates to full sub-agents)."""
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
    """Delegate a subtask to a FULL sub-agent. Each sub-agent runs its own multi-step loop
    (survey/crop/zoom/re-read/verify) over the images you pass — so each delegation is EXPENSIVE.
    Hand it a substantial, self-contained subtask; do NOT fan out many sub-agents in parallel
    (one at a time, or a small batch of <=2-3 genuinely-independent subtasks).
    Each request is (task, images): `task` is a natural-language instruction (str);
    `images` is the visual context the subtask needs and may be:
      - a single PIL Image  — one page (pages[i]) or a crop (pages[i].crop((l,t,r,b)))
      - a LIST of PIL Images — e.g. a range of pages: pages[3:8], or [pages[0], pages[4]]
      - None                — for a non-visual subtask (text / reasoning / computation)
    Returns: list of str results, same order. Single task: batch_subagent([(task, imgs)])[0].
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
# Main-agent task body: identical general-delegation framing as the subagent
# ablation (the only thing that changes vs that solver is what the sub-agent IS).
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
    "  What: delegate a SUBTASK to a capable sub-agent. **The sub-agent is a full agent, not a "
    "single look** — given an image it runs its OWN multi-step loop (survey, crop, zoom, re-read, "
    "verify) to resolve the subtask, and it can also reason over text. **Each delegation is "
    "therefore expensive — a whole agent run — so delegate deliberately, not reflexively.**\n"
    "  How to use it well:\n"
    "    * Hand the sub-agent a SUBSTANTIAL, self-contained subtask and let it do the iterative "
    "looking internally — e.g. \"find and read the total revenue figure, verifying it\" — rather "
    "than driving many tiny one-look reads yourself. One capable delegation replaces several "
    "micro-calls.\n"
    "    * **Do NOT fan out many sub-agents in parallel.** Prefer ONE delegation at a time; only "
    "batch a SMALL number (≤2-3) when they are genuinely independent and each is worth a full "
    "agent run. A long list of parallel sub-agents is wasteful and slow — that is not what this "
    "tool is for.\n"
    "    * A subtask can be visual or not. Visual: read/verify a value from a region, describe a "
    "figure, locate which page holds something. Non-visual: compare or normalize values you "
    "already have, reformat, compute, summarize text you pass in.\n"
    "  Signature: list of (task, images) tuples. `task` is a natural-language instruction (string); "
    "`images` is the visual context the subtask needs — a single image (a page `pages[i]` or a "
    "crop `pages[i].crop((left, top, right, bottom))`), a LIST of images (e.g. a page range "
    "`pages[3:8]`), or `None` if the subtask is non-visual. "
    "Returns results in the same order. For a single subtask: `batch_subagent([(task, images)])[0]`.\n"
    "- SUBMIT(answer=\"...\")\n"
    "  What: deliver the final answer and terminate.\n"
    "  When: you have the answer and have verified it.\n\n"

    "## APPROACH\n"
    "1. UNDERSTAND — read the question and work out the few substantial subtasks that would answer it.\n"
    "2. DELEGATE DELIBERATELY — hand each substantial subtask to the sub-agent, one (or a small "
    "few) at a time, and let it do its own iterative looking/verification internally. Give it an "
    "image when it needs to look; leave the image out for a text or reasoning step. Do not spawn a "
    "swarm of tiny parallel sub-agents.\n"
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
    "all the parts (do not stop at the first one), then combine or compare them. Prefer delegating "
    "the enumeration as ONE substantial subtask (\"find and list all X across these pages\") over "
    "spawning a separate sub-agent per candidate.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string: `SUBMIT(answer=\"42\")`.\n"
    "- The answer must follow these formatting rules:\n\n"
)


def _build_task_instructions(profile: DatasetProfile) -> str:
    return _TASK_BODY + profile.answer_formatting_rules


class RvlmSubagentFullProgram:
    """RVLM full-agent sub-agent variant — `batch_subagent` delegates to a full LeanRLM. See module docstring."""

    def __init__(
        self,
        vlm_lm: dspy.LM,
        profile: DatasetProfile,
        max_iterations: int = 25,
        subagent_max_iterations: int = 6,
        rlm_type: str = "lean",
        subagent_rlm_type: str = "lean",
        page_factor: float = 1.5,
        question_concurrency: int = 4,
        batch_concurrency: int = 8,
    ):
        self.vlm_lm = vlm_lm
        self.profile = profile
        self.max_iterations = max_iterations
        self.subagent_max_iterations = subagent_max_iterations
        self.rlm_type = rlm_type
        self.subagent_rlm_type = subagent_rlm_type
        self.page_factor = page_factor
        self.question_concurrency = question_concurrency
        self.batch_concurrency = batch_concurrency

        # One VLM predict shared by every inner sub-agent's batch_look.
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
            tools = _create_tools(
                self.vlm_predict,
                self.vlm_lm,
                subagent_max_iterations=self.subagent_max_iterations,
                rlm_type=self.subagent_rlm_type,
                batch_concurrency=self.batch_concurrency,
            )
            sandbox_code = _build_sandbox_code(tmpdir, len(document.images))

            def _solve_question(q: Question):
                with logfire.span(
                    "solve_rvlm_subagent_full",
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
                        "RVLM-SUBAGENT-FULL [%s] (%s) Q %s: max_iterations=%d (page_bonus=%d), sub_budget=%d",
                        self.profile.name, self.rlm_type, q.question_id, max_iter, int(page_bonus),
                        self.subagent_max_iterations,
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
                            "RVLM-SUBAGENT-FULL[%s] Q %s: %s (GT=%s, PRED=%s)",
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
                logger.info("RVLM-SUBAGENT-FULL: running %d questions with concurrency=%d", len(document.questions), max_w)
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
                    "RVLM-SUBAGENT-FULL [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name, document.doc_id, correct, scored,
                    100 * correct / scored,
                )

            return predictions, trajectories


def create_rvlm_subagent_full_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 25,
    subagent_max_iterations: int = 6,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "lean",
    subagent_rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    batch_concurrency: int = 8,
) -> RvlmSubagentFullProgram:
    """Hydra factory. Both the main agent's perception and the sub-agents run on the `vlm` model."""
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

    return RvlmSubagentFullProgram(
        vlm_lm=vlm_lm,
        profile=profile,
        max_iterations=max_iterations,
        subagent_max_iterations=subagent_max_iterations,
        rlm_type=rlm_type,
        subagent_rlm_type=subagent_rlm_type,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
        batch_concurrency=batch_concurrency,
    )
