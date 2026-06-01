"""GEPA-friendly recursive-VLM solver with a heavy (category-tips) seed prompt.

The seed prompt is self-contained here (formerly imported from the deleted
heavy-prompt unified solver; see the inlined ``_TASK_BODY`` /
``_UNIFIED_TIPS`` below). Separate research line from the proposed-method
``rvlm`` solver, kept for prompt optimization.

One injectable string: ``task_instructions`` (the merged TASK_BODY +
UNIFIED_TIPS blob the agent sees). The dataset-specific
``profile.answer_formatting_rules`` is still appended at runtime so GEPA
cannot rewrite dataset-format conventions (avoids breaking answer
parsing downstream). The VLM ``dspy.Predict`` signature stays fixed; the
lean RLM harness prompt stays fixed.

Behavior with no candidate: identical to the heavy-prompt seed (=
``_TASK_BODY + _UNIFIED_TIPS``). Behavior with a
candidate JSON loaded: ``task_instructions`` is replaced from the JSON.

Construction modes (mirrors archived ``flat_solo_gepa_solver`` pattern):

- ``candidate_path``: JSON file from a prior ``optimize_anything`` run
  (``output/optim/<run>/best_candidate.json``). Loaded once at factory.
- ``task_instructions``: direct string override (used by the optimizer's
  in-process evaluator via ``apply_candidate``).
- Neither set: uses the heavy-prompt seed.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import dspy
import logfire

from docvqa.data import Document, Question
from docvqa.datasets.profile import (
    DatasetProfile,
    get_profile,
    _DOCVQA_2026_CATEGORY_TIPS,
)
from docvqa.rlm import LeanRLM, CodeRLM, ThinkingRLM, RLM
from docvqa.solvers.rvlm_solver import (
    _build_signature,
    _create_tools,
    _build_sandbox_code,
)
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heavy-prompt seed (inlined from the former unified solver, which
# was deleted in the D-010 solver cleanup). GEPA optimizes from this seed:
# the full category-tips agent prompt minus the dataset-specific
# ``answer_formatting_rules`` (which stays profile-injected and is NOT
# optimized). Kept self-contained here so this research line is decoupled
# from the proposed-method ``rvlm`` solver's minimal prompt.
# ---------------------------------------------------------------------------

def _build_unified_category_tips() -> str:
    parts: list[str] = [
        "",
        "## CATEGORY-SPECIFIC TIPS",
        "",
        "The document's category will be one of: business_report, comics, "
        "engineering_drawing, infographics, maps, science_paper, "
        "science_poster, slide. Tips for each category are given below; "
        "apply the ones that match this document and ignore the rest.",
        "",
    ]
    for cat in sorted(_DOCVQA_2026_CATEGORY_TIPS):
        body = _DOCVQA_2026_CATEGORY_TIPS[cat].strip()
        parts.append(f"### {cat}")
        parts.append(body)
        parts.append("")
    return "\n".join(parts)

_UNIFIED_TIPS = _build_unified_category_tips()

_TASK_BODY = (
    "You are a Document Visual Question Answering agent. You answer a question about a document by "
    "writing Python code, calling vision tools iteratively, and reasoning programmatically.\n\n"

    "## DATA\n"
    "- `question`: The question you must answer.\n"
    "- `pages`: list of page images (PIL Images) (0-indexed). Pass to `batch_look()`, e.g. `batch_look([(pages[0], 'describe layout')])[0]`.\n\n"

    "## TOOLS\n"
    "- batch_look(requests) -> list[str]: Send one or more images to the VLM in parallel. "
    "Input: list of (image, query) tuples where image is any PIL Image "
    "(a page from `pages`, a crop via `pages[i].crop((l,t,r,b))`, etc). "
    "Returns: list of answers in same order. ALL visual queries go through this tool.\n"
    "  Example: batch_look([(pages[0], 'describe layout'), (pages[0].crop((0,0,500,500)), 'read text')])\n"
    "  For a single query, use: batch_look([(image, query)])[0]\n\n"

    "## APPROACH\n"
    "1. EXPLORE: Before answering, understand the document structure. "
    "Use `batch_look` to survey pages — "
    "e.g. `batch_look([(pages[0], 'Describe layout...'), (pages[1], 'Describe layout...')])`.\n"
    "Build a mental map of the document.\n"
    "2. LOCATE: Find the specific region(s) relevant to the question.\n"
    "3. EXTRACT: Use `batch_look` with tight crops to read exact values. "
    "For fine details, crop first: `batch_look([(pages[i].crop((l,t,r,b)), query)])[0]`.\n"
    "4. VERIFY: Cross-check extracted values if ambiguous.\n"
    "5. SUBMIT: Once you have the answer, SUBMIT it.\n\n"

    "## GUIDELINES\n"
    "- Full-page batch_look gives a broad overview. For fine details, crop first: `batch_look([(pages[i].crop((l,t,r,b)), query)])[0]`.\n"
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

SEED_TASK_INSTRUCTIONS: str = _TASK_BODY + "\n" + _UNIFIED_TIPS

class RvlmGepaAblationProgram:
    """RVLM solver with one optimizable prompt component (``task_instructions``).

    See module docstring.
    """

    def __init__(
        self,
        vlm_lm: dspy.LM,
        profile: DatasetProfile,
        max_iterations: int = 25,
        rlm_type: str = "lean",
        page_factor: float = 1.5,
        question_concurrency: int = 4,
        batch_concurrency: int = 8,
        task_instructions: str | None = None,
    ):
        self.vlm_lm = vlm_lm
        self.profile = profile
        self.max_iterations = max_iterations
        self.rlm_type = rlm_type
        self.page_factor = page_factor
        self.question_concurrency = question_concurrency
        self.batch_concurrency = batch_concurrency
        self.task_instructions: str = (
            task_instructions if task_instructions is not None else SEED_TASK_INSTRUCTIONS
        )

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

    def apply_candidate(self, candidate: dict[str, str]) -> None:
        """Replace optimizable strings in-place from a GEPA candidate dict.

        Only ``task_instructions`` is supported; any other keys are ignored
        (so a 1-component candidate stays forward-compatible if more
        components are exposed later).
        """
        if "task_instructions" in candidate:
            self.task_instructions = candidate["task_instructions"]

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

            # Compose the agent prompt: optimizable task_instructions +
            # dataset-specific answer_formatting_rules (not optimized).
            instructions = self.task_instructions + "\n" + self.profile.answer_formatting_rules
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
                        "RVLM-GEPA [%s] (%s) Q %s: max_iterations=%d (page_bonus=%d)",
                        self.profile.name, self.rlm_type, q.question_id, max_iter, int(page_bonus),
                    )

                    def _solve_one():
                        return rlm(question=question_text, doc_info=doc_info)

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
                            "RVLM-GEPA[%s] Q %s: %s (GT=%s, PRED=%s)",
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
                logger.info("RVLM-GEPA: running %d questions with concurrency=%d", len(document.questions), max_w)
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
                    "RVLM-GEPA [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name, document.doc_id, correct, scored,
                    100 * correct / scored,
                )

            return predictions, trajectories

def create_rvlm_gepa_ablation_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 25,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    batch_concurrency: int = 8,
    candidate_path: str | None = None,
) -> RvlmGepaAblationProgram:
    """Hydra factory.

    Profile resolution: explicit profile_name, else dataset, else DocVQA-2026. If
    ``candidate_path`` is set, loads ``{"task_instructions": str}`` JSON
    and applies it on top of the seed. Otherwise behavior matches the seed.
    """
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

    program = RvlmGepaAblationProgram(
        vlm_lm=vlm_lm,
        profile=profile,
        max_iterations=max_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
        batch_concurrency=batch_concurrency,
    )

    if candidate_path:
        path = Path(candidate_path)
        if not path.exists():
            raise FileNotFoundError(f"candidate_path does not exist: {candidate_path}")
        candidate = json.loads(path.read_text())
        if "task_instructions" not in candidate:
            raise ValueError(
                f"candidate JSON at {candidate_path} missing 'task_instructions' key; "
                f"found keys: {list(candidate)}"
            )
        program.apply_candidate(candidate)
        logger.info(
            "Loaded GEPA candidate from %s (task_instructions: %d chars)",
            candidate_path, len(candidate["task_instructions"]),
        )

    return program
