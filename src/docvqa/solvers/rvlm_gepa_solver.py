"""GEPA-friendly variant of ``rvlm_unified_solver``.

One injectable string: ``task_instructions`` (the merged TASK_BODY +
UNIFIED_TIPS blob the agent sees). The dataset-specific
``profile.answer_formatting_rules`` is still appended at runtime so GEPA
cannot rewrite dataset-format conventions (avoids breaking answer
parsing downstream). The VLM ``dspy.Predict`` signature stays fixed; the
lean RLM harness prompt stays fixed.

Behavior with no candidate: identical to ``RvlmUnifiedProgram`` (seed =
``_TASK_BODY + _UNIFIED_TIPS`` from the unified solver). Behavior with a
candidate JSON loaded: ``task_instructions`` is replaced from the JSON.

Construction modes (mirrors archived ``flat_solo_gepa_solver`` pattern):

- ``candidate_path``: JSON file from a prior ``optimize_anything`` run
  (``output/optim/<run>/best_candidate.json``). Loaded once at factory.
- ``task_instructions``: direct string override (used by the optimizer's
  in-process evaluator via ``apply_candidate``).
- Neither set: uses the seed (identical to ``rvlm_unified``).
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
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from docvqa.data import Document, Question
from docvqa.datasets.profile import DatasetProfile, get_profile
from docvqa.rlm import LeanRLM, CodeRLM, ThinkingRLM, RLM
from docvqa.solvers.rvlm_unified_solver import (
    _TASK_BODY,
    _UNIFIED_TIPS,
    _build_signature,
    _create_tools,
    _build_sandbox_code,
)
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed candidate — what the optimizer starts from.
# Identical to ``rvlm_unified``'s full agent prompt minus the
# dataset-specific ``answer_formatting_rules`` (which stays profile-injected
# and is NOT optimized).
# ---------------------------------------------------------------------------

SEED_TASK_INSTRUCTIONS: str = _TASK_BODY + "\n" + _UNIFIED_TIPS


class RvlmGepaProgram:
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


def create_rvlm_gepa_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 25,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    batch_concurrency: int = 8,
    candidate_path: str | None = None,
) -> RvlmGepaProgram:
    """Hydra factory.

    Profile resolution: same as ``rvlm_unified_solver``. If
    ``candidate_path`` is set, loads ``{"task_instructions": str}`` JSON
    and applies it on top of the seed. Otherwise behavior matches
    ``rvlm_unified``.
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

    program = RvlmGepaProgram(
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
