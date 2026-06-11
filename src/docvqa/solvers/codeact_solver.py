"""CodeAct: a ReAct-style agent whose action is Python code, with an
**append-only** context.

Same task as ``rvlm`` (recursive VLM perception via ``batch_look`` in a
Python REPL) — **identical tools, names, and prompt body** (imported
from ``rvlm_solver``). The *only* difference is the agent loop:

- ``rvlm`` runs LeanRLM, which (a) feeds the agent a ``variables_info``
  sidecar describing REPL variables whose full values live *in the
  interpreter, not the context*, and (b) lets the agent compact its own
  history via ``RESET_HISTORY``. The observable context is therefore a
  managed, lossy view of a larger hidden state → effectively a POMDP.
- ``CodeAct`` here keeps a **strictly append-only** transcript: every
  ``(reasoning, code, stdout)`` step is appended and shown in full on
  the next step; there is no variable sidecar and ``RESET_HISTORY`` is
  ignored. The agent's observation is its complete history — a clean,
  fully-observable trajectory, which is the property we want for an RL
  fine-tuning target (MDP, not POMDP).

dspy ships a ``CodeAct``, but it injects tools via
``inspect.getsource`` into a Deno ``PythonInterpreter`` sandbox, which
cannot host ``batch_look`` (a host-side callback to the VLM). So we run
our own append-only loop on ``SubprocessInterpreter`` (the same
IPC-bridged REPL ``rvlm`` uses), reusing rvlm's ``batch_look`` sandbox.

Iteration budget is higher than rvlm's (default 40 vs 25): the
append-only context grows with no compaction, so the agent gets more
steps before the extract fallback fires.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from typing import Any, Callable

import dspy
import logfire
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.utils.exceptions import AdapterParseError

from docvqa.data import Document, Question
from docvqa.datasets.profile import DatasetProfile, get_profile
from docvqa.rlm.lean import _strip_code_fences
from docvqa.rlm.subprocess_interpreter import HistoryReset, SubprocessInterpreter
from docvqa.solvers.rvlm_solver import (
    _build_sandbox_code,
    _build_task_instructions,
    _create_tools,
)
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)

# Minimal REPL mechanics appended to rvlm's task body. Mirrors rvlm's
# action instructions but states the append-only contract explicitly.
_CODEACT_MECHANICS = (
    "\n\n## REPL\n"
    "You have a Python REPL. Write raw Python code (NO markdown fences); it "
    "executes and you see stdout, then you write more code based on what you "
    "learned. State persists between steps (variables you assign stay "
    "defined). ALWAYS `print()` what you want to observe. Call "
    "`SUBMIT(answer=\"...\")` to deliver the final answer and stop.\n"
    "Your full step history (reasoning + code + output) is shown below and "
    "is APPEND-ONLY — nothing is hidden, summarized, or reset, so print only "
    "what you actually need to carry forward. This is iterative: do not try "
    "to solve everything in one step. You have up to {max_iterations} steps."
)


def _format_trajectory(entries: list[dict[str, str]]) -> str:
    if not entries:
        return "(no code executed yet)"
    parts = []
    for i, e in enumerate(entries):
        parts.append(
            f"--- Step {i + 1} ---\n"
            f"Reasoning: {e['reasoning']}\n"
            f"Code:\n{e['code']}\n"
            f"Output:\n{e['output']}"
        )
    return "\n\n".join(parts)


def _build_action_signature(instructions: str, max_iterations: int) -> dspy.Signature:
    body = instructions + _CODEACT_MECHANICS.format(max_iterations=max_iterations)
    return (
        dspy.Signature({}, body)
        .append("question", dspy.InputField(desc="The question to answer about the document"), type_=str)
        .append("doc_info", dspy.InputField(desc="Document metadata: category and page count"), type_=str)
        .append("trajectory", dspy.InputField(desc="Append-only history of prior steps (reasoning + code + stdout)"), type_=str)
        .append("iteration", dspy.InputField(desc="Current step (1-indexed) out of max_iterations"), type_=str)
        .append("reasoning", dspy.OutputField(desc="What you know, what remains, and your next action"), type_=str)
        .append("code", dspy.OutputField(desc="Python code to execute next"), type_=str)
    )


def _build_extract_signature(instructions: str) -> dspy.Signature:
    extract_body = (
        "The trajectory below is your full REPL history for the objective:\n"
        + instructions
        + "\n\nExtract the final answer now from what you gathered. "
        "Output ONLY the raw answer value — no sentences, no explanation."
    )
    sig = dspy.Signature(
        {"answer": (str, dspy.OutputField(desc="The answer string for the question."))},
        extract_body,
    )
    sig = sig.prepend("trajectory", dspy.InputField(desc="Your full REPL history so far"), type_=str)
    sig = sig.prepend("doc_info", dspy.InputField(desc="Document metadata"), type_=str)
    sig = sig.prepend("question", dspy.InputField(desc="The question to answer"), type_=str)
    return sig


class CodeActAgent(dspy.Module):
    """Append-only CodeAct loop over ``SubprocessInterpreter``."""

    def __init__(
        self,
        instructions: str,
        tools: list[Callable],
        sandbox_code: str,
        max_iterations: int = 40,
        max_output_chars: int = 8000,
    ):
        super().__init__()
        self.tools = {getattr(t, "__name__", str(t)): t for t in tools}
        self.sandbox_code = sandbox_code
        self.max_iterations = max_iterations
        self.max_output_chars = max_output_chars

        # Same ChatAdapter de-indent patch LeanRLM applies, so the long
        # task body isn't 8-space-indented in the prompt.
        self._patch_chat_adapter()

        self.generate_action = dspy.Predict(_build_action_signature(instructions, max_iterations))
        self.extract = dspy.Predict(_build_extract_signature(instructions))

    @staticmethod
    def _patch_chat_adapter():
        from dspy.adapters.chat_adapter import ChatAdapter

        def format_task_description(self, signature):
            return signature.instructions or ""

        ChatAdapter.format_task_description = format_task_description

    def _truncate(self, output: str) -> str:
        if not output:
            return "(no output — did you forget to print?)"
        if len(output) > self.max_output_chars:
            return output[: self.max_output_chars] + "\n... (truncated)"
        return output

    def forward(self, question: str, doc_info: str) -> dspy.Prediction:
        repl = SubprocessInterpreter(
            tools=dict(self.tools),
            output_fields=[{"name": "answer", "type": "str"}],
            timeout=120.0,
            sandbox_code=self.sandbox_code,
        )
        entries: list[dict[str, str]] = []
        variables = {"question": question, "doc_info": doc_info}
        try:
            for i in range(self.max_iterations):
                try:
                    action = self.generate_action(
                        question=question,
                        doc_info=doc_info,
                        trajectory=_format_trajectory(entries),
                        iteration=f"{i + 1}/{self.max_iterations}",
                    )
                except AdapterParseError as e:
                    logger.warning("CodeAct step %d/%d: parse error: %s", i + 1, self.max_iterations, e)
                    entries.append({"reasoning": "", "code": "", "output": f"[Error] {e}"})
                    continue

                reasoning = getattr(action, "reasoning", "") or ""
                code = _strip_code_fences(action.code) if action.code else ""
                logger.info("CodeAct step %d/%d\nReasoning: %s\nCode:\n%s", i + 1, self.max_iterations, reasoning, code)

                if not code:
                    entries.append({"reasoning": reasoning, "code": "", "output": "[Error] No code provided. Write Python code to execute."})
                    continue

                try:
                    result = repl.execute(code, variables=variables)
                except (CodeInterpreterError, SyntaxError) as e:
                    entries.append({"reasoning": reasoning, "code": code, "output": f"[Error] {e}"})
                    continue

                # SUBMIT → FinalOutput (possibly bundled with captured stdout)
                final_output, captured = None, ""
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], FinalOutput):
                    final_output, captured = result
                elif isinstance(result, FinalOutput):
                    final_output = result

                if final_output is not None:
                    out = final_output.output
                    if isinstance(out, dict) and "answer" in out:
                        answer = str(out["answer"]).strip()
                        entries.append({"reasoning": reasoning, "code": code, "output": f"FINAL: {answer}"})
                        return dspy.Prediction(answer=answer, trajectory=entries)
                    msg = f"[Error] SUBMIT requires answer=...; got {out!r}"
                    entries.append({"reasoning": reasoning, "code": code, "output": (f"{captured}\n{msg}" if captured else msg)})
                    continue

                # Append-only: ignore RESET_HISTORY entirely.
                if isinstance(result, HistoryReset):
                    entries.append({"reasoning": reasoning, "code": code, "output": "[RESET_HISTORY ignored — context is append-only]"})
                    continue

                if isinstance(result, list):
                    output = "\n".join(map(str, result))
                elif isinstance(result, str) and result.startswith("[Error]"):
                    output = result
                else:
                    output = str(result) if result else ""
                entries.append({"reasoning": reasoning, "code": code, "output": self._truncate(output)})

            # Max steps reached without SUBMIT → extract from the transcript.
            logger.warning("CodeAct reached max iterations (%d); extracting final answer", self.max_iterations)
            ex = self.extract(question=question, doc_info=doc_info, trajectory=_format_trajectory(entries))
            return dspy.Prediction(answer=str(getattr(ex, "answer", "") or "").strip(), trajectory=entries)
        finally:
            repl.shutdown()


class CodeActProgram:
    """CodeAct solver — append-only ReAct-with-code over ``batch_look``.

    Mirrors :class:`RvlmProgram` (same per-doc / per-question structure,
    concurrency, scoring) but swaps LeanRLM for the append-only
    :class:`CodeActAgent`.
    """

    def __init__(
        self,
        vlm_lm: dspy.LM,
        profile: DatasetProfile,
        max_iterations: int = 40,
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
                    "solve_codeact",
                    doc_id=document.doc_id,
                    question_id=q.question_id,
                    question=q.question[:200],
                    profile=self.profile.name,
                ) as q_span:
                    question_text = q.question + self._per_question_prefix(q)
                    agent = CodeActAgent(
                        instructions=instructions,
                        tools=tools,
                        sandbox_code=sandbox_code,
                        max_iterations=max_iter,
                    )
                    logger.info(
                        "CODEACT [%s] Q %s: max_iterations=%d (page_bonus=%d)",
                        self.profile.name, q.question_id, max_iter, int(page_bonus),
                    )
                    result = agent(question=question_text, doc_info=doc_info)
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
                            "CODEACT[%s] Q %s: %s (GT=%s, PRED=%s)",
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
                logger.info("CODEACT: running %d questions with concurrency=%d", len(document.questions), max_w)
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
                    "CODEACT [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name, document.doc_id, correct, scored,
                    100 * correct / scored,
                )

            return predictions, trajectories


def create_codeact_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 40,
    vlm: dict[str, Any] | None = None,
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    batch_concurrency: int = 8,
) -> CodeActProgram:
    """Hydra factory. Mirrors ``create_rvlm_program`` (no ``rlm_type``)."""
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
        max_tokens=vlm.get("max_tokens", 16384),
        temperature=vlm.get("temperature", 1.0),
        top_p=vlm.get("top_p"),
        top_k=vlm.get("top_k"),
        presence_penalty=vlm.get("presence_penalty"),
        enable_thinking=vlm.get("enable_thinking", False),
        vertex_location=vlm.get("vertex_location"),
    ) if vlm and vlm.get("model") else LMConfig()

    vlm_lm = vlm_config.to_dspy_lm()

    return CodeActProgram(
        vlm_lm=vlm_lm,
        profile=profile,
        max_iterations=max_iterations,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
        batch_concurrency=batch_concurrency,
    )
