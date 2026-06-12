"""CodeAct-Chat: a true multi-turn agentic loop whose action is Python code.

Same task as ``rvlm`` / ``codeact`` — recursive VLM perception via
``batch_look`` in a Python REPL, identical tools and prompt body
(imported from ``rvlm_solver``). The difference is **how the agent loop
maintains state**, and it is the whole point of this solver.

Why this exists (vs. ``codeact_solver``)
----------------------------------------
``codeact_solver`` is *append-only in content*, but its mechanism is a
single-turn ``dspy.Predict`` re-invoked each step: the prior history is
re-rendered into a ``trajectory`` **string input field** and wrapped in
dspy's ChatAdapter field scaffolding. The literal tokens the policy
sampled are never what gets replayed — a paraphrase of them is, inside a
user-role field. That derived rendering is a *lossy observation* of the
true interaction history → POMDP-shaped, and useless as an RL rollout
(no clean ``(action, observation)`` turns to train on).

This solver runs the **standard agentic / tool-use chat loop**: one
persistent ``messages`` array — ``system``, then strictly alternating
``assistant`` (reasoning + code) / ``user`` (execution output). Each
assistant turn is *literally* the sampled completion; each observation
is a real conversation turn. The array **is** the state → a clean,
fully-observable MDP rollout, RL-ready, with no dspy in the agent loop:
the reasoner is a direct ``litellm.completion`` call, configured from the
globally-configured ``dspy.settings.lm`` (model + kwargs read once, no
re-plumbing). The VLM perception tool (``batch_look``) is reused
unchanged — it is orthogonal to the agent-loop concern.

The returned "trajectory" is the ``messages`` list itself: the transcript
is the artifact.
"""

from __future__ import annotations

import logging
import math
import os
import re
import tempfile
from typing import Any

import dspy
import litellm
import logfire
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

from docvqa.data import Document, Question
from docvqa.datasets.profile import DatasetProfile, get_profile
from docvqa.rlm.subprocess_interpreter import HistoryReset, SubprocessInterpreter
from docvqa.solvers.rvlm_solver import (
    _build_sandbox_code,
    _build_task_instructions,
    _create_tools,
)
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)

# Chat-protocol mechanics appended to rvlm's task body. States the
# turn-based contract explicitly: one code block per turn, stdout comes
# back as the next message, conversation is the complete append-only memory.
_CHAT_MECHANICS = (
    "\n\n## REPL PROTOCOL\n"
    "You operate in a turn-based loop. On each of your turns:\n"
    "1. Briefly state your reasoning — what you now know and what you'll do next.\n"
    "2. Emit your action as Python inside a fenced ```python ... ``` block. "
    "It runs in a persistent REPL (variables you assign stay defined across "
    "turns). ALWAYS `print()` what you want to observe.\n"
    "3. You will receive the captured stdout back as the next message, wrapped "
    "in a ```output ... ``` block. Reason about it, then take your next turn.\n"
    "Repeat until you have a verified answer, then call `SUBMIT(answer=\"...\")` "
    "inside a ```python``` block to deliver the final answer and stop.\n"
    "This conversation is your complete memory — it is APPEND-ONLY; nothing is "
    "hidden, summarized, or reset, so print only what you need to carry forward. "
    "Do not try to solve everything in one turn. You have up to {max_iterations} turns."
)

# All fenced code blocks in an assistant turn (```python / ```py / bare ```).
_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
# Reasoning block emitted when enable_thinking=true.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "")


def _extract_code(text: str) -> str:
    """Concatenate every fenced code block in the assistant message.

    Robust to a turn that bundles setup + action across multiple blocks.
    Strips any ``<think>...</think>`` reasoning first so example code the model
    writes *inside* its reasoning is never mistaken for the action.
    Returns "" when the message carries no code.
    """
    blocks = _FENCE_RE.findall(_strip_think(text))
    return "\n\n".join(b.strip() for b in blocks if b.strip()).strip()


def _unwrap_final(result: Any) -> tuple[FinalOutput | None, str]:
    """Detect a SUBMIT() terminal action in an interpreter result."""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], FinalOutput):
        return result[0], result[1]
    if isinstance(result, FinalOutput):
        return result, ""
    return None, ""


class CodeActChatProgram:
    """CodeAct-Chat solver — true multi-turn chat loop over ``batch_look``.

    Mirrors :class:`RvlmProgram` / :class:`CodeActProgram` (same per-doc /
    per-question structure, concurrency, scoring) but the agent loop is a
    hand-rolled ``messages``-based chat loop instead of dspy.
    """

    def __init__(
        self,
        vlm_lm: dspy.LM,
        profile: DatasetProfile,
        max_iterations: int = 40,
        page_factor: float = 1.5,
        question_concurrency: int = 4,
        batch_concurrency: int = 8,
        max_output_chars: int = 8000,
    ):
        self.vlm_lm = vlm_lm
        self.profile = profile
        self.max_iterations = max_iterations
        self.page_factor = page_factor
        self.question_concurrency = question_concurrency
        self.batch_concurrency = batch_concurrency
        self.max_output_chars = max_output_chars

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

    # ---- reasoner: pure litellm, config read from the global dspy LM ----

    @staticmethod
    def _reasoner_lm() -> dspy.LM:
        lm = dspy.settings.lm
        if lm is None:
            raise RuntimeError("No reasoner LM configured (dspy.settings.lm is None)")
        return lm

    @staticmethod
    def _complete(lm: dspy.LM, messages: list[dict]) -> str:
        """One reasoner turn. Returns the assistant's FULL text.

        With ``enable_thinking=true`` + vllm ``--reasoning-parser qwen3`` the
        reasoning arrives in a separate ``reasoning_content`` field (NOT in
        ``content``). We splice it back into a ``<think>...</think>`` block so
        the reasoning is preserved verbatim in the transcript (the MDP rollout
        / RL target) and re-fed to the model on the next turn. Falls back
        cleanly when thinking is off or already inlined as ``<think>``.
        """
        resp = litellm.completion(
            model=lm.model,
            messages=messages,
            num_retries=getattr(lm, "num_retries", 0),
            **lm.kwargs,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
        if reasoning and "<think>" not in content:
            # reasoning-parser ON: reasoning split into its own field → re-wrap.
            return f"<think>\n{reasoning}\n</think>\n\n{content}"
        if "</think>" in content and "<think>" not in content:
            # reasoning-parser OFF: the chat template pre-fills the opening
            # <think> in the prompt, so content starts *inside* the think block
            # ("{reasoning}</think>...{answer}"). Restore the opener so the
            # block is well-formed (and _strip_think / _extract_code work).
            return f"<think>\n{content}"
        return content

    def _truncate(self, output: str) -> str:
        if not output:
            return "(no output — did you forget to print?)"
        if len(output) > self.max_output_chars:
            return output[: self.max_output_chars] + "\n... (truncated)"
        return output

    def _per_question_prefix(self, q: Question) -> str:
        if self.profile.question_format_hint_fn is None:
            return ""
        hint = self.profile.question_format_hint_fn(q)
        return f"\n{hint}\n" if hint else ""

    # ---- the agent loop ----

    def _run_chat(
        self,
        lm: dspy.LM,
        system_prompt: str,
        question: str,
        doc_info: str,
        sandbox_code: str,
        tools: list,
        max_iter: int,
    ) -> tuple[str, list[dict]]:
        repl = SubprocessInterpreter(
            tools={getattr(t, "__name__", str(t)): t for t in tools},
            output_fields=[{"name": "answer", "type": "str"}],
            timeout=120.0,
            sandbox_code=sandbox_code,
        )
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"question: {question}\n"
                    f"doc_info: {doc_info}\n\n"
                    "The page images are available in the REPL as `pages` (0-indexed "
                    "list of PIL Images) and the question text as `question`. "
                    "Begin."
                ),
            },
        ]
        # question/doc_info are also injected as REPL variables so the prompt
        # body's "`question` is available" contract holds.
        repl_vars = {"question": question, "doc_info": doc_info}

        try:
            for i in range(max_iter):
                content = self._complete(lm, messages)
                messages.append({"role": "assistant", "content": content})

                code = _extract_code(content)
                logger.info("CODEACT-CHAT step %d/%d\nAssistant:\n%s", i + 1, max_iter, content)

                if not code:
                    messages.append({
                        "role": "user",
                        "content": (
                            "[No python code block found. Emit your action as a "
                            "```python ... ``` block, or call SUBMIT(answer=\"...\") "
                            "inside one.]"
                        ),
                    })
                    continue

                try:
                    result = repl.execute(code, variables=repl_vars)
                except (CodeInterpreterError, SyntaxError) as e:
                    messages.append({"role": "user", "content": f"```output\n[Error] {e}\n```"})
                    continue

                final_output, captured = _unwrap_final(result)
                if final_output is not None:
                    out = final_output.output
                    if isinstance(out, dict) and "answer" in out:
                        # The assistant's SUBMIT(...) turn is already in `messages`
                        # and IS the terminal action — no synthetic closing turn.
                        # The transcript ends on a real policy action.
                        answer = str(out["answer"]).strip()
                        return answer, messages
                    msg = f"[Error] SUBMIT requires answer=...; got {out!r}"
                    body = f"{captured}\n{msg}" if captured else msg
                    messages.append({"role": "user", "content": f"```output\n{body}\n```"})
                    continue

                # Append-only: RESET_HISTORY has no meaning here.
                if isinstance(result, HistoryReset):
                    messages.append({
                        "role": "user",
                        "content": "```output\n[RESET_HISTORY ignored — conversation is append-only]\n```",
                    })
                    continue

                if isinstance(result, list):
                    output = "\n".join(map(str, result))
                elif isinstance(result, str) and result.startswith("[Error]"):
                    output = result
                else:
                    output = str(result) if result else ""
                messages.append({"role": "user", "content": f"```output\n{self._truncate(output)}\n```"})

            # Max turns reached without SUBMIT → one extract turn.
            logger.warning("CODEACT-CHAT reached max turns (%d); extracting final answer", max_iter)
            messages.append({
                "role": "user",
                "content": (
                    "You are out of turns. Based on everything above, output ONLY the "
                    "final answer value now — no code, no sentences, no explanation."
                ),
            })
            full = self._complete(lm, messages)
            messages.append({"role": "assistant", "content": full})
            # The prediction is the answer value only — drop any <think> block.
            answer = _strip_think(full).strip()
            return answer, messages
        finally:
            repl.shutdown()

    def solve_document(self, document: Document) -> tuple[dict[str, str], dict[str, list[dict]]]:
        lm = self._reasoner_lm()
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, img in enumerate(document.images):
                img.save(os.path.join(tmpdir, f"page_{i}.png"), format="PNG")

            doc_info = f"Category: {document.doc_category}, Pages: {len(document.images)}"

            num_pages = len(document.images)
            page_bonus = min(10, self.page_factor * math.ceil(math.sqrt(max(0, num_pages - 9))))
            max_iter = self.max_iterations + int(page_bonus)

            instructions = _build_task_instructions(self.profile)
            system_prompt = instructions + _CHAT_MECHANICS.format(max_iterations=max_iter)
            tools = _create_tools(self.vlm_predict, self.vlm_lm, self.batch_concurrency)
            sandbox_code = _build_sandbox_code(tmpdir, len(document.images))

            def _solve_question(q: Question):
                with logfire.span(
                    "solve_codeact_chat",
                    doc_id=document.doc_id,
                    question_id=q.question_id,
                    question=q.question[:200],
                    profile=self.profile.name,
                ) as q_span:
                    question_text = q.question + self._per_question_prefix(q)
                    logger.info(
                        "CODEACT-CHAT [%s] Q %s: max_iterations=%d (page_bonus=%d)",
                        self.profile.name, q.question_id, max_iter, int(page_bonus),
                    )
                    answer, trajectory = self._run_chat(
                        lm=lm,
                        system_prompt=system_prompt,
                        question=question_text,
                        doc_info=doc_info,
                        sandbox_code=sandbox_code,
                        tools=tools,
                        max_iter=max_iter,
                    )
                    answer = str(answer or "").strip()
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
                            "CODEACT-CHAT[%s] Q %s: %s (GT=%s, PRED=%s)",
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
                logger.info("CODEACT-CHAT: running %d questions with concurrency=%d", len(document.questions), max_w)
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
                    "CODEACT-CHAT [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name, document.doc_id, correct, scored,
                    100 * correct / scored,
                )

            return predictions, trajectories


def create_codeact_chat_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 40,
    vlm: dict[str, Any] | None = None,
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    batch_concurrency: int = 8,
) -> CodeActChatProgram:
    """Hydra factory. Mirrors ``create_codeact_program``."""
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

    return CodeActChatProgram(
        vlm_lm=vlm_lm,
        profile=profile,
        max_iterations=max_iterations,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
        batch_concurrency=batch_concurrency,
    )
