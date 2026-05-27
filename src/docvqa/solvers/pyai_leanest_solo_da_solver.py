"""Pydantic-AI-RLM port of :mod:`docvqa.solvers.leanest_solo_da_solver`.

Same dataset-aware leanest-solo behavior — one tool only (``batch_look``),
profile-driven prompt / tips / scorer — but the agent and REPL are built on
top of `pydantic-ai-rlm` instead of `dspy`:

- The code-writing LLM runs through `pydantic_ai.Agent` configured with
  `LiteLLMProvider` so the existing local Qwen 3.5 27B vLLM endpoint works
  without re-plumbing.
- The REPL is a thin subclass of `pydantic_ai_rlm.REPLEnvironment` that
  injects ``pages`` (list of PIL Images) and a ``batch_look`` global into
  the sandbox — `pydantic_ai_rlm` only knows about a single ``context``
  variable and an optional text-only ``llm_query``, so we add what we need.
- The VLM (used by ``batch_look``) is called via ``litellm.completion``
  directly with image content; the main agent never sees image bytes.

The class exposes the same ``solve_document(doc) -> (preds, trajectories)``
interface as the dspy variant, so the runner doesn't need to change.
"""

from __future__ import annotations

import asyncio
import base64
import json as _json
import logging
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Callable

import litellm
import logfire
from pydantic_ai import Agent, RunContext, UsageLimits, capture_run_messages
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_graph import End
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.litellm import LiteLLMProvider
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_rlm import REPLEnvironment, RLMConfig, RLMDependencies
from pydantic_ai_rlm.utils import format_repl_result

from docvqa.data import Document, Question
from docvqa.datasets.profile import DatasetProfile, get_profile
from docvqa.solvers.rvlm_solver import _build_task_instructions
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)


class _AnswerCarrier:
    """Captures the answer when the agent calls SUBMIT() inside the REPL.

    Why: pydantic-ai's "structured output via terminal tool" pattern lets the
    agent click `submit` on its first turn with answer='Unknown', skipping all
    the exploration work. The dspy variant doesn't have this failure mode
    because its SUBMIT is a Python function the agent has to compose code to
    call — which adds enough friction that the agent only does it after
    actually investigating. We replicate that here.
    """

    __slots__ = ("answer",)

    def __init__(self) -> None:
        self.answer: str | None = None


# ---------------------------------------------------------------------------
# REPL: pages + batch_look injected globals
# ---------------------------------------------------------------------------


class _PagesREPL(REPLEnvironment):
    """REPLEnvironment with ``pages``, ``batch_look``, and ``SUBMIT`` injected.

    Mirrors the dspy LeanRLM contract: the agent has to write Python that
    calls ``SUBMIT(answer=...)`` to deliver a final answer. The call captures
    the answer in ``carrier``; the surrounding execute_code wrapper detects
    a non-None carrier and tells the agent to stop.
    """

    def __init__(
        self,
        pages_dir: str,
        num_pages: int,
        batch_look_impl: Callable[[list[tuple[str, str]]], list[str]],
        carrier: _AnswerCarrier,
        config: RLMConfig | None = None,
    ) -> None:
        # Bypass REPLEnvironment.__init__ entirely. Its constructor calls
        # os.getcwd() — and its execute() does a process-wide os.chdir into
        # self.temp_dir, then back. With multiple concurrent REPLs (one per
        # question, asyncio tasks sharing one process), these flips race:
        # REPL A chdir's into its tempdir, REPL B then deletes its OWN
        # tempdir but the process CWD pointer can get invalidated; the next
        # os.getcwd() raises FileNotFoundError and the doc crashes. We
        # reimplement the parent's state setup here without touching CWD,
        # and override _temp_working_directory below to be a no-op.
        import threading as _threading
        safe_cwd = os.environ.get("HOME", "/") or "/"
        self.config = config or RLMConfig()
        self.original_cwd = safe_cwd
        self.temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")
        self._lock = _threading.Lock()
        self.locals = {}
        self.globals = {"__builtins__": self._create_builtins()}
        if self.config.sub_model:
            self._setup_llm_query()
        self._load_context("")
        from PIL import Image as PILImage

        PILImage.MAX_IMAGE_PIXELS = 500_000_000
        pages = []
        for i in range(num_pages):
            path = os.path.join(pages_dir, f"page_{i}.png")
            assert os.path.exists(path), f"Page image not found: {path}"
            pages.append(PILImage.open(path))

        def _batch_look(requests: list) -> list[str]:
            """Sandbox-facing batch_look — serializes PIL images, calls VLM impl."""
            paths: list[tuple[str, str]] = []
            for image, query in requests:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                image.save(tmp, format="PNG")
                tmp.close()
                paths.append((tmp.name, query))
            return batch_look_impl(paths)

        def _SUBMIT(answer: str = "") -> None:
            """Submit the final answer. Call this once you're confident.

            After this call the agent should stop calling tools and end its turn.
            """
            carrier.answer = str(answer)
            print(f"[SUBMITTED] {answer!r}")

        self.globals["pages"] = pages
        self.globals["batch_look"] = _batch_look
        self.globals["SUBMIT"] = _SUBMIT

    @contextmanager  # type: ignore[misc]
    def _temp_working_directory(self):  # type: ignore[override]
        """Override the parent's process-wide chdir dance.

        The parent's version flips ``os.chdir`` to ``self.temp_dir`` for the
        duration of an execute() call, then flips back. With multiple
        concurrent REPLs in the same process (one per question), these flips
        race and one REPL can leave the process chdir'd into a tempdir that
        another REPL is about to delete. The next ``os.getcwd()`` then raises
        FileNotFoundError. We don't need per-REPL CWD anyway, so just no-op.
        """
        yield


# ---------------------------------------------------------------------------
# VLM callable (litellm-based) used inside the REPL's batch_look
# ---------------------------------------------------------------------------


def _make_vlm_callable(
    vlm_cfg: LMConfig, batch_concurrency: int
) -> Callable[[list[tuple[str, str]]], list[str]]:
    vlm_system = (
        "Analyze the image content strictly to answer the query. "
        "Transcribe numbers and characters exactly. "
        "For technical drawings, trace leader lines and arrows to connect labels to their specific parts. "
        "Output ONLY the concise answer. If the information is missing, output 'Unknown'."
    )

    def _look_one(path: str, query: str) -> str:
        with logfire.span("look", image_path=path, query=query) as span:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            messages = [
                {"role": "system", "content": vlm_system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                },
            ]
            kwargs: dict[str, Any] = {
                "model": vlm_cfg.model,
                "messages": messages,
                "temperature": vlm_cfg.temperature,
                "timeout": 600,
            }
            if vlm_cfg.api_base:
                kwargs["api_base"] = vlm_cfg.api_base
            if vlm_cfg.api_key:
                kwargs["api_key"] = vlm_cfg.api_key
            if vlm_cfg.max_tokens:
                kwargs["max_tokens"] = vlm_cfg.max_tokens
            if vlm_cfg.top_p is not None:
                kwargs["top_p"] = vlm_cfg.top_p
            extra_body: dict[str, Any] = {}
            if vlm_cfg.top_k is not None:
                extra_body["top_k"] = vlm_cfg.top_k
            if vlm_cfg.enable_thinking is not None and "mistral" not in vlm_cfg.model.lower():
                extra_body["chat_template_kwargs"] = {"enable_thinking": vlm_cfg.enable_thinking}
            if extra_body:
                kwargs["extra_body"] = extra_body

            resp = litellm.completion(**kwargs)
            msg = resp.choices[0].message
            content = msg.content or getattr(msg, "reasoning_content", None) or ""
            content = content.strip()
            span.set_attribute("answer", content[:2000])
            try:
                os.unlink(path)
            except OSError:
                pass
            return content

    def _batch(paths_queries: list[tuple[str, str]]) -> list[str]:
        if not paths_queries:
            return []
        results: list[str] = [""] * len(paths_queries)
        with logfire.span("batch_look", num_requests=len(paths_queries)):
            max_w = min(len(paths_queries), batch_concurrency)
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                futures = {
                    pool.submit(_look_one, p, q): i
                    for i, (p, q) in enumerate(paths_queries)
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    results[idx] = fut.result()
        return results

    return _batch


# ---------------------------------------------------------------------------
# Toolset: single execute_code tool against our REPL
# ---------------------------------------------------------------------------


_EXECUTE_DESC = (
    "Execute Python code in a sandboxed REPL.\n"
    "## Environment\n"
    "- `pages`: list of PIL Image objects, one per document page (0-indexed).\n"
    "- `batch_look(requests)`: vision tool. Input: list of (image, query) tuples "
    "where image is any PIL Image (a page or a crop). Returns: list of str answers, same order.\n"
    "- `SUBMIT(answer='...')`: call this when you have the final answer. "
    "The orchestrator captures it and ends the run.\n"
    "- Variables persist between executions. Use `print()` to see results.\n"
    "## Notes\n"
    "- ALL visual queries must go through `batch_look`. The agent does NOT see image bytes directly.\n"
    "- Crop tightly for fine details: `pages[i].crop((l,t,r,b))`.\n"
    "- The `context` variable is unused — ignore it."
)


def _create_toolset(
    repl: _PagesREPL, carrier: _AnswerCarrier, code_timeout: float
) -> FunctionToolset[RLMDependencies]:
    toolset: FunctionToolset[RLMDependencies] = FunctionToolset()

    @toolset.tool(description=_EXECUTE_DESC)
    async def execute_code(ctx: RunContext[RLMDependencies], code: str) -> str:
        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, repl.execute, code),
                timeout=code_timeout,
            )
            formatted = format_repl_result(result)
            if carrier.answer is not None:
                return (
                    formatted
                    + "\n\n[ANSWER RECEIVED — stop calling tools and end your turn. "
                    + "Any further tool calls are wasted.]"
                )
            return formatted
        except TimeoutError:
            return f"Error: Code execution timed out after {code_timeout} seconds."
        except Exception as e:  # noqa: BLE001 — surface to agent
            return f"Error executing code: {e!s}"

    return toolset


# ---------------------------------------------------------------------------
# Trajectory conversion: pydantic-ai messages -> runner step dicts
# ---------------------------------------------------------------------------


def _messages_to_trajectory(messages: list) -> list[dict]:
    """Convert pydantic-ai message history to ``[{reasoning, code, output}, ...]``.

    The runner's ``_save_summary_md`` reads these keys.
    """
    trajectory: list[dict] = []
    current_reasoning = ""
    pending: dict[str, dict] = {}

    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    if part.content:
                        current_reasoning += part.content + "\n"
                elif isinstance(part, ToolCallPart):
                    args = part.args
                    if isinstance(args, str):
                        try:
                            args = _json.loads(args)
                        except Exception:
                            args = {"code": args}
                    code = args.get("code", "") if isinstance(args, dict) else ""
                    step = {
                        "reasoning": current_reasoning.strip(),
                        "code": code,
                        "output": "",
                    }
                    pending[part.tool_call_id] = step
                    trajectory.append(step)
                    current_reasoning = ""
        elif isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    step = pending.pop(part.tool_call_id, None)
                    if step:
                        out = part.content
                        step["output"] = out if isinstance(out, str) else str(out)

    if current_reasoning.strip():
        trajectory.append(
            {"reasoning": current_reasoning.strip(), "code": "", "output": ""}
        )
    return trajectory


# ---------------------------------------------------------------------------
# PyaiLeanestSoloDAProgram
# ---------------------------------------------------------------------------


class PyaiLeanestSoloDAProgram:
    """pydantic-ai-rlm port of LeanestSoloDAProgram. Same public interface."""

    def __init__(
        self,
        lm_cfg: LMConfig,
        vlm_cfg: LMConfig,
        profile: DatasetProfile,
        max_iterations: int = 25,
        page_factor: float = 1.5,
        question_concurrency: int = 4,
        batch_concurrency: int = 8,
        code_timeout: float = 120.0,
    ) -> None:
        self.lm_cfg = lm_cfg
        self.vlm_cfg = vlm_cfg
        self.profile = profile
        self.max_iterations = max_iterations
        self.page_factor = page_factor
        self.question_concurrency = question_concurrency
        self.batch_concurrency = batch_concurrency
        self.code_timeout = code_timeout

        provider = LiteLLMProvider(
            api_base=lm_cfg.api_base,
            api_key=lm_cfg.api_key or "dummy",
        )
        # pydantic-ai's "LiteLLMProvider" is actually a thin AsyncOpenAI client — the
        # model name is sent verbatim to the upstream. vLLM only knows the bare HF id,
        # not LiteLLM's `hosted_vllm/...` routing prefix. Strip it.
        model_id = lm_cfg.model
        for prefix in ("hosted_vllm/", "openai/", "litellm/"):
            if model_id.startswith(prefix):
                model_id = model_id[len(prefix):]
                break
        settings_kwargs: dict[str, Any] = {"temperature": lm_cfg.temperature}
        if lm_cfg.max_tokens:
            settings_kwargs["max_tokens"] = lm_cfg.max_tokens
        if lm_cfg.top_p is not None:
            settings_kwargs["top_p"] = lm_cfg.top_p
        extra_body: dict[str, Any] = {}
        if lm_cfg.top_k is not None:
            extra_body["top_k"] = lm_cfg.top_k
        if lm_cfg.enable_thinking is not None and "mistral" not in lm_cfg.model.lower():
            extra_body["chat_template_kwargs"] = {"enable_thinking": lm_cfg.enable_thinking}
        if extra_body:
            settings_kwargs["extra_body"] = extra_body
        self.model_settings = OpenAIChatModelSettings(**settings_kwargs)
        self.model = OpenAIChatModel(model_id, provider=provider)

    async def _extract_fallback(
        self, question_text: str, messages: list
    ) -> str:
        """Last-resort answer extraction from the trajectory's observations.

        Mirrors LeanRLM._extract_fallback. We concatenate every
        ToolReturnPart's content (i.e. all execute_code outputs the agent
        saw), and ask the LLM to compose the answer in a single call.
        """
        observations: list[str] = []
        for m in messages:
            if isinstance(m, ModelRequest):
                for p in m.parts:
                    if isinstance(p, ToolReturnPart):
                        out = p.content if isinstance(p.content, str) else str(p.content)
                        if out:
                            observations.append(out[:5000])
        joined = "\n\n---\n\n".join(observations[-30:]) or "(no observations recorded)"
        prompt = (
            "Below are the OBSERVATIONS from a document-VQA agent's exploration of "
            "a document. Based ONLY on these observations, give the final answer "
            "to the QUESTION.\n\n"
            "CRITICAL OUTPUT RULES:\n"
            "- Output ONLY the answer value — 1 to 5 words, on a single line.\n"
            "- NO preamble, no 'The answer is...', no quotes, no explanation.\n"
            "- If the observations don't contain the answer, output your best guess "
            "based on what you have, not a refusal.\n"
            "- For 'how many' questions, output a single integer.\n"
            "- For names/items, output the exact phrase from the document.\n\n"
            f"QUESTION: {question_text}\n\n"
            f"OBSERVATIONS:\n{joined}\n\n"
            "FINAL ANSWER:"
        )
        kwargs: dict[str, Any] = {
            "model": self.lm_cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.lm_cfg.temperature,
            "timeout": 300,
            "max_tokens": 256,
        }
        if self.lm_cfg.api_base:
            kwargs["api_base"] = self.lm_cfg.api_base
        if self.lm_cfg.api_key:
            kwargs["api_key"] = self.lm_cfg.api_key
        extra_body: dict[str, Any] = {}
        if self.lm_cfg.enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": self.lm_cfg.enable_thinking
            }
        if extra_body:
            kwargs["extra_body"] = extra_body
        try:
            resp = await litellm.acompletion(**kwargs)
            msg = resp.choices[0].message  # type: ignore[union-attr]
            content = (
                msg.content
                or getattr(msg, "reasoning_content", None)
                or ""
            )
            # Keep just the first non-empty line so verbose preambles
            # ("The information required...") get stripped to nothing
            # rather than poisoning the answer field.
            for line in content.splitlines():
                line = line.strip().strip("\"'.,;:")
                if line and not line.lower().startswith((
                    "the answer", "answer:", "final answer", "based on",
                )):
                    return line
            return content.strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("Extract fallback failed: %s", e)
            return ""

    def _per_question_prefix(self, q: Question) -> str:
        if self.profile.question_format_hint_fn is None:
            return ""
        hint = self.profile.question_format_hint_fn(q)
        return f"\n{hint}\n" if hint else ""

    def solve_document(
        self, document: Document
    ) -> tuple[dict[str, str], dict[str, list[dict]]]:
        import traceback as _tb
        try:
            return asyncio.run(self._solve_document_async(document))
        except Exception as e:
            logger.error(
                "Doc %s crashed in solve_document:\n%s",
                document.doc_id, _tb.format_exc(),
            )
            preds = {q.question_id: "Unknown" for q in document.questions}
            return preds, {}

    async def _solve_document_async(
        self, document: Document
    ) -> tuple[dict[str, str], dict[str, list[dict]]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Force-decode each page in case HF returned lazy PIL refs whose
            # backing file handles get closed by GC under fd pressure on long
            # runs. .copy() detaches from the file.
            for i, img in enumerate(document.images):
                try:
                    if hasattr(img, "load"):
                        img.load()
                    snap = img.copy()
                    snap.save(os.path.join(tmpdir, f"page_{i}.png"), format="PNG")
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Doc %s page %d: image load/save failed (%s); "
                        "saving 1x1 placeholder so the doc doesn't crash the run",
                        document.doc_id, i, e,
                    )
                    from PIL import Image as PILImage
                    PILImage.new("RGB", (1, 1)).save(
                        os.path.join(tmpdir, f"page_{i}.png"), format="PNG"
                    )

            doc_info = f"Category: {document.doc_category}, Pages: {len(document.images)}"
            num_pages = len(document.images)
            page_bonus = min(
                10, self.page_factor * math.ceil(math.sqrt(max(0, num_pages - 9)))
            )
            max_iter = self.max_iterations + int(page_bonus)

            base_instructions = _build_task_instructions(self.profile)
            tips = self.profile.category_tips_fn(document.doc_category)
            instructions = base_instructions + ("\n" + tips if tips else "")

            batch_look_impl = _make_vlm_callable(self.vlm_cfg, self.batch_concurrency)

            sem = asyncio.Semaphore(max(1, self.question_concurrency))

            async def _solve_question(
                q: Question,
            ) -> tuple[str, str, list[dict]]:
                async with sem:
                    with logfire.span(
                        "solve_pyai_leanest_solo_da",
                        doc_id=document.doc_id,
                        question_id=q.question_id,
                        question=q.question[:200],
                        profile=self.profile.name,
                    ) as q_span:
                        carrier = _AnswerCarrier()
                        repl = _PagesREPL(tmpdir, num_pages, batch_look_impl, carrier)
                        try:
                            toolset = _create_toolset(repl, carrier, self.code_timeout)
                            agent_instructions = (
                                instructions
                                + f"\n\nDocument info: {doc_info}\n\n"
                                + "## EXECUTION\n"
                                + "You have ONE tool: `execute_code`. Use it iteratively to write Python "
                                + "that inspects pages and calls batch_look. Variables persist across calls.\n"
                                + "ALWAYS wrap batch_look in print(...) so you can see the result, e.g. "
                                + "`for ans in batch_look(reqs): print(ans)`. Naked expressions may be "
                                + "swallowed.\n"
                                + f"You have up to {max_iter} execute_code calls. "
                                + "When you have the answer, write a final execute_code call that does "
                                + "`SUBMIT(answer='<your answer>')`. This is the ONLY way to deliver the "
                                + "answer — emitting plain text instead will NOT count. Do real "
                                + "exploration before submitting; don't submit 'Unknown' until you have "
                                + "looked at enough pages to be sure."
                            )
                            agent = Agent(
                                self.model,
                                deps_type=RLMDependencies,
                                toolsets=[toolset],
                                instructions=agent_instructions,
                                output_type=str,
                                model_settings=self.model_settings,
                            )

                            question_text = q.question + self._per_question_prefix(q)
                            deps = RLMDependencies(
                                context="(unused — work via `pages` and `batch_look` in execute_code)"
                            )

                            def _is_retryable(e: BaseException) -> bool:
                                s = str(e)
                                t = type(e).__name__
                                return (
                                    "429" in s
                                    or "RateLimit" in t
                                    or "RESOURCE_EXHAUSTED" in s
                                    or "ClosedResource" in t
                                    or "Connection error" in s
                                    or "ConnectionError" in t
                                    or "ConnectTimeout" in t
                                    or "ReadTimeout" in t
                                )

                            # Drive the agent manually with Agent.iter so we
                            # can break early when SUBMIT() lands in carrier
                            # and so we can fall through to an extract pass
                            # if the agent never submits (mirrors dspy's
                            # LeanRLM._extract_fallback path).
                            attempt = 0
                            messages_for_traj: list = []
                            last_result = None
                            stopped_via_text = False
                            while True:
                                try:
                                    with capture_run_messages() as captured:
                                        async with agent.iter(
                                            question_text,
                                            deps=deps,
                                            usage_limits=UsageLimits(
                                                tool_calls_limit=max_iter + 5
                                            ),
                                        ) as run:
                                            async for node in run:
                                                if isinstance(node, End):
                                                    stopped_via_text = True
                                                    break
                                                if carrier.answer is not None:
                                                    break
                                            last_result = run.result
                                        messages_for_traj = list(captured)
                                    break
                                except UsageLimitExceeded as e:
                                    logger.warning(
                                        "PyAI Q %s: tool-call budget exhausted (%d): %s",
                                        q.question_id, max_iter, e,
                                    )
                                    messages_for_traj = list(captured)
                                    break
                                except Exception as e:
                                    if _is_retryable(e) and attempt < 3:
                                        wait = min(15 * (2**attempt), 90)
                                        logger.warning(
                                            "Transient error, retry %d in %ds: %s",
                                            attempt + 1, wait, e,
                                        )
                                        await asyncio.sleep(wait)
                                        attempt += 1
                                        continue
                                    # Non-retryable / out of retries: log and bail
                                    # to "Unknown" rather than crashing the doc.
                                    logger.warning(
                                        "PyAI Q %s: giving up after error: %s",
                                        q.question_id, e,
                                    )
                                    break

                            # Decide the final answer.
                            if carrier.answer is not None:
                                answer = carrier.answer.strip()
                            elif stopped_via_text and last_result is not None:
                                answer = str(last_result.output or "").strip()
                            else:
                                answer = ""

                            if not answer:
                                # Extract fallback: re-ask the LLM with the
                                # trajectory's collected text as context.
                                answer = await self._extract_fallback(
                                    question_text, messages_for_traj
                                ) or "Unknown"

                            trajectory = _messages_to_trajectory(messages_for_traj)

                            q_span.set_attribute("num_iterations", len(trajectory))
                            q_span.set_attribute("prediction", answer[:200])

                            if q.answer is not None:
                                is_correct, extracted = self.profile.score_fn(
                                    answer, q.answer, q
                                )
                                q_span.set_attribute("is_correct", is_correct)
                                q_span.set_attribute("ground_truth", q.answer[:200])
                                q_span.set_attribute("extracted_answer", extracted[:200])
                                logger.info(
                                    "PyAI Solo[%s] Q %s: %s (GT=%s, PRED=%s)",
                                    self.profile.name,
                                    q.question_id,
                                    "CORRECT" if is_correct else "WRONG",
                                    q.answer[:40],
                                    extracted[:40],
                                )

                            return q.question_id, answer, trajectory
                        finally:
                            repl.cleanup()

            results = await asyncio.gather(
                *[_solve_question(q) for q in document.questions]
            )
            predictions = {qid: ans for qid, ans, _ in results}
            trajectories = {qid: traj for qid, _, traj in results}

            correct = 0
            scored = 0
            for q in document.questions:
                if q.answer is not None:
                    scored += 1
                    if self.profile.score_fn(predictions[q.question_id], q.answer, q)[0]:
                        correct += 1
            if scored > 0:
                logger.info(
                    "PyAI Leanest solo DA [%s] doc %s: %d/%d = %.1f%%",
                    self.profile.name,
                    document.doc_id,
                    correct,
                    scored,
                    100 * correct / scored,
                )
            return predictions, trajectories


# ---------------------------------------------------------------------------
# Hydra factory
# ---------------------------------------------------------------------------


def _to_lm_cfg(d: dict[str, Any] | None) -> LMConfig:
    if not d or not d.get("model"):
        return LMConfig()
    return LMConfig(
        model=d["model"],
        api_base=d.get("api_base"),
        api_key=d.get("api_key"),
        max_tokens=d.get("max_tokens", 65536),
        temperature=d.get("temperature", 1.0),
        top_p=d.get("top_p"),
        top_k=d.get("top_k"),
        presence_penalty=d.get("presence_penalty"),
        enable_thinking=d.get("enable_thinking", False),
        vertex_location=d.get("vertex_location"),
    )


def create_pyai_leanest_solo_da_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 25,
    lm: dict[str, Any] | None = None,
    vlm: dict[str, Any] | None = None,
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    batch_concurrency: int = 8,
    code_timeout: float = 120.0,
) -> PyaiLeanestSoloDAProgram:
    """Hydra factory. Takes both ``lm`` and ``vlm`` configs (pydantic-ai
    doesn't share a global LM context with dspy)."""
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

    lm_cfg = _to_lm_cfg(lm)
    vlm_cfg = _to_lm_cfg(vlm)

    return PyaiLeanestSoloDAProgram(
        lm_cfg=lm_cfg,
        vlm_cfg=vlm_cfg,
        profile=profile,
        max_iterations=max_iterations,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
        batch_concurrency=batch_concurrency,
        code_timeout=code_timeout,
    )
