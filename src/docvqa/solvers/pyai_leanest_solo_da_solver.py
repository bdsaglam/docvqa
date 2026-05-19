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
from typing import Any, Callable

import litellm
import logfire
from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
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
from docvqa.solvers.leanest_solo_da_solver import _build_task_instructions
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# REPL: pages + batch_look injected globals
# ---------------------------------------------------------------------------


class _PagesREPL(REPLEnvironment):
    """REPLEnvironment with ``pages`` and ``batch_look`` injected.

    The base class still gets a placeholder ``context=""`` (the load step
    is cheap and we don't want to fight the parent's init). The agent is
    told to ignore ``context`` and only use ``pages`` / ``batch_look``.
    """

    def __init__(
        self,
        pages_dir: str,
        num_pages: int,
        batch_look_impl: Callable[[list[tuple[str, str]]], list[str]],
        config: RLMConfig | None = None,
    ) -> None:
        super().__init__(context="", config=config)
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

        self.globals["pages"] = pages
        self.globals["batch_look"] = _batch_look


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
    "- Variables persist between executions. Use `print()` to see results.\n"
    "## Notes\n"
    "- ALL visual queries must go through `batch_look`. The agent does NOT see image bytes directly.\n"
    "- Crop tightly for fine details: `pages[i].crop((l,t,r,b))`.\n"
    "- The `context` variable is unused — ignore it."
)


def _create_toolset(
    repl: _PagesREPL, code_timeout: float
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
            return format_repl_result(result)
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

    def _per_question_prefix(self, q: Question) -> str:
        if self.profile.question_format_hint_fn is None:
            return ""
        hint = self.profile.question_format_hint_fn(q)
        return f"\n{hint}\n" if hint else ""

    def solve_document(
        self, document: Document
    ) -> tuple[dict[str, str], dict[str, list[dict]]]:
        return asyncio.run(self._solve_document_async(document))

    async def _solve_document_async(
        self, document: Document
    ) -> tuple[dict[str, str], dict[str, list[dict]]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, img in enumerate(document.images):
                img.save(os.path.join(tmpdir, f"page_{i}.png"), format="PNG")

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
                        repl = _PagesREPL(tmpdir, num_pages, batch_look_impl)
                        try:
                            toolset = _create_toolset(repl, self.code_timeout)
                            agent_instructions = (
                                instructions
                                + f"\n\nDocument info: {doc_info}\n\n"
                                + "## EXECUTION\n"
                                + "Use the `execute_code` tool repeatedly to write Python that inspects "
                                + "pages and calls batch_look. Variables persist across calls.\n"
                                + f"You have up to {max_iter} tool calls. When you are confident, "
                                + "stop calling tools and emit ONLY the final answer string as your reply "
                                + "(no preamble, no JSON, no markdown — just the answer)."
                            )
                            agent: Agent[RLMDependencies, str] = Agent(
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

                            def _is_rate_limit(e: BaseException) -> bool:
                                s = str(e)
                                return (
                                    "429" in s
                                    or "RateLimit" in type(e).__name__
                                    or "RESOURCE_EXHAUSTED" in s
                                )

                            attempt = 0
                            result = None
                            usage_overrun_messages: list | None = None
                            while True:
                                try:
                                    result = await agent.run(
                                        question_text,
                                        deps=deps,
                                        usage_limits=UsageLimits(
                                            tool_calls_limit=max_iter
                                        ),
                                    )
                                    break
                                except UsageLimitExceeded as e:
                                    # Agent burned through tool calls without producing a
                                    # final answer. Salvage the trajectory + log "Unknown".
                                    logger.warning(
                                        "PyAI Q %s: tool-call budget exhausted (%d): %s",
                                        q.question_id, max_iter, e,
                                    )
                                    usage_overrun_messages = list(
                                        getattr(e, "all_messages", lambda: [])()  # type: ignore[attr-defined]
                                    )
                                    break
                                except Exception as e:
                                    if _is_rate_limit(e) and attempt < 3:
                                        wait = min(30 * (2**attempt), 120)
                                        logger.warning(
                                            "Rate limit, retry %d in %ds: %s",
                                            attempt + 1, wait, e,
                                        )
                                        await asyncio.sleep(wait)
                                        attempt += 1
                                        continue
                                    raise

                            if result is None:
                                answer = "Unknown"
                                messages_for_traj = usage_overrun_messages or []
                            else:
                                answer = (result.output or "").strip()
                                messages_for_traj = result.all_messages()
                            if not answer:
                                answer = "Unknown"

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
