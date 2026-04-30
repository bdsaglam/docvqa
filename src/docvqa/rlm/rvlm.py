"""RVLM — Multimodal RLM where the agent can see displayed images inline.

The agent calls `display(image)` in its code. The image appears as an inline
image in the next LLM message, allowing multimodal LLMs to perceive images
directly without a separate VLM tool call.

Uses DSPy's CUSTOM_TYPE marker injection so that displayed images are
properly formatted as image_url content blocks by the existing ChatAdapter
pipeline.

Based on lean_rlm.py.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable

import pydantic
from pydantic import Field

import dspy
from dspy.adapters.types.base_type import (
    CUSTOM_TYPE_END_IDENTIFIER,
    CUSTOM_TYPE_START_IDENTIFIER,
)
from dspy.adapters.utils import parse_value, translate_field_type
from dspy.primitives.code_interpreter import SIMPLE_TYPES, CodeInterpreterError, FinalOutput
from dspy.utils.exceptions import AdapterParseError
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import ensure_signature

from docvqa.rlm.subprocess_interpreter import HistoryReset, SubprocessInterpreter

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Visual REPL types — REPLEntry/History with inline image support
# ---------------------------------------------------------------------------


def _make_image_marker(data_uri: str) -> str:
    """Create a DSPy CUSTOM_TYPE marker that embeds an image_url content block."""
    content = [{"type": "image_url", "image_url": {"url": data_uri}}]
    return f"{CUSTOM_TYPE_START_IDENTIFIER}{json.dumps(content)}{CUSTOM_TYPE_END_IDENTIFIER}"


class VisualREPLEntry(pydantic.BaseModel):
    """A single REPL interaction that may include displayed images."""

    reasoning: str = ""
    code: str
    output: str  # text with [Image N] placeholders from display() calls
    images: list[dict] = Field(default_factory=list)  # [{index, data_uri, description}]

    model_config = pydantic.ConfigDict(frozen=True)

    def format(
        self,
        index: int,
        max_output_chars: int = 5000,
        include_images: bool = True,
    ) -> str:
        """Format this entry for inclusion in prompts.

        When include_images=True, [Image N] placeholders are followed by
        CUSTOM_TYPE markers that DSPy converts to image_url content blocks.
        When False, only the text placeholder remains (saves tokens).
        """
        output = self.output
        if len(output) > max_output_chars:
            output = (
                output[:max_output_chars]
                + f"\n... (truncated to {max_output_chars}/{len(self.output):,} chars)"
            )
        if include_images and self.images:
            for img in self.images:
                placeholder = f"[Image {img['index']}]"
                marker = _make_image_marker(img["data_uri"])
                output = output.replace(placeholder, placeholder + "\n" + marker)
        reasoning_line = f"Reasoning: {self.reasoning}\n" if self.reasoning else ""
        return (
            f"=== Step {index + 1} ===\n"
            f"{reasoning_line}"
            f"Code:\n```python\n{self.code}\n```\n"
            f"Output ({len(self.output):,} chars):\n{output}"
        )


class VisualREPLHistory(pydantic.BaseModel):
    """Container for REPL interaction history with image support.

    Immutable: append() returns a new instance with the entry added.
    Only the last `images_for_last_n` entries include actual image data
    to control token usage.
    """

    entries: list[VisualREPLEntry] = Field(default_factory=list)
    images_for_last_n: int = 1

    model_config = pydantic.ConfigDict(frozen=True)

    def format(self, max_output_chars: int = 5000) -> str:
        if not self.entries:
            return "You have not interacted with the REPL environment yet."
        n = len(self.entries)
        return "\n".join(
            entry.format(
                index=i,
                max_output_chars=max_output_chars,
                include_images=(n - i) <= self.images_for_last_n,
            )
            for i, entry in enumerate(self.entries)
        )

    @pydantic.model_serializer()
    def serialize_model(self) -> str:
        return self.format()

    def append(
        self,
        *,
        reasoning: str = "",
        code: str,
        output: str,
        images: list[dict] | None = None,
    ) -> VisualREPLHistory:
        """Return a new VisualREPLHistory with the entry appended."""
        new_entry = VisualREPLEntry(
            reasoning=reasoning,
            code=code,
            output=output,
            images=images or [],
        )
        return VisualREPLHistory(
            entries=list(self.entries) + [new_entry],
            images_for_last_n=self.images_for_last_n,
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[VisualREPLEntry]:
        return iter(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0


# ---------------------------------------------------------------------------
# RVLM — RLM with inline image display
# ---------------------------------------------------------------------------

DEFAULT_ACTION_INSTRUCTIONS = """You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a Python REPL environment. Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.

IMPORTANT: Write raw Python code only. No need to wrap code in markdown fences.

Available:
- Variables: {inputs} (your input data)
- `print()` - ALWAYS print to see results
- `display(image)` - Show a PIL Image inline (you will see the image in the next step)
- `SUBMIT({final_output_names})` - call when done (ends the run immediately)
- Standard libraries: re, json, collections, math, etc.

Rules:
- This is ITERATIVE. State persists between steps. Do NOT solve everything in one step.
- ALWAYS print before submitting — verify results look correct.
- Use `display()` to look at images directly — you can see them.
- Re-access values from variables instead of retyping long strings/numbers.
- You have max {max_iterations} iterations."""

# Patterns for code fence stripping
_FENCE_START = re.compile(r"^```(?:python|py)?\s*\n", re.MULTILINE)
_FENCE_END = re.compile(r"\n```\s*$")
_STANDALONE_FENCE = re.compile(r"^```\s*$", re.MULTILINE)
_BARE_PYTHON_PREFIX = re.compile(r"^python\s*\n", re.MULTILINE)


def _strip_code_fences(code: str | None) -> str:
    if not code:
        return ""
    result = code.strip()
    prev = None
    while result != prev:
        prev = result
        result = _FENCE_START.sub("", result, count=1)
        result = _FENCE_END.sub("", result, count=1)
        result = _STANDALONE_FENCE.sub("", result)
        result = _BARE_PYTHON_PREFIX.sub("", result, count=1)
        result = result.strip()
    return result


class RVLM(Module):
    """Multimodal RLM — agent can see displayed images inline.

    Extends LeanRLM's architecture with:
    - `display(image)` function in the sandbox
    - VisualREPLHistory that embeds images as DSPy CUSTOM_TYPE markers
    - Token management via images_for_last_n
    """

    def __init__(
        self,
        signature: type[Signature] | str,
        max_iterations: int = 20,
        max_llm_calls: int = 50,
        max_output_chars: int = 100_000,
        verbose: bool = False,
        tools: list[Callable] | None = None,
        sub_lm: dspy.LM | None = None,
        interpreter: SubprocessInterpreter | None = None,
        sandbox_code: str | None = None,
        action_instructions: str | None = None,
        images_for_last_n: int = 1,
        max_image_pixels: int = 1_000_000,
    ):
        super().__init__()
        self.signature = ensure_signature(signature)
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        self.max_output_chars = max_output_chars
        self.verbose = verbose
        self.sub_lm = sub_lm
        self._interpreter = interpreter
        self._sandbox_code = sandbox_code
        self._action_instructions = action_instructions or DEFAULT_ACTION_INSTRUCTIONS
        self._images_for_last_n = images_for_last_n
        self._max_image_pixels = max_image_pixels
        self._user_tools = self._normalize_tools(tools)
        self._validate_tools(self._user_tools)

        # Patch ChatAdapter to avoid 8-space indentation of instructions
        self._patch_chat_adapter()

        action_sig, extract_sig = self._build_signatures()
        self.generate_action = dspy.Predict(action_sig)
        self.extract = dspy.Predict(extract_sig)

    # =========================================================================
    # Adapter Patching
    # =========================================================================

    @staticmethod
    def _patch_chat_adapter():
        """Patch ChatAdapter.format_task_description to not indent instructions with 8 spaces."""
        from dspy.adapters.chat_adapter import ChatAdapter

        def format_task_description(self, signature):
            instructions = signature.instructions
            if not instructions:
                return ""
            return instructions

        ChatAdapter.format_task_description = format_task_description

    # =========================================================================
    # Tool Creation and Validation
    # =========================================================================

    _RESERVED_TOOL_NAMES = frozenset({"SUBMIT", "RESET_HISTORY", "print", "display"})

    def _normalize_tools(self, tools: list[Callable] | None) -> dict[str, Callable]:
        if not tools:
            return {}
        result = {}
        for func in tools:
            if not callable(func):
                raise TypeError(f"Tool {func!r} must be callable")
            name = getattr(func, "__name__", None) or str(func)
            result[name] = func
        return result

    def _validate_tools(self, tools: dict[str, Callable]) -> None:
        for name in tools:
            if not name.isidentifier():
                raise ValueError(f"Invalid tool name '{name}'")
            if name in self._RESERVED_TOOL_NAMES:
                raise ValueError(f"Tool name '{name}' conflicts with built-in")

    def _format_tool_docs(self, tools: dict[str, Callable]) -> str:
        if not tools:
            return ""
        lines = []
        for name, func in tools.items():
            # Hide internal tools (prefixed with _) from the LLM prompt
            if name.startswith("_"):
                continue
            try:
                sig = inspect.signature(func)
                params = []
                for p in sig.parameters.values():
                    if p.annotation != inspect.Parameter.empty:
                        type_name = getattr(p.annotation, "__name__", str(p.annotation))
                        params.append(f"{p.name}: {type_name}")
                    else:
                        params.append(p.name)
                params_str = ", ".join(params)
                if sig.return_annotation != inspect.Parameter.empty:
                    ret_type = getattr(sig.return_annotation, "__name__", str(sig.return_annotation))
                    sig_str = f"{name}({params_str}) -> {ret_type}"
                else:
                    sig_str = f"{name}({params_str})"
            except (ValueError, TypeError):
                sig_str = f"{name}(...)"
            doc = func.__doc__.strip().split("\n")[0] if func.__doc__ else "No description"
            lines.append(f"- `{sig_str}` - {doc}")
        if not lines:
            return ""
        return "\nAdditional tools available:\n" + "\n".join(lines)

    @property
    def tools(self) -> dict[str, Callable]:
        return dict(self._user_tools)

    # =========================================================================
    # Signature Building
    # =========================================================================

    def _build_signatures(self) -> tuple[Signature, Signature]:
        inputs_str = ", ".join(f"`{n}`" for n in self.signature.input_fields)
        final_output_names = ", ".join(self.signature.output_fields.keys())
        output_fields = "\n".join(
            f"- {translate_field_type(n, f)}"
            for n, f in self.signature.output_fields.items()
        )
        task_instructions = f"{self.signature.instructions}\n\n" if self.signature.instructions else ""
        tool_docs = self._format_tool_docs(self._user_tools)

        action_sig = (
            dspy.Signature({}, task_instructions + self._action_instructions.format(
                inputs=inputs_str, final_output_names=final_output_names, output_fields=output_fields,
                max_iterations=self.max_iterations,
            ) + tool_docs)
            .append("variables_info", dspy.InputField(desc="Metadata about the variables available in the REPL"), type_=str)
            .append("repl_history", dspy.InputField(desc="Previous REPL code executions and their outputs"), type_=VisualREPLHistory)
            .append("iteration", dspy.InputField(desc="Current iteration number (1-indexed) out of max_iterations"), type_=str)
            .append("code", dspy.OutputField(desc="Python code to execute."), type_=str)
        )

        extract_instructions = """Based on the REPL trajectory, extract the final outputs now.

            Review your trajectory to see what information you gathered and what values you computed, then provide the final outputs.
            IMPORTANT: Output ONLY the raw answer value — no sentences, no explanations, no reasoning. Just the value."""

        extended_task_instructions = ""
        if task_instructions:
            extended_task_instructions = "The trajectory was generated with the following objective: \n" + task_instructions + "\n"
        full_extract_instructions = extended_task_instructions + extract_instructions

        extract_sig = dspy.Signature(
            {**self.signature.output_fields},
            full_extract_instructions,
        )
        extract_sig = extract_sig.prepend("repl_history", dspy.InputField(desc="Your REPL interactions so far"), type_=VisualREPLHistory)
        extract_sig = extract_sig.prepend("variables_info", dspy.InputField(desc="Metadata about the variables available in the REPL"), type_=str)

        return action_sig, extract_sig

    # =========================================================================
    # Input/Output Processing
    # =========================================================================

    def _get_output_fields_info(self) -> list[dict]:
        fields = []
        for name, field in self.signature.output_fields.items():
            annotation = getattr(field, "annotation", str)
            field_info = {"name": name}
            if annotation in SIMPLE_TYPES:
                field_info["type"] = annotation.__name__
            fields.append(field_info)
        return fields

    def _build_variables(self, **input_args: Any) -> list[dict]:
        """Build variable metadata for the LLM prompt."""
        variables = []
        for name, value in input_args.items():
            field_info = self.signature.input_fields.get(name)
            desc = ""
            if field_info and hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
                desc = field_info.json_schema_extra.get("desc", "")
            val_repr = repr(value)
            if len(val_repr) > 200:
                val_repr = val_repr[:200] + "..."
            variables.append({"name": name, "type": type(value).__name__, "desc": desc, "preview": val_repr})
        return variables

    def _format_output(self, output: str) -> str:
        if not output:
            return "(no output - did you forget to print?)"
        if len(output) > self.max_output_chars:
            return output[:self.max_output_chars] + "\n... (truncated)"
        return output

    def _validate_inputs(self, input_args: dict[str, Any]) -> None:
        missing = set(self.signature.input_fields.keys()) - set(input_args.keys())
        if missing:
            raise ValueError(f"Missing required inputs: {sorted(missing)}")

    # =========================================================================
    # CodeInterpreter Lifecycle
    # =========================================================================

    def _prepare_execution_tools(self) -> dict[str, Callable]:
        return dict(self._user_tools)

    @contextmanager
    def _interpreter_context(self, execution_tools: dict[str, Callable]) -> Iterator[SubprocessInterpreter]:
        if self._interpreter is not None:
            self._interpreter.tools.update(execution_tools)
            yield self._interpreter
        else:
            repl = SubprocessInterpreter(
                tools=execution_tools,
                output_fields=self._get_output_fields_info(),
                timeout=120.0,
                sandbox_code=self._sandbox_code,
                display_max_pixels=self._max_image_pixels,
            )
            try:
                yield repl
            finally:
                repl.shutdown()

    # =========================================================================
    # Execution Core
    # =========================================================================

    def _extract_fallback(
        self,
        variables: list[dict],
        history: VisualREPLHistory,
        output_field_names: list[str],
    ) -> Prediction:
        logger.warning("RVLM reached max iterations, using extract to get final output")
        # Strip images for extract — not needed for summarization
        no_image_history = VisualREPLHistory(
            entries=list(history.entries),
            images_for_last_n=0,
        )
        extract_pred = self.extract(
            variables_info=str(variables),
            repl_history=no_image_history,
        )
        return Prediction(
            trajectory=[e.model_dump() for e in history],
            final_reasoning="Extract forced final output",
            **{name: getattr(extract_pred, name) for name in output_field_names},
        )

    def _process_final_output(
        self,
        result: FinalOutput,
        output_field_names: list[str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        raw_output = result.output
        if not isinstance(raw_output, dict):
            return None, f"[Error] FINAL returned {type(raw_output).__name__}, expected dict with fields: {output_field_names}"
        missing = set(output_field_names) - set(raw_output.keys())
        if missing:
            return None, f"[Error] Missing output fields: {sorted(missing)}. Use SUBMIT({', '.join(output_field_names)})"
        parsed_outputs = {}
        type_errors = []
        for name in output_field_names:
            field = self.signature.output_fields[name]
            annotation = getattr(field, "annotation", str)
            try:
                parsed_outputs[name] = parse_value(raw_output[name], annotation)
            except (ValueError, pydantic.ValidationError) as e:
                type_errors.append(
                    f"{name}: expected {annotation.__name__ if hasattr(annotation, '__name__') else annotation}, "
                    f"got {type(raw_output[name]).__name__}: {e}"
                )
        if type_errors:
            return None, "[Type Error] " + "; ".join(type_errors)
        return parsed_outputs, None

    def _process_execution_result(
        self,
        pred: Any,
        result: Any,
        history: VisualREPLHistory,
        output_field_names: list[str],
        images: list[dict],
    ) -> Prediction | VisualREPLHistory:
        code = _strip_code_fences(pred.code) if pred.code else ""

        # Handle HistoryReset signal
        if isinstance(result, HistoryReset):
            return VisualREPLHistory(images_for_last_n=self._images_for_last_n).append(
                code=code,
                output=f"[History compacted] {result.summary}",
            )

        # Handle error strings
        if isinstance(result, str) and result.startswith("[Error]"):
            output = self._format_output(result)
            return history.append(code=code, output=output, images=images)

        # Handle FinalOutput — SubprocessInterpreter returns (FinalOutput, captured_stdout) tuple
        final_output = None
        captured_stdout = ""
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], FinalOutput):
            final_output, captured_stdout = result
        elif isinstance(result, FinalOutput):
            final_output = result

        if final_output is not None:
            parsed_outputs, error = self._process_final_output(final_output, output_field_names)
            if error:
                output = f"{captured_stdout}\n{error}" if captured_stdout else error
                return history.append(code=code, output=output, images=images)
            final_history = history.append(
                code=code, output=f"FINAL: {parsed_outputs}", images=images
            )
            return Prediction(
                **parsed_outputs,
                trajectory=[e.model_dump() for e in final_history],
                final_reasoning="",
            )

        # Normal string output
        if isinstance(result, list):
            output = "\n".join(map(str, result))
        else:
            output = str(result) if result else ""

        output = self._format_output(output)
        return history.append(code=code, output=output, images=images)

    def _execute_iteration(
        self,
        repl: SubprocessInterpreter,
        variables: list[dict],
        history: VisualREPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
    ) -> Prediction | VisualREPLHistory:
        try:
            action = self.generate_action(
                variables_info=str(variables),
                repl_history=history,
                iteration=f"{iteration + 1}/{self.max_iterations}",
            )
        except AdapterParseError as e:
            logger.warning("RVLM iteration %d/%d: parse error after adapter retries: %s", iteration + 1, self.max_iterations, e)
            history.append(VisualREPLEntry(code="", output=f"[Error] {e}", images=[]))
            return history
        # Coerce None code to empty string
        if action.code is None:
            action.code = ""

        if self.verbose:
            logger.info(
                f"RVLM iteration {iteration + 1}/{self.max_iterations}\n"
                f"Code:\n{action.code}"
            )

        try:
            code = _strip_code_fences(action.code)
            if not code:
                result = "[Error] No code provided. Write Python code to execute."
            else:
                result = repl.execute(code, variables=dict(input_args))
        except (CodeInterpreterError, SyntaxError) as e:
            result = f"[Error] {e}"

        # Collect any images displayed during this execution
        images = repl.pop_images()

        return self._process_execution_result(action, result, history, output_field_names, images)

    # =========================================================================
    # Public Interface
    # =========================================================================

    def forward(self, **input_args) -> Prediction:
        self._validate_inputs(input_args)
        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools) as repl:
            history = VisualREPLHistory(images_for_last_n=self._images_for_last_n)

            for iteration in range(self.max_iterations):
                result: Prediction | VisualREPLHistory = self._execute_iteration(
                    repl, variables, history, iteration, input_args, output_field_names
                )
                if isinstance(result, Prediction):
                    return result
                history = result

            return self._extract_fallback(variables, history, output_field_names)
