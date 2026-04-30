"""Thinking RLM — code-only output, reasoning populated from LLM thinking tokens.

Like CodeRLM, the model only generates a `code` output field. But unlike CodeRLM,
this variant extracts the model's reasoning/thinking tokens (via reasoning_content
in the LLM response) and records them in the REPL trajectory.

This gives the best of both worlds: no wasted output tokens on explicit reasoning
fields, but full visibility into the model's chain-of-thought via thinking tokens.

Requires the LM to have thinking/reasoning enabled (e.g. enable_thinking=True for
Qwen models via vLLM).
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable

import pydantic

import dspy
from dspy.adapters.utils import parse_value, translate_field_type
from dspy.primitives.code_interpreter import SIMPLE_TYPES, CodeInterpreterError, FinalOutput
from dspy.utils.exceptions import AdapterParseError
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.primitives.repl_types import REPLHistory, REPLVariable
from dspy.signatures.signature import ensure_signature

from docvqa.rlm.subprocess_interpreter import HistoryReset, SubprocessInterpreter

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)

DEFAULT_ACTION_INSTRUCTIONS = """Produce the following outputs given the inputs {inputs}:
{output_fields}

You have access to a Python REPL environment. Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.

IMPORTANT: Write raw Python code only. No need to wrap code in markdown fences.

Available:
- Variables: {inputs} (your input data)
- `print()` - ALWAYS print to see results
- `SUBMIT({final_output_names})` - call when done (ends the run immediately)
- Standard libraries: re, json, collections, math, etc.

Rules:
- This is ITERATIVE. State persists between steps. Do NOT solve everything in one step.
- ALWAYS print before submitting — verify results look correct.
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


def _extract_reasoning_content(lm: dspy.LM | None) -> str:
    """Extract reasoning_content from the last LM history entry.

    When the LLM has thinking/reasoning enabled, dspy stores
    reasoning_content in the history outputs dict (see base_lm.py).
    """
    if lm is None or not lm.history:
        return ""
    entry = lm.history[-1]
    outputs = entry.get("outputs", [])
    if not outputs:
        return ""
    first = outputs[0]
    if isinstance(first, dict):
        return first.get("reasoning_content", "") or ""
    return ""


class ThinkingRLM(Module):
    """Thinking RLM — code-only output, reasoning from LLM thinking tokens."""

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
        self._user_tools = self._normalize_tools(tools)
        self._validate_tools(self._user_tools)

        self._patch_chat_adapter()

        action_sig, extract_sig = self._build_signatures()
        self.generate_action = dspy.Predict(action_sig)
        self.extract = dspy.Predict(extract_sig)

    @staticmethod
    def _patch_chat_adapter():
        from dspy.adapters.chat_adapter import ChatAdapter

        def format_task_description(self, signature):
            instructions = signature.instructions
            if not instructions:
                return ""
            return instructions

        ChatAdapter.format_task_description = format_task_description

    _RESERVED_TOOL_NAMES = frozenset({"SUBMIT", "RESET_HISTORY", "print"})

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

    def _build_signatures(self) -> tuple[Signature, Signature]:
        inputs_str = ", ".join(f"`{n}`" for n in self.signature.input_fields)
        final_output_names = ", ".join(self.signature.output_fields.keys())
        output_fields = "\n".join(
            f"- {translate_field_type(n, f)}"
            for n, f in self.signature.output_fields.items()
        )
        task_instructions = f"{self.signature.instructions}\n\n" if self.signature.instructions else ""
        tool_docs = self._format_tool_docs(self._user_tools)

        # Code-only output: reasoning comes from thinking tokens, not an output field
        action_sig = (
            dspy.Signature({}, task_instructions + self._action_instructions.format(
                inputs=inputs_str, final_output_names=final_output_names, output_fields=output_fields,
                max_iterations=self.max_iterations,
            ) + tool_docs)
            .append("variables_info", dspy.InputField(desc="Metadata about the variables available in the REPL"), type_=str)
            .append("repl_history", dspy.InputField(desc="Previous REPL code executions and their outputs"), type_=REPLHistory)
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
        extract_sig = extract_sig.prepend("repl_history", dspy.InputField(desc="Your REPL interactions so far"), type_=REPLHistory)
        extract_sig = extract_sig.prepend("variables_info", dspy.InputField(desc="Metadata about the variables available in the REPL"), type_=str)

        return action_sig, extract_sig

    def _get_output_fields_info(self) -> list[dict]:
        fields = []
        for name, field in self.signature.output_fields.items():
            annotation = getattr(field, "annotation", str)
            field_info = {"name": name}
            if annotation in SIMPLE_TYPES:
                field_info["type"] = annotation.__name__
            fields.append(field_info)
        return fields

    def _build_variables(self, **input_args: Any) -> list[REPLVariable]:
        variables = []
        for name, value in input_args.items():
            field_info = self.signature.input_fields.get(name)
            variables.append(REPLVariable.from_value(name, value, field_info=field_info))
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
            )
            try:
                yield repl
            finally:
                repl.shutdown()

    def _get_active_lm(self) -> dspy.LM | None:
        """Get the LM that generate_action uses (mirrors Predict._forward_preprocess)."""
        return self.generate_action.lm or dspy.settings.lm

    def _extract_fallback(
        self,
        variables: list[REPLVariable],
        history: REPLHistory,
        output_field_names: list[str],
    ) -> Prediction:
        logger.warning("ThinkingRLM reached max iterations, using extract to get final output")
        variables_info = [variable.format() for variable in variables]
        extract_pred = self.extract(
            variables_info=variables_info,
            repl_history=history,
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
        history: REPLHistory,
        output_field_names: list[str],
        reasoning: str,
    ) -> Prediction | REPLHistory:
        code = _strip_code_fences(pred.code) if pred.code else ""

        if isinstance(result, HistoryReset):
            return REPLHistory().append(
                reasoning=reasoning,
                code=code,
                output=f"[History compacted] {result.summary}",
            )

        if isinstance(result, str) and result.startswith("[Error]"):
            output = self._format_output(result)
            return history.append(reasoning=reasoning, code=code, output=output)

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
                return history.append(reasoning=reasoning, code=code, output=output)
            final_history = history.append(
                reasoning=reasoning, code=code, output=f"FINAL: {parsed_outputs}"
            )
            return Prediction(
                **parsed_outputs,
                trajectory=[e.model_dump() for e in final_history],
                final_reasoning=reasoning,
            )

        if isinstance(result, list):
            output = "\n".join(map(str, result))
        else:
            output = str(result) if result else ""

        output = self._format_output(output)
        return history.append(reasoning=reasoning, code=code, output=output)

    def _execute_iteration(
        self,
        repl: SubprocessInterpreter,
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        variables_info = [variable.format() for variable in variables]
        try:
            action = self.generate_action(
                variables_info=variables_info,
                repl_history=history,
                iteration=f"{iteration + 1}/{self.max_iterations}",
            )
        except AdapterParseError as e:
            logger.warning("ThinkingRLM iteration %d/%d: parse error: %s", iteration + 1, self.max_iterations, e)
            history.append({"role": "assistant", "content": f"[Error] {e}"})
            return history

        # Extract reasoning from LLM thinking tokens (reasoning_content)
        reasoning = _extract_reasoning_content(self._get_active_lm())

        if action.code is None:
            action.code = ""

        if self.verbose:
            logger.info(
                f"ThinkingRLM iteration {iteration + 1}/{self.max_iterations}\n"
                f"Thinking: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}\n"
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

        return self._process_execution_result(action, result, history, output_field_names, reasoning)

    def forward(self, **input_args) -> Prediction:
        self._validate_inputs(input_args)
        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools) as repl:
            history: REPLHistory = REPLHistory()

            for iteration in range(self.max_iterations):
                result: Prediction | REPLHistory = self._execute_iteration(
                    repl, variables, history, iteration, input_args, output_field_names
                )
                if isinstance(result, Prediction):
                    return result
                history = result

            return self._extract_fallback(variables, history, output_field_names)
