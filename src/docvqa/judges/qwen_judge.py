"""Local Qwen-as-judge for MMLongBench-Doc style evaluation.

MMLongBench-Doc's official protocol is a two-stage thing:
  1. GPT-4o extracts a clean ``Extracted answer`` from the free-form model
     analysis, conditioned on the question and the expected answer format
     (Integer / Float / String / List / Not answerable / Fail to answer).
  2. A deterministic ``eval_score`` compares extracted vs GT based on format
     (exact for Int, isclose for Float, ANLS for Str/None, ordered ANLS
     for List).

For this scaffold we want a *single* local-model call that does both jobs:
extract from the prediction and decide correct / incorrect. Reasons:
  - We run thousands of judging calls and don't want two LM hops.
  - Our predictions are typically already short (no "Analysis:" chain),
    so the extraction stage is closer to a normalization step.
  - The pipeline must work fully offline against the local Qwen vLLM.

The prompt below is a direct adaptation of the official extraction prompt
(see https://github.com/mayubo2333/MMLongBench-Doc/blob/main/eval/prompt_for_answer_extraction.md)
with an added scoring instruction at the end. We mirror their handling of
"Not answerable" and "Fail to answer".
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = os.environ.get("QWEN_JUDGE_BASE_URL", "http://localhost:8928/v1")
DEFAULT_MODEL = os.environ.get("QWEN_JUDGE_MODEL", "Qwen/Qwen3.5-27B")
DEFAULT_API_KEY = os.environ.get("QWEN_JUDGE_API_KEY", "dummy")


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for document-VQA answers.

Given a question, the expected answer format, the ground-truth answer, and a
candidate prediction, decide whether the prediction is correct.

Rules adapted from MMLongBench-Doc:
- Expected answer formats: Integer, Float, String, List, Not answerable.
- For "Integer": numeric equality after stripping units / punctuation.
- For "Float": equal within ~1% relative tolerance; treat percentages
  liberally (e.g. "0.25" == "25%" == "25").
- For "String": semantic match. Ignore casing, articles, trailing units
  ("miles", "million", "$"), and surrounding quotes / parentheses.
- For "List": all required items must be present; order does not matter
  unless the question implies it; ignore casing and trivial punctuation.
- "Not answerable" GT: the prediction must indicate the question cannot be
  answered from the document (e.g. "Not answerable", "Unknown", "Cannot be
  determined"). A specific wrong answer is incorrect. "Fail to answer"
  (the model refused/could not read the doc) is also INCORRECT against a
  factual GT, but correct against a "Not answerable" GT only if it
  clearly says the question is unanswerable rather than that the model
  itself failed.
- Be strict about numbers and named entities; be lenient about phrasing.

Respond in EXACTLY this format and nothing else:

Extracted answer: <one-line normalized version of the prediction, or "Not answerable" / "Fail to answer">
Answer format: <one of: Integer | Float | String | List | Not answerable>
Verdict: <correct | incorrect>
"""


JUDGE_USER_TEMPLATE = """Question: {question}
Expected answer format: {answer_format}
Ground truth: {ground_truth}
Prediction: {prediction}
"""


_VERDICT_RE = re.compile(r"verdict\s*:\s*(correct|incorrect)", re.IGNORECASE)
_EXTRACTED_RE = re.compile(r"extracted answer\s*:\s*(.+)", re.IGNORECASE)


# Map HF dataset format codes to the names used in the prompt.
_FORMAT_NAME = {
    "Int": "Integer",
    "Float": "Float",
    "Str": "String",
    "List": "List",
    "None": "Not answerable",
}


@dataclass
class JudgeResult:
    is_correct: bool
    verdict_raw: str  # full LM response
    extracted_answer: str  # may be empty if parsing failed


def _make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def qwen_judge(
    question: str,
    ground_truth: str,
    prediction: str,
    answer_format: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    client: OpenAI | None = None,
    temperature: float = 0.0,
    max_tokens: int = 200,
    enable_thinking: bool = False,
) -> tuple[bool, str]:
    """Score one (question, gt, prediction, answer_format) triple.

    Returns ``(is_correct, judge_response)`` where ``judge_response`` is the
    raw model reply (parseable for ``Extracted answer`` and ``Verdict``).

    ``answer_format`` accepts either the HF code (``Int``, ``Str``, ...) or
    the long name (``Integer``, ``String``, ...). Unknown values are passed
    through verbatim — the judge prompt is tolerant.
    """
    if client is None:
        client = _make_client(base_url, api_key)

    fmt = _FORMAT_NAME.get(answer_format, answer_format)
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question.strip(),
        answer_format=fmt,
        ground_truth=str(ground_truth).strip(),
        prediction=str(prediction).strip(),
    )

    extra_body = {}
    if not enable_thinking:
        # vLLM-served Qwen3.5 honours this flag to skip the thinking trace.
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body or None,
        )
    except Exception as exc:
        logger.exception("qwen_judge: request failed; defaulting to incorrect: %s", exc)
        return False, f"[judge-error] {exc}"

    text = resp.choices[0].message.content or ""
    # Some thinking-style models stash the visible answer in `reasoning_content`.
    if not text.strip():
        reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
        if reasoning:
            text = reasoning

    m = _VERDICT_RE.search(text)
    is_correct = bool(m and m.group(1).lower() == "correct")
    return is_correct, text


def parse_judge_response(text: str) -> JudgeResult:
    """Parse a stored judge response into its components."""
    verdict_m = _VERDICT_RE.search(text or "")
    extracted_m = _EXTRACTED_RE.search(text or "")
    return JudgeResult(
        is_correct=bool(verdict_m and verdict_m.group(1).lower() == "correct"),
        verdict_raw=text or "",
        extracted_answer=(extracted_m.group(1).strip() if extracted_m else ""),
    )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    ok, resp = qwen_judge(
        question="What is the answer to life, the universe, and everything?",
        ground_truth="42",
        prediction="42",
        answer_format="Int",
    )
    print("ok =", ok)
    print(resp)
