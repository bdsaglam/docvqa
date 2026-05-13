"""Per-dataset configuration ("profile") for the solver + eval loop.

A ``DatasetProfile`` bundles every benchmark-specific knob that an
otherwise-generic solver/runner needs:

- **Answer-formatting rules** prepended to the agent's task instructions.
  DocVQA-2026's are rich (unit normalization, date format, percentage
  rules). MP-DocVQA wants short faithful spans. MMLongBench-Doc has 5
  formal answer formats, and the right one is per-question.
- **Per-category tips**: DocVQA-2026 has hand-tuned tips by category;
  MP-DocVQA and MMLongBench-Doc each have a single category and the
  tips would misfire.
- **Per-question format hint**: For datasets like MMLongBench-Doc that
  ship a per-question answer_format, the solver can show it inline.
- **Scorer**: DocVQA-2026 / MP-DocVQA use ANLS-based
  :func:`docvqa.metrics.evaluate_prediction`; MMLongBench-Doc uses a
  Qwen-judge call against the official extraction+score protocol.

Look up a profile with :func:`get_profile(dataset_name)`. New datasets
fall back to the DocVQA-2026 profile by default — register them in
``_PROFILES`` when adding a new loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from docvqa.data import Question
from docvqa.metrics import evaluate_prediction
from docvqa.prompts import (
    ANSWER_FORMATTING_RULES,
    get_baseline_category_tips,
    get_category_tips,
)

ScoreFn = Callable[[str, Optional[str], Question], tuple[bool, str]]
TipsFn = Callable[[str], str]
HintFn = Callable[[Question], Optional[str]]


def _anls_score(pred: str, gt: str | None, question: Question) -> tuple[bool, str]:
    """ANLS-based scorer — the DocVQA-2026 default.

    Used by DocVQA-2026 (multi-aliases handled by ``evaluate_prediction``'s
    ``ast.literal_eval`` round-trip) and MP-DocVQA (loader stores
    multi-alias answers as ``repr(list)``).
    """
    if gt is None:
        return False, pred.strip()
    return evaluate_prediction(pred, gt)


def _no_tips(category: str) -> str:
    """No-op category-tips function — for benchmarks without categories."""
    return ""


@dataclass
class DatasetProfile:
    """Bundle of dataset-specific knobs for solver + runner.

    Fields:
        name: short slug for logging / config; should match the
            ``data/<slug>`` directory.
        answer_formatting_rules: text block appended to task instructions.
        category_tips_fn: returns per-category tips ("" if none).
        score_fn: callable used by the runner to compute per-question
            correctness. Defaults to ANLS.
        question_format_hint_fn: optional hint string inserted into the
            per-question prompt. Used to surface MMLongBench-Doc's
            ``answer_format`` so the agent picks the right formatter.
    """

    name: str
    answer_formatting_rules: str = ANSWER_FORMATTING_RULES
    category_tips_fn: TipsFn = field(default_factory=lambda: get_category_tips)
    # Separate tips for the raw-VLM baseline — DocVQA-2026 tunes these
    # differently than the scaffold tips. For benchmarks with a single
    # doc category, both are ``_no_tips``.
    baseline_category_tips_fn: TipsFn = field(default_factory=lambda: get_baseline_category_tips)
    score_fn: ScoreFn = field(default_factory=lambda: _anls_score)
    question_format_hint_fn: HintFn | None = None


# ---------------------------------------------------------------------------
# DocVQA-2026 (project default)
# ---------------------------------------------------------------------------

DOCVQA_2026_PROFILE = DatasetProfile(name="docvqa-2026")


# ---------------------------------------------------------------------------
# MP-DocVQA — short faithful spans, ANLS, no category tips
# ---------------------------------------------------------------------------

MP_DOCVQA_FORMATTING = (
    "## ANSWER FORMATTING\n"
    "Output a short answer string copied as faithfully as possible from the document.\n"
    "- Do not paraphrase; quote spans verbatim where possible.\n"
    "- Preserve the document's own number / currency / date representation.\n"
    "- For multi-item answers, separate with ', '.\n"
    "- If the answer is not in the document, output exactly: Unknown\n"
    "- Output ONLY the answer string. No preamble, no explanation.\n"
)

MP_DOCVQA_PROFILE = DatasetProfile(
    name="mp-docvqa",
    answer_formatting_rules=MP_DOCVQA_FORMATTING,
    category_tips_fn=_no_tips,
    baseline_category_tips_fn=_no_tips,
)


# ---------------------------------------------------------------------------
# MMLongBench-Doc — 5 formal answer formats, Qwen judge, per-question hint
# ---------------------------------------------------------------------------

MMLB_FORMATTING = (
    "## ANSWER FORMATTING\n"
    "MMLongBench-Doc answers fall into 5 formats. Use the right one for each question:\n"
    "- **Integer**: bare integer, no units, no punctuation. Example: 42\n"
    "- **Float**: a decimal value. If the document shows a percentage, you may output 25%% "
    "or 0.25 — keep ~1%% precision.\n"
    "- **String**: a short text span. Strip leading articles ('the', 'a'), "
    "surrounding quotes, and trailing units when the question already names them.\n"
    "- **List**: items separated by ', ' (no 'and'). Match the document's order.\n"
    "- **Not answerable**: output literally: Not answerable\n"
    "If you cannot answer from the document, output: Not answerable\n"
    "Do NOT paraphrase numbers or include extra explanation — just the final answer.\n"
)

_MMLB_FORMAT_LONG = {
    "Int": "Integer",
    "Float": "Float",
    "Str": "String",
    "List": "List",
    "None": "Not answerable",
}


def _mmlb_question_hint(q: Question) -> str | None:
    meta = getattr(q, "mmlb", None)
    if meta is None:
        return None
    fmt = getattr(meta, "answer_format", None)
    if fmt is None:
        return None
    long = _MMLB_FORMAT_LONG.get(fmt, fmt)
    return f"Expected answer format: **{long}**."


def _mmlb_judge_score(pred: str, gt: str | None, question: Question) -> tuple[bool, str]:
    # Late import to avoid pulling openai into OCR / data-prep code paths.
    from docvqa.judges.qwen_judge import qwen_judge

    extracted = pred.strip()
    if extracted.startswith("FINAL ANSWER:"):
        extracted = extracted[len("FINAL ANSWER:"):].strip()

    if gt is None:
        return False, extracted

    meta = getattr(question, "mmlb", None)
    fmt = getattr(meta, "answer_format", "Str") if meta is not None else "Str"
    is_correct, _ = qwen_judge(
        question=question.question,
        ground_truth=gt,
        prediction=extracted,
        answer_format=fmt,
    )
    return is_correct, extracted


MMLONGBENCH_PROFILE = DatasetProfile(
    name="mmlongbench-doc",
    answer_formatting_rules=MMLB_FORMATTING,
    category_tips_fn=_no_tips,
    baseline_category_tips_fn=_no_tips,
    score_fn=_mmlb_judge_score,
    question_format_hint_fn=_mmlb_question_hint,
)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_PROFILES: dict[str, DatasetProfile] = {
    "VLR-CVC/DocVQA-2026": DOCVQA_2026_PROFILE,
    "lmms-lab/MP-DocVQA": MP_DOCVQA_PROFILE,
    "yubo2333/MMLongBench-Doc": MMLONGBENCH_PROFILE,
}


def get_profile(dataset_name: str) -> DatasetProfile:
    """Return the registered :class:`DatasetProfile` for a HF dataset id.

    Unknown ids fall back to :data:`DOCVQA_2026_PROFILE`.
    """
    return _PROFILES.get(dataset_name, DOCVQA_2026_PROFILE)
