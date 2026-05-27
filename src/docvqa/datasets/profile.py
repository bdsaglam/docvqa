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

Per D-009 (docs/paper/decisions.md, 2026-05-27), per-dataset
*tool-agnostic semantic* prompt content lives here. Tool-routing verbs
(`batch_look`, `look`, `search`, `page_texts`) belong inside individual
solvers' `TASK_INSTRUCTIONS` and optional per-category overlays, not in
the profile. The DocVQA-2026 dicts below are derived from
`leanest_solo_solver.py`'s reconciled CATEGORY_TIPS with tool-call
references stripped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from docvqa.data import Question
from docvqa.metrics import evaluate_prediction

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


# ---------------------------------------------------------------------------
# DocVQA-2026 — answer-formatting rules
# ---------------------------------------------------------------------------
# Verbatim from the historical ``docvqa.prompts.ANSWER_FORMATTING_RULES``
# (kept there as DEPRECATED for shelved-solver back-compat). The profile
# is now the source of truth.

_DOCVQA_2026_ANSWER_FORMATTING_RULES = (
    "## ANSWER FORMATTING RULES\n"
    "Source Adherence: Only provide answers found directly within the document. "
    "If the question is unanswerable given the provided image, the response must be exactly: Unknown\n"
    "Multiple Answers: List multiple answers in their order of appearance, "
    "separated by a comma and a single space. Do not use the word \"and\".\n"
    "Example: Answer A, Answer B\n"
    "Numbers & Units: Convert units to their standardized abbreviations "
    "(e.g., use kg instead of \"kilograms\", m instead of \"meters\"). "
    "Always place a single space between the number and the unit.\n"
    "Example: 50 kg, 10 USD\n"
    "Percentages: Attach the % symbol directly to the number with no space.\n"
    "Example: 50%\n"
    "Dates: Convert all dates to the standardized YYYY-MM-DD format.\n"
    "Example: \"Jan 1st 24\" becomes 2024-01-01\n"
    "Decimals: Use a single period (.) as a decimal separator, never a comma.\n"
    "Example: 3.14\n"
    "Thousands Separator: Remove commas and spaces from within numbers.\n"
    "Example: 713809, not 713,809 or 713 809\n"
    "Percentage Differences: When asked for a 'percentage difference' or 'difference in percentages' "
    "between two percentage values, return the absolute difference in percentage points "
    "(e.g., 15% vs 11% → 4%), NOT the relative change (not 36.36%). "
    "Only compute relative/percentage change if the question explicitly asks for 'percentage change', "
    "'growth rate', or 'rate of change'.\n"
    "No Filler Text: Output only the requested data. Do not frame your answer "
    "in full sentences (e.g., avoid \"The answer is...\").\n"
)


# ---------------------------------------------------------------------------
# DocVQA-2026 — per-category tips (scaffold path, tool-agnostic)
# ---------------------------------------------------------------------------
# Source: ``leanest_solo_solver.py`` CATEGORY_TIPS (the 2026-05-27 reconciled
# version with the 4 cross-solver-canonical bullets). Tool-routing verbs
# (`batch_look(...)`, `pages[i]`, `.crop((l,t,r,b))`) replaced with
# tool-agnostic phrasing per D-009. References to "the VLM" are kept — the
# VLM is the project's modeling stack, not a solver-specific tool.

_DOCVQA_2026_CATEGORY_TIPS: dict[str, str] = {
    "engineering_drawing": (
        "- PRECISION IS CRITICAL: Crop tables and labels at full resolution before reading values.\n"
        "- BOM has two parallel numbering systems: ITEM NUMBERS (sequential index in the parts list) and "
        "PART / IDENTIFYING NUMBERS (the actual hardware identifier, often alphanumeric with dashes). "
        "Questions about 'part number' / 'identifying number' refer to the latter; 'item number' refers to the former.\n"
        "- 'VIEW IN DIRECTION X' labels indicate a viewing direction. The answer is the direction letter alone, "
        "not prefixed with 'Direction'.\n"
        "- For counting parts in the BOM (clamps, bolts, etc.), read the QTY column row-by-row and sum programmatically — "
        "don't estimate visually.\n"
        "- VLM OCR CONFUSION: Part numbers are almost always digits + dashes. If the VLM reads letters like I, O, l "
        "where digits 1, 0 would be expected, re-read at higher zoom. Common confusions: I↔1, O↔0, l↔1.\n"
        "- For labels or numbers adjacent to a specific schematic or view, crop tightly around that view rather "
        "than relying on a single full-page query — small text gets lost at thumbnail resolution.\n"
        "- LEADER LINES: when a label points to a part via a leader line, inspect the label region and the "
        "pointed-to region as two separate crops to verify the connection — don't rely on a single full-page "
        "query that may mis-associate label and part by proximity.\n"
        "- DIMENSIONS: 'Width' typically refers to the shorter cross-sectional dimension (from a Section view), "
        "not the longest overall dimension (which is 'Length'). Dimensions tagged 'REF' (reference) are valid answers.\n"
    ),
    "business_report": (
        "- Crop tables at full resolution before reading numbers or labels — dense tables are hard to read at thumbnail zoom.\n"
        "- Multiple tables may contain similar-looking data. Verify the table's title/header matches the question's "
        "subject before extracting values.\n"
        "- For YoY / period-over-period calculations, extract raw values from the table first, then compute "
        "differences programmatically — do not rely on the VLM for arithmetic.\n"
        "- CHART VALUES: VLM readings of bar/line chart values vary between calls. Use the first clear reading "
        "rather than re-querying the same chart to 'verify' — repeated reads add noise, not signal.\n"
        "- 'Broken down into' refers to immediate sub-categories only, not sub-sub-categories.\n"
        "- TEXT TRUNCATION: When a question asks for a phrase truncated at a punctuation boundary "
        "(first words before a punctuation mark, first sentence, etc.), read the full passage and do the "
        "truncation programmatically — the VLM over-shortens when asked to truncate directly.\n"
        "- PICTOGRAMS: When looking for a described pictogram among many, crop each icon individually and ask the "
        "VLM to describe it, rather than asking a single yes/no filtering question across all icons at once.\n"
        "- If a qualitative description (e.g., an adjective) does not appear in the table, look in the surrounding "
        "text paragraphs or footnotes.\n"
    ),
    "comics": (
        "- STORY MAP FIRST: For multi-story anthologies, build a story index before answering — scan each page "
        "to get (story title, page range, key characters). Match question keywords to the correct story.\n"
        "- COUNTING EVENTS: For 'how many times X happens', query panel-by-panel with HIGHLY SPECIFIC inclusion "
        "criteria — e.g., 'Is someone physically [exact action] in this panel? Exclude mentions in past-tense "
        "dialogue, near-misses, and aftermath.' Then count the positive panels programmatically.\n"
        "- VERIFY COUNTS: The VLM over-attributes actions in busy panels — it infers events from context clues "
        "(sound effects, weapons, postures) even when no action is depicted. After collecting candidates, "
        "re-examine each one with a tight crop and a disconfirming question ('Did this action ACTUALLY occur, or "
        "is it a near-miss / different action / aftermath?'). Expect many initial candidates to be false positives.\n"
        "- PANEL-BY-PANEL: When you need extractable events, ask 'what happens in each panel?' explicitly. "
        "Generic 'describe the page' queries miss the panel structure that makes events countable.\n"
        "- LITERAL VS FIGURATIVE: When a question contains qualifiers like 'in reality', 'actually', or 'truly', "
        "the answer likely contradicts the surface label/title shown in the panel — distinguish what something "
        "is called from what it factually is.\n"
        "- CHARACTER IDENTIFICATION: Use the exact term that appears in the speech bubbles. When the VLM gives "
        "conflicting answers about a small object or character, use narrative context (story setting, nearby "
        "objects, character role) to disambiguate.\n"
    ),
    "maps": (
        "- COARSE-TO-FINE: Start with a full-page view of the map for rough layout, then zoom into areas of "
        "interest (~800px crops), then tighter (~400px) for small text. Each step refines the previous.\n"
        "- COUNTING OBJECTS ON MAPS: For 'how many X are on the map', NEVER try to count from a full-page view — "
        "small objects are invisible at low resolution. Instead:\n"
        "  1. Estimate the object size relative to the map and pick a grid size so each tile shows individual "
        "objects clearly (large objects → 3x3; medium → 4x4 or 5x5; small dots/pins/symbols → 6x6 or more).\n"
        "  2. Split the map into tiles with ~15% overlap between adjacent tiles.\n"
        "  3. Per-tile, ask the VLM: 'List every [object] visible in this tile, with each one's relative "
        "position (top/bottom/left/right/center) and any distinguishing label nearby.'\n"
        "  4. Collect across tiles and deduplicate objects near tile boundaries by checking similar "
        "positions or matching labels.\n"
        "  5. Count the deduplicated list.\n"
        "- LOCATE INDEPENDENTLY: Find each landmark/feature with simple per-tile queries ('what labels are "
        "visible here?', 'is feature X present in this tile?'). Record approximate pixel positions using tile "
        "offset + relative position within the tile.\n"
        "- REASON WITH MATH: Compute spatial relationships programmatically — distances, directions, relative "
        "positions — using the coordinates you collected. Basic vector math gives reliable answers with "
        "explicit error bounds.\n"
        "- LEGEND + ROAD TYPES: Crop the legend early. For road-type questions, crop the specific road segment "
        "at HIGH resolution alongside the legend and ask the VLM to compare the line style directly. Small "
        "differences (solid vs dashed, thin vs thick) are easy to misread at low resolution.\n"
        "- GRID COORDINATES: Cross-reference TWO sources — (1) crop the actual grid cell on the map to see "
        "what's there, and (2) look up the same coordinate in any feature index/legend that lists entries by "
        "grid coordinate. Disagreement between the two is usually an indexing-error trap.\n"
    ),
    "science_paper": (
        "- Papers can be long — locate the relevant section first (abstract, headings, figure/table captions) "
        "before reading in detail.\n"
        "- CITATION NUMBERS: For 'first/last citation on this page' style questions, treat citations as text "
        "patterns ([N], (Author, Year)) and enumerate them yourself in order rather than asking the VLM to "
        "identify them — VLM ordering of inline references is unreliable. Distinguish body-text citations from "
        "table headers and figure captions, which are often numbered separately.\n"
        "- CITED PAPER FINDINGS: To find what a cited work claims, first find its reference number in the "
        "bibliography (crop the references section), then locate the place(s) in body text where that number "
        "is discussed. If the cited paper's actual content isn't in this document, answer 'Unknown' rather "
        "than hallucinating from the title or context.\n"
        "- ABLATION STUDIES: Papers often contain multiple ablation studies on different components. Verify "
        "the section you're reading is about the specific component the question asks about, not a different "
        "subsystem.\n"
        "- If a question references a specific entity (layer number, model variant, dataset name) that does "
        "not appear anywhere in the document after thorough inspection, answer 'Unknown' — do not extrapolate "
        "from a similar-sounding entity.\n"
    ),
    "science_poster": (
        "- Posters are dense single-page documents. Crop specific sections at full resolution for precise values.\n"
        "- CHART ANNOTATIONS: If a chart has numeric labels printed directly on bars/lines, read those labels "
        "rather than estimating from bar heights — printed labels are exact, visual estimates are noisy.\n"
        "- For table values and percentages, crop the specific cell at full resolution before reading.\n"
        "- 'Percentage improvement' refers to the absolute difference in percentage points (e.g., 80% − 50% "
        "= 30 percentage points), not the relative change.\n"
        "- COLOR-CODED VALUES: For questions about colored numbers in a table (red, blue, highlighted), crop "
        "the table at maximum resolution and enumerate all candidates of that color before selecting — VLM "
        "color recall across an entire table is unreliable, but per-cell color checks are accurate.\n"
        "- GROUPED BAR CHARTS: A 'set of columns' / 'group of bars' refers to the bars at one x-axis position "
        "(one category, one benchmark), not all bars of one color across positions.\n"
    ),
    "infographics": (
        "- Infographics mix text, icons, and illustrations — a full-page view gives useful structural context "
        "before zooming in.\n"
        "- For precise numbers or dates, crop the specific data point at full resolution. For identifying "
        "broad visual elements (icons, sections, themes), a full-page view suffices.\n"
        "- SYSTEMATIC ENUMERATION: When a question asks for a first/last/only item that has or lacks some "
        "property, enumerate ALL items and their status before answering. Don't stop after finding two or "
        "three candidates — the answer hinges on which one is at the boundary.\n"
    ),
    "slide": (
        "- Slide decks can span many pages. Locate the relevant slide first by skimming titles/headers, then "
        "read in detail.\n"
        "- PAGE NAVIGATION: When a question refers to 'the page before X' or 'the page that contains Y', "
        "first locate X or Y, then verify the page index by cropping the page's header/title. Off-by-one "
        "errors on page indexing are common — double-check before submitting a page number or page-specific "
        "content.\n"
        "- For position-on-page questions (a specific word/bullet at the top/bottom/edge of a page), crop "
        "the relevant region at full resolution and read carefully.\n"
        "- Tables on slides are often small; crop at full resolution to read cell values.\n"
        "- EXACT ENTITY MATCHING: If a question references a specific column name, variable, or equation "
        "that does not appear anywhere in the document after thorough inspection, answer 'Unknown'. Do NOT "
        "substitute a similar-sounding name.\n"
        "- COMPUTATION: When a question says 'total', 'sum', or 'considering X and Y', extract all "
        "referenced values and compute programmatically. Show the values entering the calculation "
        "before submitting.\n"
    ),
}


# ---------------------------------------------------------------------------
# DocVQA-2026 — per-category tips (baseline / single-shot path)
# ---------------------------------------------------------------------------
# Source: ``docvqa.prompts.BASELINE_CATEGORY_TIPS`` (post-refactor reconciled
# content). These are already free of agent verbs; they describe semantic /
# question-interpretation guidance only. Used by raw-VLM / single-shot
# baselines (e.g. ``no_loop_multi``) where iterative tool use is not
# available.

_DOCVQA_2026_BASELINE_CATEGORY_TIPS: dict[str, str] = {
    "engineering_drawing": (
        "- BOM has two parallel numbering systems: ITEM NUMBERS (sequential index in the parts list) and "
        "PART / IDENTIFYING NUMBERS (the actual hardware identifier, often alphanumeric with dashes). "
        "Questions about 'part number' / 'identifying number' refer to the latter; 'item number' refers to "
        "the former.\n"
        "- 'VIEW IN DIRECTION X' labels indicate a viewing direction. The answer is the direction letter "
        "alone, not prefixed with 'Direction'.\n"
        "- OCR CONFUSION: Part numbers are almost always digits + dashes. Common confusions: I↔1, O↔0, l↔1.\n"
        "- LEADER LINES: when a label points to a part via a leader line, verify each label is correctly "
        "associated with the part it connects to — follow the line, not just proximity on the page.\n"
        "- DIMENSIONS: 'Width' typically refers to the shorter cross-sectional dimension (from a Section "
        "view), not the longest overall dimension (which is 'Length'). Dimensions tagged 'REF' (reference) "
        "are valid answers.\n"
    ),
    "business_report": (
        "- Multiple tables may contain similar-looking data. Verify the table you're reading matches the "
        "question's subject before extracting values.\n"
        "- 'Broken down into' refers to immediate sub-categories only, not sub-sub-categories.\n"
        "- TEXT TRUNCATION: For a phrase truncated at a punctuation boundary (first words before a "
        "punctuation mark, first sentence, etc.), read the full passage and do the truncation yourself — "
        "do not over-shorten.\n"
        "- If a qualitative description (e.g., an adjective) does not appear in a table, it may be in "
        "surrounding text paragraphs or footnotes.\n"
    ),
    "comics": (
        "- For multi-story anthologies, each story has its own title, page range, and characters. Match "
        "question keywords to the correct story.\n"
        "- LITERAL VS FIGURATIVE: When a question contains qualifiers like 'in reality', 'actually', or "
        "'truly', the answer likely contradicts the surface label/title — distinguish what something is "
        "called from what it factually is.\n"
        "- CHARACTER IDENTIFICATION: Use the exact term that appears in the speech bubbles when available.\n"
        "- For COUNTING EVENTS, use strict inclusion criteria — exclude near-misses, past events referenced "
        "in dialogue, and aftermath. Sound effects or weapons in a panel do not by themselves prove an "
        "action occurred.\n"
    ),
    "maps": (
        "- LEGEND: Map symbols and line styles are defined in the legend. For road-type questions, compare "
        "the line style of the specific road segment to legend entries.\n"
        "- GRID COORDINATES: Cross-reference what is visible in the grid cell with any feature index that "
        "lists entries by grid coordinate.\n"
    ),
    "science_paper": (
        "- CITATION NUMBERS: Citations appear as [N] (or (Author, Year)) in body text. Distinguish body-text "
        "citations from table headers and figure captions, which are often numbered separately.\n"
        "- CITED PAPER FINDINGS: To find what a cited work claims, locate the reference number in the "
        "bibliography, then find where that number is discussed in the body text.\n"
        "- ABLATION STUDIES: Papers often have multiple ablation studies on different components. Verify "
        "the section you're reading is about the specific component the question asks about, not a "
        "different subsystem.\n"
        "- If a question references a specific entity (layer number, model variant, dataset name) that "
        "does not appear in the document, answer 'Unknown' — do not extrapolate from a similar-sounding "
        "entity.\n"
    ),
    "science_poster": (
        "- CHART ANNOTATIONS: If a chart has numeric labels printed directly on bars/lines, use those "
        "labels rather than estimating from bar heights.\n"
        "- 'Percentage improvement' refers to the absolute difference in percentage points (e.g., 80% − "
        "50% = 30 percentage points), not the relative change.\n"
        "- GROUPED BAR CHARTS: A 'set of columns' / 'group of bars' refers to the bars at one x-axis "
        "position (one category, one benchmark), not all bars of one color across positions.\n"
    ),
    "infographics": (
        "- SYSTEMATIC ENUMERATION: When a question asks for a first/last/only item that has or lacks some "
        "property, enumerate ALL items and their status before answering — don't stop after finding a few.\n"
    ),
    "slide": (
        "- PAGE NAVIGATION: When a question refers to 'the page before X' or 'the page that contains Y', "
        "locate X or Y in the document, then take the page directly preceding or containing it. Off-by-one "
        "errors on page indexing are common — verify by checking page headers/titles.\n"
        "- EXACT ENTITY MATCHING: If a question references a specific column name, variable, or equation "
        "that does not exist in the document, answer 'Unknown'. Do NOT substitute a similar-sounding name.\n"
        "- COMPUTATION: When a question says 'total', 'sum', or 'considering X and Y', extract all "
        "referenced values and compute the result explicitly before deciding.\n"
    ),
}


def _docvqa_2026_category_tips(category: str) -> str:
    """DocVQA-2026 scaffold-path per-category tips.

    Returns the formatted block (with the canonical header) for known
    categories, or ``""`` for unknown ones.
    """
    tips = _DOCVQA_2026_CATEGORY_TIPS.get(category, "")
    if tips:
        return f"## CATEGORY-SPECIFIC TIPS ({category})\n{tips}"
    return ""


def _docvqa_2026_baseline_category_tips(category: str) -> str:
    """DocVQA-2026 single-shot / baseline per-category tips.

    Mirror of :func:`_docvqa_2026_category_tips` for raw-VLM baselines.
    """
    tips = _DOCVQA_2026_BASELINE_CATEGORY_TIPS.get(category, "")
    if tips:
        return f"## CATEGORY-SPECIFIC TIPS ({category})\n{tips}"
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
    answer_formatting_rules: str = _DOCVQA_2026_ANSWER_FORMATTING_RULES
    category_tips_fn: TipsFn = field(
        default_factory=lambda: _docvqa_2026_category_tips
    )
    # Separate tips for the raw-VLM baseline — DocVQA-2026 tunes these
    # differently than the scaffold tips. For benchmarks with a single
    # doc category, both are ``_no_tips``.
    baseline_category_tips_fn: TipsFn = field(
        default_factory=lambda: _docvqa_2026_baseline_category_tips
    )
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
