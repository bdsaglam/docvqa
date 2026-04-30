"""ANLS evaluation metrics for DocVQA 2026.

Adapted from the official eval_utils.py at
https://github.com/VLR-CVC/DocVQA2026
"""

import ast
import re
import string

import Levenshtein
from dateutil import parser as dateutil_parser


def get_anls(s1: str, s2: str) -> float:
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    dist = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (dist / max_len)


def is_string_correct(prediction: str, ground_truths: list[str], threshold: float = 0.80) -> bool:
    return any(get_anls(prediction, gt) >= threshold for gt in ground_truths)


def parse_magnitude_unit(text: str) -> tuple[float | None, str | None]:
    text = text.lower().strip()
    match = re.match(r"^(-?\d+(?:\.\d+)?)\s*(.*)$", text)
    if not match:
        return None, None
    try:
        val = float(match.group(1))
        return val, match.group(2).strip()
    except ValueError:
        return None, None


_UNIT_ALIASES = {
    "inches": "in", "inch": "in",
    "feet": "ft", "foot": "ft",
    "meters": "m", "meter": "m", "metre": "m", "metres": "m",
    "centimeters": "cm", "centimeter": "cm", "centimetre": "cm", "centimetres": "cm",
    "millimeters": "mm", "millimeter": "mm", "millimetre": "mm", "millimetres": "mm",
    "kilometers": "km", "kilometer": "km", "kilometre": "km", "kilometres": "km",
    "pounds": "lb", "pound": "lb", "lbs": "lb",
    "kilograms": "kg", "kilogram": "kg",
    "grams": "g", "gram": "g",
    "percent": "%", "pct": "%",
    "dollars": "usd", "dollar": "usd", "$": "usd",
}


def _normalize_unit(unit: str) -> str:
    u = unit.lower().strip()
    return _UNIT_ALIASES.get(u, u)


def _check_strict_match(pred_text: str, gt_text: str) -> bool:
    pred_val, pred_unit = parse_magnitude_unit(pred_text)
    gt_val, gt_unit = parse_magnitude_unit(gt_text)

    if pred_val is not None and gt_val is not None:
        if pred_val == gt_val:
            return _normalize_unit(pred_unit or "") == _normalize_unit(gt_unit or "")
        return False

    try:
        p_clean = pred_text.strip()
        g_clean = gt_text.strip()
        version_regex = r"^\d+\.\d+\.\d+$"
        if re.match(version_regex, p_clean) or re.match(version_regex, g_clean):
            return p_clean == g_clean
        if len(p_clean) >= 6 and len(g_clean) >= 6:
            pred_date = dateutil_parser.parse(p_clean, fuzzy=False).date()
            gt_date = dateutil_parser.parse(g_clean, fuzzy=False).date()
            return pred_date == gt_date
    except (ValueError, TypeError, OverflowError):
        pass

    return False


_PUNCT_TRANSLATOR = str.maketrans(string.punctuation, " " * len(string.punctuation))


def _clean_text(text: str) -> str:
    t = text.lower().translate(_PUNCT_TRANSLATOR)
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return " ".join(t.split())


def evaluate_prediction(raw_prediction: str, ground_truth: str) -> tuple[bool, str]:
    """Evaluate a single prediction against ground truth.

    Returns (is_correct, extracted_answer).
    """
    if not isinstance(raw_prediction, str):
        raw_prediction = str(raw_prediction)

    marker = "FINAL ANSWER:"
    if marker in raw_prediction:
        extracted_answer = raw_prediction.split(marker)[-1].strip()
    else:
        extracted_answer = raw_prediction.strip()

    gt_candidates: list[str] = []
    try:
        parsed_gt = ast.literal_eval(str(ground_truth))
        if isinstance(parsed_gt, list):
            gt_candidates = [str(x) for x in parsed_gt]
        else:
            gt_candidates = [str(ground_truth)]
    except (ValueError, SyntaxError):
        gt_candidates = [str(ground_truth)]

    # Strict matching (numbers/units/dates)
    gt_is_numeric = False
    for gt in gt_candidates:
        if _check_strict_match(extracted_answer, gt):
            return True, extracted_answer
        gt_val, _ = parse_magnitude_unit(gt)
        if gt_val is not None:
            gt_is_numeric = True

    if gt_is_numeric:
        return False, extracted_answer

    # Normalize "unknown" variants (ground truth has typos like "Unkown")
    _unknown_variants = {"unknown", "unkown", "uknown"}
    if _clean_text(extracted_answer) in _unknown_variants:
        if any(_clean_text(gt) in _unknown_variants for gt in gt_candidates):
            return True, extracted_answer

    # Relaxed text match (ANLS)
    clean_pred = _clean_text(extracted_answer)
    clean_gt_candidates = [_clean_text(gt) for gt in gt_candidates]
    is_correct = is_string_correct(clean_pred, clean_gt_candidates, threshold=0.9)

    return is_correct, extracted_answer
