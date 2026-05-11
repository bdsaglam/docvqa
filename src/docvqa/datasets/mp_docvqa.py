"""Loader for the MP-DocVQA benchmark.

The HuggingFace dataset `lmms-lab/MP-DocVQA` ships one row per
(document, question) pair. Each row holds:

    - ``questionId`` (str/int): unique question id
    - ``question`` (str)
    - ``doc_id`` (str): shared across rows that target the same document
    - ``page_ids`` (str): python-literal list of per-page slugs
      e.g. ``"['snbx0223_p11', 'snbx0223_p12', ...]"``
    - ``answers`` (str): python-literal list of GT answer strings.
      Empty list (``"[]"``) on the test split (answers are hidden).
    - ``answer_page_idx`` (str): index into ``page_ids`` of the page
      containing the answer (or empty on test).
    - ``data_split`` (str): "val" or "test"
    - ``image_1`` ... ``image_20`` (PIL.Image | None): up to 20 page
      images, in page order; trailing slots are ``None`` for shorter docs.

The loader groups all rows for a given ``doc_id`` into a single
:class:`docvqa.data.Document`, dedupes the page images via the first row
that carries them, and turns each row into a :class:`docvqa.data.Question`
with the GT answer joined into a single string when the dataset provides
multiple aliases.

Splits
------
Only ``val`` ships ground-truth answers; ``test`` answers are hidden
(empty list per row). For local scoring we always evaluate on ``val``.

Per-doc page cap
----------------
The HF schema caps documents at 20 pages — anything longer is already
truncated upstream. We do not impose an additional cap here.
"""

from __future__ import annotations

import ast
import logging
from collections import OrderedDict
from pathlib import Path

from datasets import load_dataset
from PIL import Image

from docvqa.data import Document, Question

logger = logging.getLogger(__name__)

HF_REPO_ID = "lmms-lab/MP-DocVQA"
DEFAULT_DATA_DIR = Path("data/mp-docvqa")
MAX_IMAGE_COLS = 20  # image_1 .. image_20 in the HF schema
DOC_CATEGORY = "mp_docvqa"  # MP-DocVQA has no native category


def _parse_listish(raw):
    """Parse a python-literal list. Tolerates already-parsed values and empties."""
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return raw
    try:
        v = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _coerce_image(value) -> Image.Image | None:
    """Normalize an HF image cell into a PIL.Image (or None).

    HF can serve images as :class:`PIL.Image.Image` instances or as
    ``{"bytes": ..., "path": ...}`` dicts depending on the decoding mode.
    """
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        if value.get("bytes"):
            from io import BytesIO

            return Image.open(BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    return None


def _doc_pages_from_row(row) -> list[Image.Image]:
    """Extract the ordered, non-empty page images from a single HF row."""
    pages: list[Image.Image] = []
    for i in range(1, MAX_IMAGE_COLS + 1):
        cell = row.get(f"image_{i}")
        img = _coerce_image(cell)
        if img is None:
            break  # the schema is dense up to n_pages then trailing Nones
        pages.append(img)
    return pages


def _split_doc_ids_file(split: str) -> list[str] | None:
    """Read ``data/mp-docvqa/<split>/sample_200q_doc_ids.txt`` if present."""
    base = split.split("[")[0]
    sample_file = DEFAULT_DATA_DIR / base / "sample_200q_doc_ids.txt"
    if not sample_file.exists():
        return None
    return [
        line.strip()
        for line in sample_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _ocr_dir_for(split: str) -> Path:
    base = split.split("[")[0]
    return DEFAULT_DATA_DIR / base / "ocr"


def _load_ocr_texts(doc_id: str, num_pages: int, ocr_dir: Path) -> list[str] | None:
    doc_dir = ocr_dir / doc_id
    if not doc_dir.exists():
        return None
    out = []
    for i in range(num_pages):
        p = doc_dir / f"page_{i}.md"
        out.append(p.read_text() if p.exists() else "")
    return out


def load_mp_docvqa_documents(
    split: str = "val",
    num_samples: int | None = None,
    doc_ids: list[str] | None = None,
    ocr_dir: Path | str | None = None,
) -> list[Document]:
    """Load MP-DocVQA and return a list of :class:`Document` objects.

    Args:
        split: HF split — ``"val"`` (with answers) or ``"test"`` (hidden).
            If ``data/mp-docvqa/<split>/sample_200q_doc_ids.txt`` exists
            and ``doc_ids`` is not provided, the file is used to subset.
        num_samples: Cap on the number of documents returned (after the
            sample file / ``doc_ids`` filter). Ignored if ``doc_ids`` is
            explicit.
        doc_ids: Explicit allowlist of ``doc_id`` strings.
        ocr_dir: Optional override for cached OCR markdown
            (defaults to ``data/mp-docvqa/<split>/ocr``).

    Returns:
        One :class:`Document` per unique ``doc_id``, with all questions
        targeting that document grouped together. ``doc_category`` is the
        constant ``"mp_docvqa"`` (the dataset has no native categories).
    """
    base_split = split.split("[")[0]
    if doc_ids is None:
        doc_ids = _split_doc_ids_file(split)

    ds = load_dataset(HF_REPO_ID, split=base_split)

    allow: set[str] | None = set(doc_ids) if doc_ids is not None else None

    # Group rows by doc_id while keeping insertion order stable.
    by_doc: "OrderedDict[str, list]" = OrderedDict()
    for row in ds:
        doc_id = row["doc_id"]
        if allow is not None and doc_id not in allow:
            continue
        by_doc.setdefault(doc_id, []).append(row)

    if doc_ids is not None:
        missing = allow - set(by_doc.keys())
        if missing:
            logger.warning(
                "mp_docvqa: %d doc_ids not found in split=%s: %s",
                len(missing),
                base_split,
                sorted(missing)[:5],
            )

    if doc_ids is None and num_samples is not None:
        # Truncate by document, not by row, to keep questions/doc intact.
        keep = list(by_doc.keys())[:num_samples]
        by_doc = OrderedDict((k, by_doc[k]) for k in keep)

    ocr_dir = Path(ocr_dir) if ocr_dir is not None else _ocr_dir_for(split)

    documents: list[Document] = []
    for doc_id, rows in by_doc.items():
        # Use the first row whose images decode successfully — page sets are
        # identical across rows for the same doc, but be defensive.
        images: list[Image.Image] = []
        for row in rows:
            pages = _doc_pages_from_row(row)
            if pages:
                images = pages
                break
        if not images:
            logger.warning("mp_docvqa: no images decoded for doc_id=%s; skipping", doc_id)
            continue

        questions: list[Question] = []
        for row in rows:
            answers = _parse_listish(row.get("answers"))
            # MP-DocVQA ships multiple answer aliases — join with " | " so the
            # downstream metric can still hit the canonical form. ANLS-style
            # scoring typically reduces over the alias list, but our dataclass
            # only has a single string slot.
            answer_str: str | None
            if answers:
                answer_str = answers[0] if len(answers) == 1 else " | ".join(map(str, answers))
            else:
                answer_str = None  # test split
            questions.append(
                Question(
                    question_id=str(row["questionId"]),
                    question=row["question"],
                    answer=answer_str,
                )
            )

        documents.append(
            Document(
                doc_id=doc_id,
                doc_category=DOC_CATEGORY,
                images=images,
                questions=questions,
                page_texts=_load_ocr_texts(doc_id, len(images), ocr_dir),
            )
        )

    return documents


def get_dataset_stats(split: str = "val") -> dict:
    """Quick stats over an MP-DocVQA split (no image decoding needed for counts)."""
    from collections import defaultdict

    ds = load_dataset(HF_REPO_ID, split=split)
    doc_q: dict[str, int] = defaultdict(int)
    doc_pages: dict[str, int] = {}
    for row in ds:
        d = row["doc_id"]
        doc_q[d] += 1
        if d not in doc_pages:
            doc_pages[d] = sum(
                1 for i in range(1, MAX_IMAGE_COLS + 1) if row[f"image_{i}"] is not None
            )
    qpd = list(doc_q.values())
    pgs = list(doc_pages.values())
    return {
        "split": split,
        "n_questions": len(ds),
        "n_docs": len(doc_q),
        "avg_questions_per_doc": sum(qpd) / max(1, len(qpd)),
        "avg_pages_per_doc": sum(pgs) / max(1, len(pgs)),
        "max_pages_per_doc": max(pgs, default=0),
        "min_pages_per_doc": min(pgs, default=0),
    }


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(get_dataset_stats("val"))
    docs = load_mp_docvqa_documents(split="val", num_samples=2)
    for d in docs:
        print(d.doc_id, "pages=", len(d.images), "qs=", len(d.questions))
