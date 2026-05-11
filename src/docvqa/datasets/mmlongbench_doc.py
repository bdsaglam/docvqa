"""Loader for the MMLongBench-Doc benchmark.

The HuggingFace dataset `yubo2333/MMLongBench-Doc` ships:
    - `data/train-00000-of-00001.parquet` — 1091 question rows
      (doc_id, doc_type, question, answer, evidence_pages,
       evidence_sources, answer_format).
    - `documents/<doc_id>` — 135 source PDFs (avg ~47.5 pages).

This module rasterises every required PDF once (cached under
`data/mmlongbench-doc/<split>/pages/<doc_id>/page_<i>.png`) and assembles
`Document` objects with the same shape used by `src/docvqa/data.py`. The
question-level metadata (answer_format, evidence_pages, evidence_sources,
doc_type) is attached as extra attributes on each `Question` via
`object.__setattr__` so it survives without touching the shared dataclass.

Notes
-----
- We use ``pypdfium2`` (no system poppler dep) at 150 DPI by default. That
  matches the rendering knob mp-docvqa-style benchmarks use and keeps each
  page in the 1–2 MB range for typical layouts.
- PDFs longer than ``max_pages`` are truncated. Render order is preserved
  (pages 0..N-1) so ``evidence_pages`` indices remain valid as long as they
  fall inside the cap. We surface the truncation in a per-doc log line.
- The HF dataset uses ``Str / Int / Float / List / None`` answer-format
  labels; ``None`` is MMLongBench-Doc's "Not answerable" class and the GT
  string is literally the word ``"Not answerable"``.
"""

from __future__ import annotations

import ast
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

from docvqa.data import Document, Question

logger = logging.getLogger(__name__)

HF_REPO_ID = "yubo2333/MMLongBench-Doc"
DEFAULT_DATA_DIR = Path("data/mmlongbench-doc")
DEFAULT_DPI = 150
DEFAULT_MAX_PAGES = 80  # cap to keep memory/eval cost bounded; ~95% of docs ≤80
ANSWER_FORMATS = ("Str", "Int", "Float", "List", "None")


@dataclass
class MMLBQuestionMeta:
    """MMLongBench-Doc-specific question metadata attached to ``Question``."""

    answer_format: str  # one of ANSWER_FORMATS
    evidence_pages: list[int]
    evidence_sources: list[str]
    doc_type: str


def _parse_listish(raw: str | None) -> list:
    """Parse a python-literal list like ``"[5, 6]"`` or ``"['Chart']"``.

    Falls back to ``[]`` on empty / malformed input. The HF dataset stores
    these columns as strings.
    """
    if raw is None or raw == "":
        return []
    try:
        v = ast.literal_eval(raw)
        return list(v) if isinstance(v, (list, tuple)) else [v]
    except (ValueError, SyntaxError):
        return []


def _split_doc_ids(split: str) -> tuple[str, list[str] | None]:
    """The HF dataset has only one ``train`` split (1091 rows, 135 docs).

    To keep ``split="val"`` / ``split="test"`` semantics, we treat the whole
    HF split as the universe and let callers narrow via ``doc_ids`` or a
    sample-id file (see ``data/mmlongbench-doc/val/sample_200q_doc_ids.txt``).
    """
    base = split.split("[")[0]
    sample_file = DEFAULT_DATA_DIR / base / "sample_200q_doc_ids.txt"
    if sample_file.exists():
        doc_ids = [
            line.strip()
            for line in sample_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        return base, doc_ids
    return base, None


def _ensure_pdf(doc_id: str) -> Path:
    """Download a PDF from the HF dataset repo into the HF cache."""
    return Path(
        hf_hub_download(
            HF_REPO_ID,
            f"documents/{doc_id}",
            repo_type="dataset",
        )
    )


def _render_pdf(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = DEFAULT_DPI,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> tuple[list[Path], int]:
    """Render a PDF to PNGs under ``out_dir``. Idempotent / cached.

    Returns ``(page_paths, total_pages_in_pdf)``. ``page_paths`` is capped
    at ``max_pages``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        n_total = len(pdf)
        n = min(n_total, max_pages)
        scale = dpi / 72.0  # PDF base = 72 DPI
        paths: list[Path] = []
        for i in range(n):
            png = out_dir / f"page_{i}.png"
            if not png.exists():
                page = pdf[i]
                try:
                    img = page.render(scale=scale).to_pil()
                    img.save(png, format="PNG", optimize=True)
                finally:
                    page.close()
            paths.append(png)
        return paths, n_total
    finally:
        pdf.close()


def _load_ocr_texts(doc_id: str, num_pages: int, ocr_dir: Path) -> list[str] | None:
    doc_dir = ocr_dir / doc_id
    if not doc_dir.exists():
        return None
    texts = []
    for i in range(num_pages):
        p = doc_dir / f"page_{i}.md"
        texts.append(p.read_text() if p.exists() else "")
    return texts


def load_mmlongbench_doc_documents(
    split: str = "val",
    num_samples: int | None = None,
    doc_ids: list[str] | None = None,
    ocr_dir: Path | str | None = None,
    *,
    dpi: int = DEFAULT_DPI,
    max_pages: int = DEFAULT_MAX_PAGES,
    pages_dir: Path | str | None = None,
) -> list[Document]:
    """Load MMLongBench-Doc and return a list of ``Document`` objects.

    Args:
        split: Logical split name; currently maps to the single HF "train"
            split, optionally restricted by
            ``data/mmlongbench-doc/<split>/sample_200q_doc_ids.txt`` if it
            exists. ``num_samples`` truncates the resulting doc list.
        num_samples: Cap on the number of docs returned. Ignored when
            ``doc_ids`` is provided.
        doc_ids: Restrict to these doc_ids (must match HF ``doc_id``,
            including the ``.pdf`` suffix).
        ocr_dir: Optional override for cached OCR markdown
            (defaults to ``data/mmlongbench-doc/<split>/ocr``).
        dpi: PDF rasterisation resolution. Lower to save disk; 150 is the
            default and matches what the official authors used.
        max_pages: Cap on rendered pages per PDF. Pages beyond this cap
            are silently dropped (logged at WARNING).
        pages_dir: Override for the rendered-page cache directory.

    Returns:
        List of ``Document``. Each ``Question`` carries an extra ``mmlb``
        attribute (``MMLBQuestionMeta``) with answer_format / evidence info.
    """
    base_split, sample_doc_ids = _split_doc_ids(split)
    if doc_ids is None:
        doc_ids = sample_doc_ids

    ds = load_dataset(HF_REPO_ID, split="train")
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        by_doc[row["doc_id"]].append(row)

    universe = list(by_doc.keys())
    if doc_ids is not None:
        universe = [d for d in doc_ids if d in by_doc]
        missing = set(doc_ids) - set(universe)
        if missing:
            logger.warning("mmlongbench: %d doc_ids not found: %s", len(missing), sorted(missing)[:5])
    if num_samples is not None:
        universe = universe[:num_samples]

    pages_dir = Path(pages_dir or DEFAULT_DATA_DIR / base_split / "pages")
    ocr_dir = Path(ocr_dir or DEFAULT_DATA_DIR / base_split / "ocr")

    documents: list[Document] = []
    for doc_id in universe:
        rows = by_doc[doc_id]
        doc_type = rows[0]["doc_type"]
        try:
            pdf_path = _ensure_pdf(doc_id)
        except Exception as exc:
            logger.error("mmlongbench: failed to fetch %s: %s", doc_id, exc)
            continue

        png_paths, n_total = _render_pdf(
            pdf_path,
            pages_dir / doc_id,
            dpi=dpi,
            max_pages=max_pages,
        )
        if n_total > max_pages:
            logger.warning(
                "mmlongbench: %s truncated %d -> %d pages (max_pages=%d)",
                doc_id, n_total, len(png_paths), max_pages,
            )

        images = [Image.open(p) for p in png_paths]
        questions: list[Question] = []
        for row in rows:
            q = Question(
                question_id=f"{doc_id}::{row['question'][:60]}",
                question=row["question"],
                answer=row["answer"],
            )
            object.__setattr__(
                q,
                "mmlb",
                MMLBQuestionMeta(
                    answer_format=row["answer_format"],
                    evidence_pages=[int(p) for p in _parse_listish(row.get("evidence_pages"))],
                    evidence_sources=[str(s) for s in _parse_listish(row.get("evidence_sources"))],
                    doc_type=doc_type,
                ),
            )
            questions.append(q)

        documents.append(
            Document(
                doc_id=doc_id,
                doc_category=doc_type,
                images=images,
                questions=questions,
                page_texts=_load_ocr_texts(doc_id, len(images), ocr_dir),
            )
        )
    return documents


def get_dataset_stats() -> dict:
    """Quick stats over the full HF dataset (no PDF rendering)."""
    ds = load_dataset(HF_REPO_ID, split="train")
    fmt_counts: dict[str, int] = defaultdict(int)
    doc_q_counts: dict[str, int] = defaultdict(int)
    for row in ds:
        fmt_counts[row["answer_format"]] += 1
        doc_q_counts[row["doc_id"]] += 1
    return {
        "n_questions": len(ds),
        "n_docs": len(doc_q_counts),
        "format_counts": dict(fmt_counts),
        "avg_questions_per_doc": len(ds) / max(1, len(doc_q_counts)),
    }


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(get_dataset_stats())
    docs = load_mmlongbench_doc_documents(split="val", num_samples=1)
    for d in docs:
        print(d.doc_id, len(d.images), len(d.questions))
