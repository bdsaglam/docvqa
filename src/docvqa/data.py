"""Dataset loading for DocVQA 2026."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000


@dataclass
class Question:
    question_id: str
    question: str
    answer: str | None = None  # None for test set


@dataclass
class Document:
    doc_id: str
    doc_category: str
    images: list[Image.Image]
    questions: list[Question]
    page_texts: list[str] | None = None  # OCR text per page, if available
    bm25_dir: Path | None = None  # where this doc's BM25 index lives; None → DEFAULT_BM25_DIR

    @property
    def question_ids(self) -> list[str]:
        return [q.question_id for q in self.questions]


DATA_DIR = Path("data")


def _ocr_dir_for_split(split: str) -> Path:
    """Return OCR directory for a DocVQA-2026 split.

    Layout: ``data/docvqa-2026/{split}/ocr/{doc_id}/page_*.md``. New
    datasets follow ``data/{dataset-slug}/{split}/...`` and supply
    their own ocr path.
    """
    base_split = split.split("[")[0]  # strip slice like "val[:5]"
    return DATA_DIR / "docvqa-2026" / base_split / "ocr"


def _bm25_dir_for_split(split: str) -> Path:
    """Return BM25 index directory for a DocVQA-2026 split."""
    base_split = split.split("[")[0]
    return DATA_DIR / "docvqa-2026" / base_split / "bm25"


def _load_ocr_texts(doc_id: str, num_pages: int, ocr_dir: Path) -> list[str] | None:
    """Load cached OCR text for a document, if available."""
    doc_dir = ocr_dir / doc_id
    if not doc_dir.exists():
        return None
    page_texts = []
    for i in range(num_pages):
        md_path = doc_dir / f"page_{i}.md"
        page_texts.append(md_path.read_text() if md_path.exists() else "")
    return page_texts


def load_documents(
    dataset_name: str,
    split: str,
    num_samples: int | None = None,
    doc_ids: list[str] | None = None,
    ocr_dir: Path | str | None = None,
) -> list[Document]:
    """Load a DocVQA-style dataset and convert to ``Document`` objects.

    Dispatch by dataset name. The default path handles ``VLR-CVC/DocVQA-2026``
    directly; ``lmms-lab/MP-DocVQA`` and ``yubo2333/MMLongBench-Doc`` are
    routed to dedicated loaders in :mod:`docvqa.datasets`.
    """
    if dataset_name == "lmms-lab/MP-DocVQA":
        from docvqa.datasets.mp_docvqa import load_mp_docvqa_documents

        return load_mp_docvqa_documents(
            split=split,
            num_samples=num_samples,
            doc_ids=doc_ids,
            ocr_dir=ocr_dir,
        )
    if dataset_name == "yubo2333/MMLongBench-Doc":
        from docvqa.datasets.mmlongbench_doc import load_mmlongbench_doc_documents

        return load_mmlongbench_doc_documents(
            split=split,
            num_samples=num_samples,
            doc_ids=doc_ids,
            ocr_dir=ocr_dir,
        )

    if doc_ids is None and num_samples is not None:
        split = f"{split}[:{num_samples}]"
    ds = load_dataset(dataset_name, split=split)
    if ocr_dir is None:
        ocr_dir = _ocr_dir_for_split(split)
    ocr_dir = Path(ocr_dir)
    documents = []
    for sample in ds:
        if doc_ids is not None and sample["doc_id"] not in doc_ids:
            continue
        questions = []
        q_ids = sample["questions"]["question_id"]
        q_texts = sample["questions"]["question"]

        # Answers may not exist for test split
        a_ids = sample.get("answers", {}).get("question_id", [])
        a_texts = sample.get("answers", {}).get("answer", [])
        answer_map = dict(zip(a_ids, a_texts)) if a_ids else {}

        for qid, qtext in zip(q_ids, q_texts):
            questions.append(Question(
                question_id=qid,
                question=qtext,
                answer=answer_map.get(qid),
            ))

        images = sample["document"]  # list of PIL Images
        documents.append(Document(
            doc_id=sample["doc_id"],
            doc_category=sample["doc_category"],
            images=images,
            questions=questions,
            page_texts=_load_ocr_texts(sample["doc_id"], len(images), ocr_dir),
            bm25_dir=_bm25_dir_for_split(split),
        ))
    return documents
