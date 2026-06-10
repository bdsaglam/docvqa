"""Tests for multimodal embedding page search (vsearch).

Uses synthetic rendered-text pages so no dataset is needed. First run
downloads the 250M ColModernVBERT weights (~0.5 GB) — slow once, cached
after.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from docvqa.vsearch import (
    build_page_index,
    get_or_build_page_index,
    load_page_index,
)


def _text_page(lines: list[str], size=(640, 832)) -> Image.Image:
    """Render lines of text onto a white page."""
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=36)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    y = 40
    for line in lines:
        draw.text((30, y), line, fill="black", font=font)
        y += 60
    return img


@pytest.fixture(scope="module")
def pages() -> list[Image.Image]:
    return [
        _text_page(["QUARTERLY REVENUE REPORT", "Total revenue: $4.2M",
                    "Sales by region", "Q3 2024 financial results"]),
        _text_page(["WIRING DIAGRAM", "Electrical schematic",
                    "Circuit breaker panel", "Voltage: 240V"]),
        _text_page([]),  # blank page
    ]


@pytest.fixture(scope="module")
def index(pages):
    idx = build_page_index("test_doc", pages)
    assert idx is not None
    return idx


def test_text_query_ranks_relevant_page_first(index):
    records = index.search("quarterly revenue financial report", k=3)
    assert len(records) == 3
    assert records[0]["page"] == 0
    assert records[0]["score"] >= records[1]["score"] >= records[2]["score"]
    assert all(set(r) == {"page", "score"} for r in records)


def test_image_query_self_retrieval(index, pages):
    records = index.search(pages[1], k=3)
    assert records[0]["page"] == 1


def test_k_capped_at_num_pages(index):
    records = index.search("anything", k=10)
    assert len(records) == 3


def test_cache_roundtrip(pages, tmp_path: Path):
    idx = get_or_build_page_index("doc_a", pages, vsearch_dir=tmp_path)
    cache_dir = tmp_path / "doc_a"
    assert (cache_dir / "embeddings.pt").exists()
    meta = json.loads((cache_dir / "meta.json").read_text())
    assert meta["num_pages"] == 3

    loaded = load_page_index("doc_a", tmp_path, model_name=idx.model_name,
                             num_pages=3)
    assert loaded is not None
    assert loaded.search("wiring electrical diagram", k=1)[0]["page"] == 1


def test_cache_invalidated_on_model_mismatch(pages, tmp_path: Path):
    get_or_build_page_index("doc_b", pages, vsearch_dir=tmp_path)
    loaded = load_page_index("doc_b", tmp_path, model_name="other/model",
                             num_pages=3)
    assert loaded is None


def test_cache_invalidated_on_page_count_mismatch(pages, tmp_path: Path):
    idx = get_or_build_page_index("doc_c", pages, vsearch_dir=tmp_path)
    loaded = load_page_index("doc_c", tmp_path, model_name=idx.model_name,
                             num_pages=5)
    assert loaded is None


def test_document_vsearch_dir_derived_from_bm25_dir():
    from docvqa.data import Document

    doc = Document(doc_id="d", doc_category="c", images=[], questions=[],
                   bm25_dir=Path("data/docvqa-2026/val/bm25"))
    assert doc.vsearch_dir == Path("data/docvqa-2026/val/vsearch")

    doc_no_dir = Document(doc_id="d", doc_category="c", images=[], questions=[])
    assert doc_no_dir.vsearch_dir is None


def test_vsearch_solver_sandbox_code_compiles():
    from docvqa.solvers.rvlm_vsearch_solver import _build_sandbox_code

    code = _build_sandbox_code("/tmp/x", 3)
    compile(code, "<sandbox>", "exec")
    assert "def search(query, k=5):" in code
    assert "def batch_look(requests):" in code
    assert "def look(" not in code
