"""Tests for BM25 search quality on real eval documents.

Loads OCR text from data/val/ocr/, builds BM25 indexes, and verifies that
realistic agent queries return relevant pages in the top results.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docvqa.search import build_index, make_search_tool, _chunk_page

OCR_DIR = Path("data/val/ocr")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_page_texts(doc_id: str) -> list[str]:
    """Load all OCR page texts for a document."""
    doc_dir = OCR_DIR / doc_id
    pages = sorted(doc_dir.glob("page_*.md"), key=lambda p: int(p.stem.split("_")[1]))
    return [p.read_text() for p in pages]


def build_search(doc_id: str):
    """Build a search tool for a document. Returns (search_fn, page_texts)."""
    page_texts = load_page_texts(doc_id)
    index = build_index(doc_id, page_texts)
    assert index is not None, f"Failed to build index for {doc_id}"
    search = make_search_tool(index)
    return search, page_texts


# ---------------------------------------------------------------------------
# Chunking unit tests
# ---------------------------------------------------------------------------

class TestChunking:
    def test_empty_text_returns_no_chunks(self):
        assert _chunk_page(0, "") == []
        assert _chunk_page(0, "   \n\n  ") == []

    def test_short_text_single_chunk(self):
        chunks = _chunk_page(3, "Hello world")
        assert len(chunks) == 1
        assert chunks[0]["page"] == 3
        assert chunks[0]["text"] == "Hello world"

    def test_long_text_splits_on_paragraphs(self):
        para_a = "A" * 300
        para_b = "B" * 300
        text = f"{para_a}\n\n{para_b}"
        chunks = _chunk_page(0, text, max_chunk_chars=500)
        assert len(chunks) == 2
        assert "A" in chunks[0]["text"]
        assert "B" in chunks[1]["text"]

    def test_page_number_preserved(self):
        chunks = _chunk_page(42, "Some text\n\nMore text")
        assert all(c["page"] == 42 for c in chunks)


# ---------------------------------------------------------------------------
# Index construction tests
# ---------------------------------------------------------------------------

class TestIndexConstruction:
    def test_build_index_returns_none_for_empty(self):
        assert build_index("empty", []) is None
        assert build_index("empty", ["", "  ", "\n\n"]) is None

    @pytest.mark.skipif(not (OCR_DIR / "slide_1").exists(), reason="OCR data not available")
    def test_build_index_for_real_doc(self):
        page_texts = load_page_texts("slide_1")
        index = build_index("slide_1", page_texts)
        assert index is not None
        assert hasattr(index, "_chunk_meta")
        assert len(index._chunk_meta) > 0


# ---------------------------------------------------------------------------
# Search quality tests — business_report_1 (181 pages, NVIDIA annual report)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (OCR_DIR / "business_report_1").exists(), reason="OCR data not available")
class TestSearchBusinessReport1:
    """NVIDIA 2025 Annual Review — 181 pages.

    Tests that BM25 can locate specific content in a very large document.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.search, self.page_texts = build_search("business_report_1")

    def test_tsr_shareholder_return(self):
        """Q: 'NVIDIA TSR value exceed Nasdaq-100 Index' — answer on page 16/18."""
        results = self.search("Total Shareholder Return TSR Nasdaq-100")
        pages = {r["page"] for r in results}
        # TSR comparison stock performance graph is on pages 16-18
        assert pages & {16, 18}, f"Expected page 16 or 18, got {pages}"

    def test_agent_blueprints(self):
        """Q: about 'Agent Blueprints' bullet point — on early overview pages."""
        results = self.search("Agent Blueprints")
        pages = {r["page"] for r in results}
        assert pages & {10, 11, 12, 95, 97, 98}, f"Expected Agent Blueprints pages, got {pages}"

    def test_competitor_information(self):
        """Agent would search for competitors like AMD, Intel."""
        results = self.search("competitors AMD Intel Huawei")
        assert len(results) > 0
        # Should find content in the competitive landscape section
        assert any("AMD" in r["text"] or "Intel" in r["text"] for r in results)

    def test_tsmc_manufacturing(self):
        """Agent would search for semiconductor manufacturing partners."""
        results = self.search("TSMC semiconductor manufacturing")
        assert len(results) > 0
        top_text = " ".join(r["text"] for r in results[:3])
        assert "TSMC" in top_text or "semiconductor" in top_text.lower()

    def test_definitions_section(self):
        """The definitions/glossary is on page 16."""
        results = self.search("DEFINITIONS 2007 Plan Bylaws")
        pages = {r["page"] for r in results}
        assert 16 in pages, f"Expected page 16 (definitions), got {pages}"

    def test_fiscal_2025_revenue(self):
        """Agent would search for financial figures."""
        results = self.search("fiscal 2025 revenue operating income")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Search quality tests — science_paper_1 (44 pages, Perception Encoder)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (OCR_DIR / "science_paper_1").exists(), reason="OCR data not available")
class TestSearchSciencePaper1:
    """Perception Encoder paper — 44 pages.

    Tests that BM25 correctly locates sections, methods, and results.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.search, self.page_texts = build_search("science_paper_1")

    def test_ablation_study(self):
        """Q: 'How much data to ablate the Perception Encoder?' — ablation in pages 11-13."""
        results = self.search("ablation Perception Encoder data")
        pages = {r["page"] for r in results}
        assert pages & {11, 12, 13}, f"Expected ablation pages 11-13, got {pages}"

    def test_related_work_section(self):
        """Q: 'page before Related Work, last citation number' — Related Work on page 19."""
        results = self.search("Related Work")
        pages = {r["page"] for r in results}
        assert 19 in pages, f"Expected page 19 (Related Work), got {pages}"

    def test_docvqa_performance(self):
        """Paper reports 94.6 DocVQA score."""
        results = self.search("DocVQA InfographicVQA performance score")
        assert len(results) > 0
        # Abstract on page 0 mentions these scores
        pages = {r["page"] for r in results}
        assert 0 in pages, f"Expected page 0 (abstract), got {pages}"

    def test_coco_detection(self):
        """Paper claims COCO state-of-the-art 66.0 box mAP."""
        results = self.search("COCO detection box mAP state-of-the-art")
        assert len(results) > 0

    def test_qwen_model_family(self):
        """Q: 'What family of models can make comparisons with native resolution?' — answer is Qwen."""
        results = self.search("native resolution comparison model", k=10)
        assert len(results) > 0

    def test_spatial_alignment(self):
        """Q: 'spatial alignment, similarity type, layer 29'."""
        results = self.search("spatial alignment similarity layer")
        assert len(results) > 0

    def test_video_data_engine(self):
        """Paper discusses video data engine for training."""
        results = self.search("video data engine synthetic annotation")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Search quality tests — business_report_3 (89 pages, Samsung Securities)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (OCR_DIR / "business_report_3").exists(), reason="OCR data not available")
class TestSearchBusinessReport3:
    """Samsung Securities 2024 Sustainability Report — 89 pages."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.search, self.page_texts = build_search("business_report_3")

    def test_unionization_rates(self):
        """Q: 'percentage difference between 2024 and 2023 unionization rates'."""
        results = self.search("unionization rate 2023 2024")
        pages = {r["page"] for r in results}
        assert 48 in pages, f"Expected page 48 (unionization data), got {pages}"

    def test_total_assets(self):
        """Agent would search for financial highlights."""
        results = self.search("total assets net income")
        assert len(results) > 0

    def test_esg_strategy(self):
        """ESG framework content."""
        results = self.search("ESG strategy environment social governance")
        assert len(results) > 0

    def test_employee_benefits(self):
        """Q: about pictogram — search for employee benefit categories."""
        results = self.search("employee benefits child education leisure health")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Search quality tests — slide_1 (36 pages, Indian Ocean Tuna Tagging)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (OCR_DIR / "slide_1").exists(), reason="OCR data not available")
class TestSearchSlide1:
    """Indian Ocean Tuna Tagging Programme — 36 slides."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.search, self.page_texts = build_search("slide_1")

    def test_total_releases(self):
        """Q: 'What is the correct number of total tuna releases?' — 201425 on page 12."""
        results = self.search("total releases tuna")
        pages = {r["page"] for r in results}
        assert 12 in pages, f"Expected page 12 (release totals table), got {pages}"

    def test_total_releases_by_number(self):
        """Search by the actual number."""
        results = self.search("201425")
        pages = {r["page"] for r in results}
        assert 12 in pages, f"Expected page 12, got {pages}"

    def test_skj_skipjack_percentage(self):
        """Q: 'How much did SKJ percentage increase between Releases and Recoveries?'."""
        results = self.search("SKJ skipjack percentage recoveries releases")
        assert len(results) > 0
        pages = {r["page"] for r in results}
        # SKJ data appears on pages 12, 15-26
        assert pages & set(range(12, 27)), f"Expected SKJ pages, got {pages}"

    def test_species_breakdown(self):
        """Agent would search for species composition."""
        results = self.search("species YFT BET SKJ composition")
        assert len(results) > 0

    def test_rttp_programme(self):
        """RTTP-IO is the main tagging programme."""
        results = self.search("RTTP-IO Regional Tuna Tagging Programme")
        assert len(results) > 0

    def test_double_tagging(self):
        """Double tagging and tag shedding data on page 12."""
        results = self.search("double tagging tag shedding")
        pages = {r["page"] for r in results}
        assert 12 in pages, f"Expected page 12 (double tagging), got {pages}"


# ---------------------------------------------------------------------------
# Search quality tests — science_paper_3 (30 pages)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (OCR_DIR / "science_paper_3").exists(), reason="OCR data not available")
class TestSearchSciencePaper3:
    """30-page science paper — tests on a medium-length academic doc."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.search, self.page_texts = build_search("science_paper_3")

    def test_abstract_found(self):
        """Abstract should be on page 0."""
        results = self.search("abstract introduction")
        pages = {r["page"] for r in results}
        assert 0 in pages, f"Expected page 0 (abstract), got {pages}"

    def test_results_not_empty(self):
        """Any query should return something for a 30-page doc."""
        results = self.search("method results experiment")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Search robustness tests
# ---------------------------------------------------------------------------

class TestSearchRobustness:
    """Tests that search handles edge cases gracefully."""

    @pytest.mark.skipif(not (OCR_DIR / "slide_1").exists(), reason="OCR data not available")
    def test_empty_query(self):
        search, _ = build_search("slide_1")
        results = search("")
        # Should not crash, may return empty or all
        assert isinstance(results, list)

    @pytest.mark.skipif(not (OCR_DIR / "slide_1").exists(), reason="OCR data not available")
    def test_nonsense_query_returns_empty_or_low_score(self):
        search, _ = build_search("slide_1")
        results = search("xyzzy frobnicator quantum entanglement")
        # Either empty or all scores should be very low
        if results:
            assert all(r["score"] < 5 for r in results), "Nonsense query got high scores"

    @pytest.mark.skipif(not (OCR_DIR / "slide_1").exists(), reason="OCR data not available")
    def test_k_parameter_limits_results(self):
        search, _ = build_search("slide_1")
        results_3 = search("tuna", k=3)
        results_10 = search("tuna", k=10)
        assert len(results_3) <= 3
        assert len(results_10) <= 10

    @pytest.mark.skipif(not (OCR_DIR / "business_report_1").exists(), reason="OCR data not available")
    def test_large_k_on_large_doc(self):
        search, _ = build_search("business_report_1")
        results = search("revenue", k=50)
        assert len(results) <= 50
        # Large doc should have many revenue hits
        assert len(results) > 5

    @pytest.mark.skipif(not (OCR_DIR / "slide_1").exists(), reason="OCR data not available")
    def test_results_have_required_fields(self):
        search, _ = build_search("slide_1")
        results = search("tuna tagging")
        assert len(results) > 0
        for r in results:
            assert "page" in r
            assert "score" in r
            assert "text" in r
            assert isinstance(r["page"], int)
            assert isinstance(r["score"], float)
            assert isinstance(r["text"], str)
            assert r["score"] > 0

    @pytest.mark.skipif(not (OCR_DIR / "slide_1").exists(), reason="OCR data not available")
    def test_results_sorted_by_score_descending(self):
        search, _ = build_search("slide_1")
        results = search("releases recoveries species")
        if len(results) > 1:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True), "Results not sorted by score"
