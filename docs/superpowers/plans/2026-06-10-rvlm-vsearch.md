# `rvlm_vsearch` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new solver `rvlm_vsearch` = rvlm (`batch_look`) + an OCR-free multimodal page-retrieval tool `search(query, k)` where the query is text or a PIL image, backed by ColModernVBERT late-interaction embeddings with per-doc on-disk caching.

**Architecture:** New module `src/docvqa/vsearch.py` mirrors `src/docvqa/search.py`'s get-or-build/cache API but stores per-page multi-vector embeddings (float16 tensors) and scores queries via MaxSim. New solver `src/docvqa/solvers/rvlm_vsearch_solver.py` is a clone of `rvlm_solver.py` (the OCR-free base; shared helpers live there) with the `search` tool added the same way `rvlm_ocr_ablation_solver.py` adds BM25. Spec: `docs/superpowers/specs/2026-06-10-rvlm-vsearch-design.md`.

**Tech Stack:** `colpali-engine` (MIT, ≥0.3.17) with `ColModernVBert`/`ColModernVBertProcessor`; model `ModernVBERT/colmodernvbert` (250M, MIT); torch (pulled in by colpali-engine); pytest; Hydra config.

**Conventions you must follow:**
- Python ≥3.13, `from __future__ import annotations`, `uv run` for everything.
- Logging/observability: wrap tool calls in `logfire.span(...)` like existing solvers.
- Tools prefixed `_` are hidden from the agent prompt; clean names are sandbox wrappers (see `rvlm_ocr_ablation_solver.py:201-239`).

---

### Task 1: Add colpali-engine dependency

**Files:**
- Modify: `pyproject.toml` (via `uv add`)
- Modify: `uv.lock` (generated)

- [ ] **Step 1: Add the dependency**

```bash
uv add colpali-engine
```

Expected: resolves and installs (pulls `torch`, `transformers` — multi-GB first install; this is known and accepted per spec). If resolution conflicts with `gepa`/`dspy` pins arise, report them rather than forcing.

- [ ] **Step 2: Verify the model classes exist in the installed release**

```bash
uv run python -c "from colpali_engine.models import ColModernVBert, ColModernVBertProcessor; import colpali_engine; print(colpali_engine.__version__ if hasattr(colpali_engine,'__version__') else 'ok')"
```

Expected: prints without ImportError. (Confirmed present on colpali main branch; the model card's "not yet merged" note is stale.)

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add colpali-engine for multimodal page embeddings"
```

> ⚠ `uv.lock` currently has unrelated uncommitted changes in the working tree. Check `git diff uv.lock` BEFORE Step 1; if dirty, ask the user whether to commit/stash those first so this commit stays clean.

---

### Task 2: `vsearch.py` — embedding index module (TDD)

**Files:**
- Create: `src/docvqa/vsearch.py`
- Test: `tests/test_vsearch.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vsearch.py`:

```python
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
```

> Fragility note: `test_text_query_ranks_relevant_page_first` and the
> semantic assertion in `test_cache_roundtrip` depend on real model
> behavior on rendered text — squarely the model's training domain, so
> expected to pass. If one proves flaky, weaken to "relevant text page
> outranks the blank page" rather than deleting the assertion. Do NOT
> mock the model; retrieval quality is the thing under test.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_vsearch.py -x -q 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'docvqa.vsearch'` (collection error counts as failing).

- [ ] **Step 3: Implement `src/docvqa/vsearch.py`**

```python
"""Multimodal embedding search over document page images.

OCR-free visual retrieval: pages are embedded with a late-interaction
visual retriever (ColModernVBERT by default) and queried with text or
an image via MaxSim scoring. Sibling of :mod:`docvqa.search` (BM25 over
OCR text), mirroring its get-or-build/cache API.

The embedder is a lazy process-wide singleton behind a lock: the eval
runner solves documents concurrently, and we want one model instance
and serialized forward passes (large page batches would otherwise
contend for memory).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ModernVBERT/colmodernvbert"
# Fallback for ad-hoc use; solvers pass Document.vsearch_dir.
DEFAULT_VSEARCH_DIR = Path("data/docvqa-2026/val/vsearch")

_BATCH_SIZE = 4  # pages per forward pass when building an index

_EMBEDDER_LOCK = threading.Lock()  # guards singleton AND forward passes
_EMBEDDER: tuple | None = None  # (model_name, device, model, processor)


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_embedder(model_name: str, device: str | None):
    """Load (once) and return (model, processor, device). Caller must hold _EMBEDDER_LOCK."""
    global _EMBEDDER
    resolved = _resolve_device(device)
    if _EMBEDDER is not None and _EMBEDDER[:2] == (model_name, resolved):
        return _EMBEDDER[2], _EMBEDDER[3], resolved

    import torch
    from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

    logger.info("Loading vsearch embedder %s on %s", model_name, resolved)
    processor = ColModernVBertProcessor.from_pretrained(model_name)
    model = (
        ColModernVBert.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True
        )
        .to(resolved)
        .eval()
    )
    _EMBEDDER = (model_name, resolved, model, processor)
    return model, processor, resolved


def _embed(inputs_list, process_fn, model, device) -> list:
    """Embed a list of inputs in batches. Returns per-item 2D cpu float16 tensors [seq, dim].

    Caller must hold _EMBEDDER_LOCK.
    """
    import torch

    out: list = []
    for i in range(0, len(inputs_list), _BATCH_SIZE):
        batch = process_fn(inputs_list[i : i + _BATCH_SIZE])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            embs = model(**batch)  # [b, seq, dim]
        out.extend(e.cpu().to(torch.float16) for e in embs.unbind(0))
    return out


@dataclass
class PageIndex:
    doc_id: str
    model_name: str
    embeddings: list  # per-page 2D float16 cpu tensors [seq, dim]
    device: str | None = None

    def search(self, query: str | Image.Image, k: int = 5) -> list[dict]:
        """Retrieve top-k pages for a text or image query. Returns [{page, score}]."""
        with _EMBEDDER_LOCK:
            model, processor, device = _get_embedder(self.model_name, self.device)
            if isinstance(query, Image.Image):
                q_emb = _embed([query], processor.process_images, model, device)
            else:
                q_emb = _embed([str(query)], processor.process_texts, model, device)
        scores = processor.score(
            [q_emb[0].float()], [e.float() for e in self.embeddings]
        )  # [1, num_pages]
        row = scores[0].tolist()
        order = sorted(range(len(row)), key=lambda i: row[i], reverse=True)
        return [
            {"page": i, "score": round(float(row[i]), 3)}
            for i in order[: min(k, len(row))]
        ]


def build_page_index(
    doc_id: str,
    images: list[Image.Image],
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
) -> PageIndex | None:
    """Embed all pages of a document. Returns None if there are no pages."""
    if not images:
        return None
    with _EMBEDDER_LOCK:
        model, processor, resolved = _get_embedder(model_name, device)
        logger.info("vsearch: embedding %d pages for %s", len(images), doc_id)
        embeddings = _embed(images, processor.process_images, model, resolved)
    return PageIndex(doc_id=doc_id, model_name=model_name, embeddings=embeddings, device=device)


def save_page_index(index: PageIndex, vsearch_dir: Path | None = None) -> None:
    import torch

    index_dir = (vsearch_dir or DEFAULT_VSEARCH_DIR) / index.doc_id
    index_dir.mkdir(parents=True, exist_ok=True)
    torch.save(index.embeddings, index_dir / "embeddings.pt")
    (index_dir / "meta.json").write_text(
        json.dumps({"model": index.model_name, "num_pages": len(index.embeddings)})
    )


def load_page_index(
    doc_id: str,
    vsearch_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    num_pages: int | None = None,
    device: str | None = None,
) -> PageIndex | None:
    """Load a cached index. Returns None if missing or stale (model/page-count mismatch)."""
    index_dir = (vsearch_dir or DEFAULT_VSEARCH_DIR) / doc_id
    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("model") != model_name:
            return None
        if num_pages is not None and meta.get("num_pages") != num_pages:
            return None
        import torch

        embeddings = torch.load(index_dir / "embeddings.pt", weights_only=True)
        return PageIndex(doc_id=doc_id, model_name=model_name, embeddings=embeddings, device=device)
    except Exception as e:
        logger.warning("Failed to load vsearch index for %s: %s", doc_id, e)
        return None


def get_or_build_page_index(
    doc_id: str,
    images: list[Image.Image],
    vsearch_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
) -> PageIndex | None:
    """Load cached index or build and cache a new one."""
    cached = load_page_index(doc_id, vsearch_dir, model_name, num_pages=len(images), device=device)
    if cached is not None:
        return cached
    index = build_page_index(doc_id, images, model_name, device)
    if index is None:
        return None
    save_page_index(index, vsearch_dir)
    return index
```

> API verification during this step: the exact kwargs of
> `processor.score(...)` and the output shape of `model(**batch)` must be
> checked against the installed colpali-engine (the model card shows
> `processor.score(q_embeddings, corpus_embeddings)` with batch tensors;
> `BaseVisualRetrieverProcessor.score` accepts lists of 2D tensors). If
> the installed signature differs, adapt `_embed`/`PageIndex.search`
> accordingly — the tests are the contract, not this listing.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_vsearch.py -x -q 2>&1 | tail -5
```

Expected: all 6 tests PASS (first run downloads weights; allow a few minutes).

- [ ] **Step 5: Commit**

```bash
git add src/docvqa/vsearch.py tests/test_vsearch.py
git commit -m "feat: vsearch — multimodal embedding page search (ColModernVBERT)"
```

---

### Task 3: `Document.vsearch_dir` property

**Files:**
- Modify: `src/docvqa/data.py:22-32` (the `Document` dataclass)
- Test: `tests/test_vsearch.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_vsearch.py`)

```python
def test_document_vsearch_dir_derived_from_bm25_dir():
    from docvqa.data import Document

    doc = Document(doc_id="d", doc_category="c", images=[], questions=[],
                   bm25_dir=Path("data/docvqa-2026/val/bm25"))
    assert doc.vsearch_dir == Path("data/docvqa-2026/val/vsearch")

    doc_no_dir = Document(doc_id="d", doc_category="c", images=[], questions=[])
    assert doc_no_dir.vsearch_dir is None
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_vsearch.py::test_document_vsearch_dir_derived_from_bm25_dir -x -q 2>&1 | tail -3
```

Expected: FAIL with `AttributeError: 'Document' object has no attribute 'vsearch_dir'`.

- [ ] **Step 3: Implement** — add to the `Document` dataclass in `src/docvqa/data.py`, after the `question_ids` property:

```python
    @property
    def vsearch_dir(self) -> Path | None:
        """Where this doc's visual-embedding index lives — sibling of bm25_dir
        (``data/{slug}/{split}/vsearch``). None when bm25_dir is unset."""
        return self.bm25_dir.parent / "vsearch" if self.bm25_dir else None
```

(Deriving from `bm25_dir` means all three dataset loaders get this for free — no loader changes.)

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run pytest tests/test_vsearch.py::test_document_vsearch_dir_derived_from_bm25_dir -x -q 2>&1 | tail -3
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/docvqa/data.py tests/test_vsearch.py
git commit -m "feat: Document.vsearch_dir property (sibling of bm25_dir)"
```

---

### Task 4: `rvlm_vsearch` solver

**Files:**
- Create: `src/docvqa/solvers/rvlm_vsearch_solver.py` (clone of `rvlm_solver.py` + edits below)
- Test: `tests/test_vsearch.py` (append one sandbox-code test)

- [ ] **Step 1: Clone the base solver**

```bash
cp src/docvqa/solvers/rvlm_solver.py src/docvqa/solvers/rvlm_vsearch_solver.py
```

- [ ] **Step 2: Apply edits.** Every edit below is exact and complete; everything not mentioned stays as in `rvlm_solver.py`.

**(a) Module docstring** — replace the entire docstring (lines 1–32) with:

```python
"""RVLM+VSEARCH solver — RVLM with an OCR-free visual retrieval channel.

Tool surface = ``batch_look`` (the RVLM recursive sub-call) plus
``search`` — multimodal embedding retrieval over page images
(ColModernVBERT late-interaction, see :mod:`docvqa.vsearch`). The query
can be a text string or a PIL image. Unlike ``rvlm_ocr_ablation``,
there is NO OCR anywhere: no ``page_texts``, no BM25 — the retrieval
extension without the OCR dependency.

Clean fork of :mod:`docvqa.solvers.rvlm_solver`; prompt parity per
D-007 (same minimal ``_TASK_BODY`` plus the generic ``search`` tool
docs, mirroring how ``rvlm_ocr_ablation`` documents its lexical
``search``). No single-image ``look()`` is registered — single visual
queries use ``batch_look([(image, query)])[0]``.
"""
```

**(b) Imports** — after `from docvqa.types import LMConfig` add:

```python
from docvqa.vsearch import PageIndex, get_or_build_page_index
```

**(c) `_create_tools`** — change the signature to

```python
def _create_tools(vlm_predict: dspy.Predict, vlm_lm: dspy.LM, page_index: PageIndex | None, batch_concurrency: int = 8) -> list:
```

and, after `_batch_look_impl`, add `_search` and change the return:

```python
    def _search(query: str, k: int = 5, is_image: bool = False) -> list[dict]:
        """Visual page retrieval. ``query`` is text, or a PNG path when ``is_image``."""
        if page_index is None:
            return [{"error": "No visual search index available"}]
        with logfire.span("vsearch", query=str(query)[:200], k=k, is_image=is_image) as span:
            q = PILImage.open(query) if is_image else query
            records = page_index.search(q, k=k)
            span.set_attribute("num_results", len(records))
            return records

    # Only _batch_look_impl and _search are sandbox-visible tool proxies;
    # _look_impl stays internal (no `look()` symbol in the REPL).
    return [_batch_look_impl, _search]
```

**(d) `_build_sandbox_code`** — inside the returned f-string, after the `batch_look` def, add:

```python
def search(query, k=5):
    """Visual page search. query: a text string OR a PIL Image (e.g. a crop).
    Returns list of {{page, score}} dicts ranked by visual relevance."""
    if isinstance(query, Image.Image):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        query.save(tmp, format="PNG")
        tmp.close()
        return _search(tmp.name, k, True)
    return _search(query, k, False)
```

(Note the doubled `{{page, score}}` — the sandbox code is an f-string template.)

**(e) `_TASK_BODY`** — two insertions:

In `## TOOLS`, between the `batch_look` entry and the `SUBMIT` entry, insert:

```python
    "- search(query, k=5) -> list[dict]\n"
    "  What: retrieve the pages most relevant to a query by visual-semantic "
    "similarity over page images. The query can be a text string "
    "(e.g. 'revenue table 2023') or a PIL Image (e.g. a crop whose match "
    "you want to find elsewhere in the document). Returns [{page, score}] "
    "ranked by relevance — page numbers only, no content; follow up with "
    "`batch_look` on `pages[i]` to read the page.\n"
    "  When: many-page documents — narrow the candidate pages cheaply "
    "before any visual call. For short documents, survey directly.\n"
```

In `## DOCUMENT-SHAPE GUIDANCE`, replace the **Many-page document** bullet with:

```python
    "- **Many-page document** (slides, papers, reports): you do NOT need to "
    "read every page. Use `search(query)` to rank candidate pages, and/or "
    "survey in batches "
    "(`batch_look([(pages[i], 'summarize') for i in sample])`) to build a "
    "table-of-contents in your head. Then drill into the relevant section. "
    "search() is approximate — if the top pages don't contain the answer, "
    "widen k or fall back to a batched survey.\n\n"
```

**(f) Class** — rename `RvlmProgram` → `RvlmVsearchProgram`; extend `__init__` params (after `batch_concurrency: int = 8`):

```python
        vsearch_model: str = "ModernVBERT/colmodernvbert",
        vsearch_device: str | None = None,
```

storing `self.vsearch_model = vsearch_model` and `self.vsearch_device = vsearch_device`.

**(g) `solve_document`** — after the page-PNG dump loop, insert (error handling per spec: tool degrades to an error record; the agent falls back to `batch_look`):

```python
            page_index = None
            try:
                page_index = get_or_build_page_index(
                    document.doc_id,
                    document.images,
                    vsearch_dir=document.vsearch_dir,
                    model_name=self.vsearch_model,
                    device=self.vsearch_device,
                )
            except Exception as e:
                logger.warning("vsearch index failed for %s: %s — search() will return an error", document.doc_id, e)
```

and change the tools line to:

```python
            tools = _create_tools(self.vlm_predict, self.vlm_lm, page_index, self.batch_concurrency)
```

**(h) Renames for observability:** logfire span `"solve_rvlm"` → `"solve_rvlm_vsearch"`; log prefixes `RVLM-MIN` → `RVLM-VS` (all 3 occurrences); factory `create_rvlm_program` → `create_rvlm_vsearch_program`, returning `RvlmVsearchProgram` and passing through the two new params:

```python
def create_rvlm_vsearch_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 25,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
    batch_concurrency: int = 8,
    vsearch_model: str = "ModernVBERT/colmodernvbert",
    vsearch_device: str | None = None,
) -> RvlmVsearchProgram:
```

(body identical to `create_rvlm_program` except the return constructs `RvlmVsearchProgram` with `vsearch_model=vsearch_model, vsearch_device=vsearch_device` added). Also drop the `SEED_TASK_INSTRUCTIONS_LENGTH` line — it belongs to the canonical rvlm module only.

- [ ] **Step 3: Add a sandbox-code sanity test** (append to `tests/test_vsearch.py`)

```python
def test_vsearch_solver_sandbox_code_compiles():
    from docvqa.solvers.rvlm_vsearch_solver import _build_sandbox_code

    code = _build_sandbox_code("/tmp/x", 3)
    compile(code, "<sandbox>", "exec")
    assert "def search(query, k=5):" in code
    assert "def batch_look(requests):" in code
    assert "def look(" not in code
```

- [ ] **Step 4: Run the test suite**

```bash
uv run pytest tests/test_vsearch.py -x -q 2>&1 | tail -3
```

Expected: all PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/docvqa/solvers/rvlm_vsearch_solver.py tests/test_vsearch.py
git commit -m "feat: rvlm_vsearch solver — rvlm + multimodal embedding search tool"
```

---

### Task 5: Hydra config

**Files:**
- Create: `configs/solver/rvlm_vsearch.yaml`

- [ ] **Step 1: Write the config**

```yaml
_target_: docvqa.solvers.rvlm_vsearch_solver.create_rvlm_vsearch_program
dataset: ${data.dataset}
profile_name: null
max_iterations: 25
vlm: ${vlm}
rlm_type: lean
page_factor: 1.5
question_concurrency: 4
batch_concurrency: 8
vsearch_model: ModernVBERT/colmodernvbert
vsearch_device: null   # auto: cuda if available else cpu
```

- [ ] **Step 2: Verify Hydra resolves and the factory instantiates**

```bash
uv run python -c "
from hydra import compose, initialize
from hydra.utils import instantiate
with initialize(config_path='configs', version_base=None):
    cfg = compose(config_name='config', overrides=['solver=rvlm_vsearch'])
    prog = instantiate(cfg.solver)
    print(type(prog).__name__, prog.vsearch_model)
"
```

Expected: `RvlmVsearchProgram ModernVBERT/colmodernvbert`. (If `initialize` complains about the relative path, run from the repo root; `config_path` is relative to the CWD-invoked script.)

- [ ] **Step 3: Commit**

```bash
git add configs/solver/rvlm_vsearch.yaml
git commit -m "feat: rvlm_vsearch solver config"
```

---

### Task 6: Smoke run on fast test docs

**Files:** none created (run artifacts under `output/runs/`, gitignored)

Requires the Qwen 3.5 27B vllm server at `localhost:8927` (check: `curl -s localhost:8927/v1/models | head -c 200`). If it's down, stop and ask the user — don't bring up servers unprompted.

- [ ] **Step 1: Launch the smoke run** (fast test docs per project convention)

```bash
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false solver=rvlm_vsearch \
  data.split=val 'data.doc_ids=[engineering_drawing_2,business_report_3,science_paper_2]' \
  max_concurrency=3 run_id=rvlm-vsearch-smoke-t0
```

Expected: completes without crashing; per-doc accuracy lines in the log.

- [ ] **Step 2: Verify outputs and tool use**

```bash
ls output/runs/rvlm-vsearch-smoke-t0/tasks/*/result.json | wc -l   # expect 3
ls data/docvqa-2026/val/vsearch/                                    # expect the 3 doc dirs
grep -l "search(" output/runs/rvlm-vsearch-smoke-t0/tasks/*/result.json | wc -l  # ≥1: agent actually called search()
```

If the agent never calls `search()` on any doc, inspect one trajectory — likely fine on short docs (prompt says survey directly), but confirm the tool is wired (an explicit `search` mention should appear in the system prompt section of the trajectory).

- [ ] **Step 3: Report results to the user** — smoke accuracy vs the known `rvlm` ballpark on these docs, whether `search()` was used, index build time per doc. Do NOT queue val cells; that's the user's call (coordination protocol).

---

### Task 7: Docs

**Files:**
- Modify: `docs/solvers/README.md` (solver↔paper-role table)

- [ ] **Step 1: Add the solver row** — in the solver map table, after the `rvlm_ocr` row:

```markdown
| **`rvlm_vsearch`** | OCR-free retrieval extension | `batch_look` + `search` (multimodal page embeddings, text-or-image query; no OCR) | — |
```

(Read the surrounding table first and match its exact column conventions; create a `rvlm-vsearch.md` detail doc only if every other solver row links one — otherwise leave the link cell as the table's convention for undocumented solvers.)

- [ ] **Step 2: Commit**

```bash
git add docs/solvers/README.md
git commit -m "docs(solvers): add rvlm_vsearch to solver map"
```

---

## Out of scope (explicitly)

- Queuing experiment cells (`coordination/*.md`) — user decision per the two-host protocol.
- Cross-document retrieval, alternative embedders (ColNomic etc.) — config-swappable later via `vsearch_model`.
- `docs/results.md` / CLAUDE.md updates — only after real cells run.
