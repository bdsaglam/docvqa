"""BM25 search index for document pages.

Pre-indexes OCR text per page so the agent can search for relevant
content without scanning every page manually.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import bm25s
import Stemmer

logger = logging.getLogger(__name__)

# Default index dir; callers can override via bm25_dir parameter
DEFAULT_BM25_DIR = Path("data/val/bm25")


def _chunk_page(page_num: int, text: str, max_chunk_chars: int = 500) -> list[dict]:
    """Split a page's text into chunks for finer-grained search."""
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chunk_chars:
            chunks.append({"page": page_num, "text": current})
            current = para
        else:
            current = current + "\n\n" + para if current else para
    if current:
        chunks.append({"page": page_num, "text": current})
    return chunks


def build_index(doc_id: str, page_texts: list[str]) -> bm25s.BM25 | None:
    """Build a BM25 index from per-page OCR text. Returns None if no text."""
    chunks = []
    for i, text in enumerate(page_texts):
        chunks.extend(_chunk_page(i, text))
    if not chunks:
        return None

    corpus = [c["text"] for c in chunks]
    tokens = bm25s.tokenize(corpus, stemmer=Stemmer.Stemmer("english"))

    retriever = bm25s.BM25()
    retriever.index(tokens)

    # Stash chunk metadata on the retriever for lookup
    retriever._chunk_meta = chunks  # type: ignore[attr-defined]
    return retriever


def save_index(doc_id: str, retriever: bm25s.BM25, chunks: list[dict], bm25_dir: Path = DEFAULT_BM25_DIR) -> None:
    """Save BM25 index and chunk metadata to disk."""
    index_dir = bm25_dir / doc_id
    index_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(index_dir))
    (index_dir / "chunks.json").write_text(json.dumps(chunks))


def load_index(doc_id: str, bm25_dir: Path = DEFAULT_BM25_DIR) -> tuple[bm25s.BM25, list[dict]] | None:
    """Load a cached BM25 index from disk. Returns None if not cached."""
    index_dir = bm25_dir / doc_id
    if not (index_dir / "chunks.json").exists():
        return None
    try:
        retriever = bm25s.BM25.load(str(index_dir))
        chunks = json.loads((index_dir / "chunks.json").read_text())
        retriever._chunk_meta = chunks  # type: ignore[attr-defined]
        return retriever, chunks
    except Exception as e:
        logger.warning("Failed to load BM25 index for %s: %s", doc_id, e)
        return None


def get_or_build_index(doc_id: str, page_texts: list[str], bm25_dir: Path = DEFAULT_BM25_DIR) -> bm25s.BM25 | None:
    """Load cached index or build and cache a new one."""
    cached = load_index(doc_id, bm25_dir)
    if cached is not None:
        retriever, _ = cached
        return retriever

    retriever = build_index(doc_id, page_texts)
    if retriever is None:
        return None

    save_index(doc_id, retriever, retriever._chunk_meta, bm25_dir)  # type: ignore[attr-defined]
    return retriever


def make_search_tool(retriever: bm25s.BM25, top_k: int = 5):
    """Create a search tool function for the agent."""
    chunks = retriever._chunk_meta  # type: ignore[attr-defined]

    def search(query: str, k: int = top_k) -> list[dict]:
        """Search document text using BM25. Returns list of {page, score, text} records.
        Use this to quickly find relevant pages/sections without calling inspect_page."""
        query_tokens = bm25s.tokenize([query], stemmer=Stemmer.Stemmer("english"))
        n = min(k, len(chunks))
        results, scores = retriever.retrieve(query_tokens, k=n)

        records = []
        for idx, score in zip(results[0], scores[0]):
            if score <= 0:
                continue
            chunk = chunks[idx]
            records.append({
                "page": chunk["page"],
                "score": round(float(score), 2),
                "text": chunk["text"],
            })

        return records

    return search
