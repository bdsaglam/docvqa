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

    if _EMBEDDER is not None:
        logger.warning(
            "vsearch: replacing embedder %s with %s — old model memory is not reclaimed",
            _EMBEDDER[:2],
            (model_name, resolved),
        )

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
        # MaxSim scoring on the embedder's own device. colpali's score()
        # defaults device to get_torch_device("auto") == "cuda:0", which would
        # run the pad+einsum on GPU 0 — where the vllm server lives — and
        # contend with it. Pin to `device` (the embedder GPU, e.g. cuda:1) so
        # scoring never touches GPU 0.
        scores = processor.score(
            [q_emb[0].float()], [e.float() for e in self.embeddings], device=device
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
    import time

    t0 = time.perf_counter()
    with _EMBEDDER_LOCK:
        model, processor, resolved = _get_embedder(model_name, device)
        logger.info("vsearch: embedding %d pages for %s on %s", len(images), doc_id, resolved)
        embeddings = _embed(images, processor.process_images, model, resolved)
    logger.info(
        "vsearch: embedded %d pages for %s in %.1fs",
        len(images),
        doc_id,
        time.perf_counter() - t0,
    )
    return PageIndex(doc_id=doc_id, model_name=model_name, embeddings=embeddings, device=device)


def save_page_index(index: PageIndex, vsearch_dir: Path | None = None) -> None:
    import os

    import torch

    index_dir = (vsearch_dir or DEFAULT_VSEARCH_DIR) / index.doc_id
    index_dir.mkdir(parents=True, exist_ok=True)
    # Atomic writes: two eval-trial processes can share this cache dir, so write
    # each file to a temp path in the same dir then os.replace() it into place
    # (atomic rename). Write meta.json LAST so its presence signals a complete index.
    emb_tmp = index_dir / f"embeddings.pt.tmp.{os.getpid()}"
    torch.save(index.embeddings, emb_tmp)
    os.replace(emb_tmp, index_dir / "embeddings.pt")
    meta_tmp = index_dir / f"meta.json.tmp.{os.getpid()}"
    meta_tmp.write_text(
        json.dumps({"model": index.model_name, "num_pages": len(index.embeddings)})
    )
    os.replace(meta_tmp, index_dir / "meta.json")


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
