"""Pre-extract text from DocVQA documents using docling-serve RQ workers.

OCR + picture description (granite-vision) via docling-serve with Redis queue.
All pages are submitted as async tasks upfront, then polled for results.
Throughput scales with the number of RQ workers (configured in docker-compose).

Usage:
    uv run python scripts/run_ocr.py                             # DocVQA-2026 val
    uv run python scripts/run_ocr.py --split test
    uv run python scripts/run_ocr.py --num-samples 5
    uv run python scripts/run_ocr.py --doc-ids maps_1 infographics_1
    uv run python scripts/run_ocr.py --force                      # re-process all
    uv run python scripts/run_ocr.py --docling-url http://localhost:5001

    # MP-DocVQA / MMLongBench-Doc — uses the existing load_documents dispatch:
    uv run python scripts/run_ocr.py \\
        --dataset lmms-lab/MP-DocVQA \\
        --doc-ids-file data/mp-docvqa/val/sample_200q_doc_ids.txt
    uv run python scripts/run_ocr.py \\
        --dataset yubo2333/MMLongBench-Doc \\
        --doc-ids-file data/mmlongbench-doc/val/sample_200q_doc_ids.txt

Requires: docker compose up -d (redis + api + workers)
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from docvqa.data import load_documents

# Allow very large images (some test docs are 200M+ pixels)
Image.MAX_IMAGE_PIXELS = 500_000_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET = "VLR-CVC/DocVQA-2026"
DEFAULT_SPLIT = "val"
DATA_DIR = Path("data")
DOCLING_URL = "http://localhost:5001"


# Map HF dataset id → on-disk slug for the OCR output directory. Kept
# in sync with the per-dataset layout convention
# (``data/<dataset-slug>/<split>/ocr/<doc_id>/page_*.md``).
DATASET_SLUGS = {
    "VLR-CVC/DocVQA-2026": "docvqa-2026",
    "lmms-lab/MP-DocVQA": "mp-docvqa",
    "yubo2333/MMLongBench-Doc": "mmlongbench-doc",
}


def _default_ocr_dir(dataset: str, split: str) -> Path:
    slug = DATASET_SLUGS.get(dataset)
    if slug is None:
        # Fall back to the dataset id with slashes replaced — robust to new
        # datasets even if they're not in the explicit table.
        slug = dataset.split("/", 1)[-1].lower().replace("/", "-")
    return DATA_DIR / slug / split / "ocr"

PICTURE_PROMPT = (
    "Describe this image thoroughly. Include all visible text, labels, numbers, "
    "and data. Describe the layout, any charts, tables, diagrams, or figures. "
    "Be specific about colors, positions, and relationships between elements."
)

TASK_TIMEOUT = 0  # disabled — RQ workers have their own job timeout (4h)
POLL_INTERVAL = 2  # seconds between poll sweeps
SUBMIT_BATCH_SIZE = 50  # pages per submit batch


@dataclass
class PendingPage:
    doc_id: str
    page_idx: int
    image: Image.Image
    task_id: str | None = None
    submitted_at: float = 0.0


def build_payload(img: Image.Image, picture_description: bool = False) -> dict:
    """Build the docling-serve conversion request payload for a single page.

    ``picture_description`` enables the granite-vision picture-description
    pipeline. The current docling-serve build (2026-05-12) raises an
    HFValidationError when loading the granite-vision processor from a
    local cache path, so picture description is disabled by default.
    OCR + table-structure extraction still produce the page markdown we
    need for flat_solo's BM25/text channel.
    """
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    options: dict = {
        "to_formats": ["md"],
        "from_formats": ["image"],
        "do_ocr": True,
        "do_table_structure": True,
        "do_picture_description": picture_description,
        "image_export_mode": "placeholder",
    }
    if picture_description:
        options["picture_description_area_threshold"] = 0.05
        options["picture_description_local"] = {
            "repo_id": "ibm-granite/granite-vision-3.3-2b",
            "prompt": PICTURE_PROMPT,
            "generation_config": {
                "max_new_tokens": 4096,
                "do_sample": False,
                "repetition_penalty": 1.2,
            },
        }
    return {
        "options": options,
        "sources": [{"kind": "file", "base64_string": b64, "filename": "page.png"}],
    }


def submit_page(page: PendingPage, docling_url: str) -> bool:
    """Submit a single page as an async task. Returns True on success."""
    payload = build_payload(page.image)
    try:
        resp = requests.post(
            f"{docling_url}/v1/convert/source/async", json=payload, timeout=60
        )
        resp.raise_for_status()
        page.task_id = resp.json()["task_id"]
        page.submitted_at = time.time()
        return True
    except Exception as e:
        logger.warning("Submit failed for %s page %d: %s", page.doc_id, page.page_idx, e)
        return False


def poll_and_collect(
    outstanding: list[PendingPage],
    docling_url: str,
    ocr_dir: Path,
    pbar: tqdm,
) -> list[PendingPage]:
    """Poll all outstanding tasks once. Returns still-outstanding pages."""
    still_pending = []
    for page in outstanding:
        if page.task_id is None:
            still_pending.append(page)
            continue

        try:
            poll = requests.get(
                f"{docling_url}/v1/status/poll/{page.task_id}?wait=2", timeout=10
            )
            status = poll.json().get("task_status", "")
        except Exception as e:
            logger.debug("Poll error for %s page %d: %s", page.doc_id, page.page_idx, e)
            still_pending.append(page)
            continue

        if status == "success":
            md = _fetch_result(docling_url, page.task_id, page.doc_id, page.page_idx)
            _write_page(ocr_dir, page.doc_id, page.page_idx, md)
            pbar.update(1)
        elif status == "failure":
            error_msg = poll.json().get("error_message", "unknown")
            logger.warning(
                "Failed: %s page %d: %s", page.doc_id, page.page_idx, error_msg
            )
            _write_page(ocr_dir, page.doc_id, page.page_idx, "")
            pbar.update(1)
        else:
            # pending, started, or other — keep waiting
            still_pending.append(page)

    return still_pending


def _fetch_result(docling_url: str, task_id: str, doc_id: str, page_idx: int) -> str:
    """Fetch and clean the markdown result for a completed task."""
    try:
        result = requests.get(f"{docling_url}/v1/result/{task_id}", timeout=30)
        result.raise_for_status()
        md = result.json()["document"]["md_content"]
        md = md.replace("\n\n<!-- image -->", "").replace("<!-- image -->", "")
        return md
    except Exception as e:
        logger.warning("Result fetch failed for %s page %d: %s", doc_id, page_idx, e)
        return ""


def _write_page(ocr_dir: Path, doc_id: str, page_idx: int, markdown: str) -> None:
    """Write a single page markdown file."""
    doc_dir = ocr_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / f"page_{page_idx}.md").write_text(markdown)


def write_metadata(
    ocr_dir: Path, doc_id: str, num_pages: int, start_time: float
) -> None:
    """Write metadata.json for a completed document."""
    doc_dir = ocr_dir / doc_id
    pages = []
    for i in range(num_pages):
        md_path = doc_dir / f"page_{i}.md"
        char_count = len(md_path.read_text()) if md_path.exists() else 0
        pages.append({"page_num": i, "char_count": char_count})

    metadata = {
        "doc_id": doc_id,
        "num_pages": num_pages,
        "ocr_time_seconds": round(time.time() - start_time, 1),
        "pages": pages,
    }
    (doc_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Pre-extract OCR text using docling-serve RQ")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--doc-ids", nargs="+", default=None)
    parser.add_argument(
        "--doc-ids-file",
        type=Path,
        default=None,
        help="File with one doc_id per line (# comments ignored). Mutually exclusive with --doc-ids.",
    )
    parser.add_argument("--docling-url", default=DOCLING_URL)
    parser.add_argument("--force", action="store_true", help="Re-process even if cached")
    parser.add_argument("--ocr-dir", type=Path, default=None)
    args = parser.parse_args()

    ocr_dir = args.ocr_dir or _default_ocr_dir(args.dataset, args.split)
    logger.info("OCR output dir: %s", ocr_dir)

    # Health check
    try:
        resp = requests.get(f"{args.docling_url}/health", timeout=5)
        resp.raise_for_status()
        logger.info("docling-serve healthy at %s", args.docling_url)
    except Exception:
        logger.error(
            "docling-serve not reachable at %s. Start with: docker compose up -d",
            args.docling_url,
        )
        return

    # Resolve doc_ids — explicit list, then file, then None (all docs).
    doc_ids: list[str] | None = args.doc_ids
    if doc_ids is None and args.doc_ids_file is not None:
        doc_ids = [
            line.strip()
            for line in args.doc_ids_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        logger.info("Loaded %d doc_ids from %s", len(doc_ids), args.doc_ids_file)

    # Use the same loader the eval pipeline does — dispatches by dataset_name.
    documents = load_documents(
        args.dataset,
        args.split,
        num_samples=args.num_samples,
        doc_ids=doc_ids,
    )
    logger.info(
        "Loaded %d documents with %d total pages",
        len(documents),
        sum(len(d.images) for d in documents),
    )

    # Build list of pending pages
    pending: list[PendingPage] = []
    doc_page_counts: dict[str, int] = {}

    for doc in documents:
        doc_id = doc.doc_id
        images = doc.images
        doc_page_counts[doc_id] = len(images)

        for i, img in enumerate(images):
            md_path = ocr_dir / doc_id / f"page_{i}.md"
            if not args.force and md_path.exists() and md_path.stat().st_size > 0:
                continue
            pending.append(PendingPage(doc_id=doc_id, page_idx=i, image=img))

    if not pending:
        logger.info("All %d documents already processed", len(documents))
        return

    logger.info(
        "Processing %d pages across %d documents (%d pages cached)",
        len(pending),
        len({p.doc_id for p in pending}),
        sum(doc_page_counts.values()) - len(pending),
    )

    ocr_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Submit phase: fire all pages as async tasks
    logger.info("Submitting %d pages...", len(pending))
    submit_failures = []
    for i in range(0, len(pending), SUBMIT_BATCH_SIZE):
        batch = pending[i : i + SUBMIT_BATCH_SIZE]
        for page in batch:
            if not submit_page(page, args.docling_url):
                submit_failures.append(page)
        if i + SUBMIT_BATCH_SIZE < len(pending):
            time.sleep(0.1)  # brief pause between batches

    # Remove pages that failed to submit
    outstanding = [p for p in pending if p.task_id is not None]
    for page in submit_failures:
        logger.warning("Could not submit %s page %d, writing empty", page.doc_id, page.page_idx)
        _write_page(ocr_dir, page.doc_id, page.page_idx, "")

    logger.info("Submitted %d tasks, %d failed", len(outstanding), len(submit_failures))

    # Poll phase: collect results
    pbar = tqdm(total=len(pending), initial=len(submit_failures), desc="OCR pages")
    while outstanding:
        outstanding = poll_and_collect(outstanding, args.docling_url, ocr_dir, pbar)
        if outstanding:
            time.sleep(POLL_INTERVAL)
    pbar.close()

    # Write metadata for all documents
    for doc_id, num_pages in doc_page_counts.items():
        write_metadata(ocr_dir, doc_id, num_pages, start_time)

    elapsed = time.time() - start_time
    total_pages = sum(doc_page_counts.values())
    logger.info(
        "Done. %d pages across %d documents in %.1fs (%.1fs/page avg). Results: %s",
        total_pages,
        len(documents),
        elapsed,
        elapsed / max(total_pages, 1),
        ocr_dir,
    )


if __name__ == "__main__":
    main()
