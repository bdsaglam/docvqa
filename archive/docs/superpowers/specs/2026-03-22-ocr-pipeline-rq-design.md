# OCR Pipeline Redesign: RQ-Based High-Throughput

## Problem

The current `run_ocr.py` sends 1 page per HTTP request to a single docling-serve container. This:
- Defeats docling's internal pipeline batching (stages process 1 page at a time)
- Uses only 1 of 2 available GPUs
- Achieves ~89s/page on picture-heavy documents (comics_1: 36 pages in 53 minutes)
- Cannot scale beyond a single GPU's throughput

## Solution

Replace the single docling-serve container with an RQ (Redis Queue) architecture: a lightweight API server, Redis, and 6 GPU workers (3 per GPU). Rewrite `run_ocr.py` to submit all pages upfront and poll for results, keeping all workers saturated.

## Infrastructure: docker-compose.yaml

### Architecture

```
┌────────────┐     ┌───────┐     ┌──────────────────┐
│ run_ocr.py │────▶│ Redis │◀───▶│ Worker GPU1 (×3) │
│            │     │ :6373 │     └──────────────────┘
│            │────▶│       │     ┌──────────────────┐
│            │     │       │◀───▶│ Worker GPU2 (×3) │
│            │     └───┬───┘     └──────────────────┘
│            │     ┌───┴───┐
│            │────▶│  API  │
│            │     │ :5001 │
│            │     └───────┘
└────────────┘
```

### Services

| Service | Image | GPU | Purpose |
|---------|-------|-----|---------|
| `redis` | `redis:latest` | — | Task queue, port 6373, no persistence |
| `docling-api` | `docling-serve-cu128` | — | API server, port 5001, `ENG_KIND=rq` |
| `docling-worker-gpu1-{1,2,3}` | `docling-serve-cu128` | GPU 1 | `docling-serve rq-worker` |
| `docling-worker-gpu2-{1,2,3}` | `docling-serve-cu128` | GPU 2 | `docling-serve rq-worker` |

### Why 3 Workers Per GPU

Models consume ~8-10GB per worker (layout ~500MB, EasyOCR ~1-2GB, TableFormer ~500MB, granite-vision-3.3-2b ~6GB). 3 workers × 10GB = ~30GB, well within 80GB A100 VRAM. Multiple workers per GPU enable:
- Process-level parallelism (no GIL)
- One worker does CPU prep while another runs GPU inference
- Independent failure isolation

### Worker Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| `DOCLING_PERF_PAGE_BATCH_SIZE` | `1` | Single page per task |
| `DOCLING_NUM_THREADS` | `4` | Don't over-subscribe CPU across 6 workers |
| `OMP_NUM_THREADS` | `4` | Same |
| `MKL_NUM_THREADS` | `4` | Same |
| `DOCLING_SERVE_LOAD_MODELS_AT_BOOT` | `true` | Avoid cold-start on first task |
| `DOCLING_SERVE_ALLOW_CUSTOM_PICTURE_DESCRIPTION_CONFIG` | `true` | Needed for granite-vision prompt |
| `DOCLING_SERVE_MAX_DOCUMENT_TIMEOUT` | `1200` | 20 min max per page |

### Shared Volumes

- `/home/baris/.cache/huggingface:/cache/huggingface` — shared HF model cache (read-only after download)
- `docling-cache` — shared docling model cache

## Script: scripts/run_ocr.py

### CLI Interface

```
uv run python scripts/run_ocr.py                          # all val docs
uv run python scripts/run_ocr.py --split test             # test set
uv run python scripts/run_ocr.py --num-samples 5
uv run python scripts/run_ocr.py --doc-ids maps_1 comics_1
uv run python scripts/run_ocr.py --force                  # re-process all
uv run python scripts/run_ocr.py --docling-url http://localhost:5001
```

Single `--docling-url` (default `http://localhost:5001`). Throughput is controlled by the number of RQ workers, not the client.

### Flow

```
1. Load dataset from HuggingFace
2. Health check: GET /health on API server, fail fast if unreachable
3. Scan existing OCR files → build list of pending (doc_id, page_idx) pairs
   - A page is "done" if page_{i}.md exists and is non-empty
   - Skip done pages unless --force
4. Submit phase:
   - For each pending page, POST /v1/convert/source/async
   - Store mapping: {task_id → (doc_id, page_idx)}
   - Submit in batches of 50 with small delay between batches
5. Poll phase:
   - Loop over outstanding task_ids
   - Use long-polling: GET /v1/status/poll/{task_id}?wait=2
   - On success: GET /v1/result/{task_id}, extract md_content, write page_{i}.md
   - On failure: log warning, write empty page_{i}.md
   - On timeout (>20 min): treat as failure
   - tqdm progress bar over total pending pages
6. Metadata phase:
   - For each document where all pages are now present, write metadata.json
```

### Request Payload (Per Page)

Same as current — each page is a base64-encoded PNG:

```json
{
  "options": {
    "to_formats": ["md"],
    "from_formats": ["image"],
    "do_ocr": true,
    "do_table_structure": true,
    "do_picture_description": true,
    "image_export_mode": "placeholder",
    "picture_description_area_threshold": 0.05,
    "picture_description_local": {
      "repo_id": "ibm-granite/granite-vision-3.3-2b",
      "prompt": "Describe this image thoroughly...",
      "generation_config": {
        "max_new_tokens": 4096,
        "do_sample": false,
        "repetition_penalty": 1.2
      }
    }
  },
  "sources": [{"kind": "file", "base64_string": "<b64>", "filename": "page.png"}]
}
```

### Output Contract (Unchanged)

```
data/{split}/ocr/{doc_id}/
├── page_0.md
├── page_1.md
├── ...
└── metadata.json   # {doc_id, num_pages, ocr_time_seconds, pages: [...]}
```

### Resumability

- Individual `page_{i}.md` files are the unit of resumability
- Re-running the script skips pages with existing non-empty `.md` files
- `--force` re-processes everything
- `metadata.json` is written per document only when all pages are present
- Interrupted runs leave valid partial state — completed pages are preserved

## Error Handling

| Scenario | Behavior |
|----------|----------|
| API server unreachable | Fail fast before submitting any tasks |
| Task fails (worker error) | Log warning, write empty `.md`, continue |
| Task times out (>20 min) | Treat as failure |
| Worker crashes mid-task | RQ marks job failed, poll loop sees failure status |
| Script interrupted (Ctrl+C) | Written `.md` files persist, re-run resumes |
| Empty OCR result | Log warning, write empty `.md` |

## Expected Performance

| Metric | Current | After |
|--------|---------|-------|
| GPUs | 1 | 2 |
| Workers | 4 threads (GIL-bound) | 6 processes (true parallelism) |
| Queue model | Client round-robin | RQ work-stealing |
| Effective parallelism | ~1-2 pages concurrent | ~6 pages concurrent |
| Estimated speedup | baseline | ~4-6× |

Conservative estimate: comics_1 (36 pages, 53 min currently) → ~10-15 min with 6 workers.

## Files Changed

| File | Change |
|------|--------|
| `docker-compose.yaml` | Replace single container with RQ architecture (redis + api + 6 workers) |
| `scripts/run_ocr.py` | Rewrite: submit-all-then-poll pattern, single URL, no client-side LB |
