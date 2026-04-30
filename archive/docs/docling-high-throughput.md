# Docling-Serve High-Throughput Guide

## Hardware Available

- **GPU 1**: NVIDIA A100 80GB — free
- **GPU 2**: NVIDIA A100 80GB — currently running docling-serve (single instance)
- **GPU 0**: NVIDIA A100 80GB — occupied by Qwen VLM, not available

## Current Setup & Why It's Slow

### The Problem: One Page = One Request = One Task

Our `run_ocr.py` sends each page image as a **separate HTTP request** to `/v1/convert/source/async`. This means:

1. **No cross-page batching** — Docling's internal 5-stage pipeline (preprocess → OCR → layout → table → assembly) is designed to batch multiple pages through each stage in parallel. With 1 page per request, each stage processes exactly 1 page → GPU is underutilized.
2. **Per-request overhead** — each request creates a new task, gets queued, polled, and result-fetched. With 1-second poll intervals, latency adds up.
3. **`DOCLING_PERF_PAGE_BATCH_SIZE=16` is wasted** — this setting batches pages within a single conversion job. Since each job has only 1 page, batch size 16 does nothing.
4. **`ENG_LOC_NUM_WORKERS=4` partially helps** — 4 worker threads process 4 single-page tasks in parallel, but each worker's GPU inference is unbatched. We get concurrency but not batching.

### Docling's Internal Pipeline (5-stage producer-consumer)

```
Pages Queue → [Preprocess] → [OCR] → [Layout] → [Table Structure] → [Assembly] → Result
                 thread 1    thread 2   thread 3      thread 4         thread 5

Each stage has a bounded queue. Pages flow through stages concurrently.
When batch_size=N, each stage waits to accumulate N pages before running GPU inference.
```

**This pipeline shines when a single task contains many pages** — e.g., a 50-page PDF flows 50 pages through 5 parallel stages with batch_size=8, keeping the GPU saturated. Our 1-page-per-task approach defeats this entirely.

## Architecture Options

### Option A: Multi-Source Requests (Minimal Change)

The v1 API accepts **multiple sources per request**:

```json
{
  "options": { "to_formats": ["md"], "do_ocr": true, ... },
  "sources": [
    {"kind": "file", "base64_string": "<page0_b64>", "filename": "page_0.png"},
    {"kind": "file", "base64_string": "<page1_b64>", "filename": "page_1.png"},
    {"kind": "file", "base64_string": "<page2_b64>", "filename": "page_2.png"}
  ]
}
```

All sources are processed in a **single task**, enabling cross-page batching. However, each source is still a separate "document" to docling — the pipeline batching (`DOCLING_PERF_PAGE_BATCH_SIZE`) applies within a document, not across documents in the same request.

**Verdict**: Modest improvement. Reduces HTTP/task overhead but doesn't unlock intra-pipeline batching across pages.

### Option B: Combine Pages into Multi-Page PDF Before Sending

Convert all page images for a document into a single multi-page PDF (using Pillow or pikepdf), then send that as one source:

```python
from PIL import Image
images = [page_img for page_img in document_pages]
pdf_bytes = BytesIO()
images[0].save(pdf_bytes, format="PDF", save_all=True, append_images=images[1:])
# Send pdf_bytes as a single source
```

Now docling receives a multi-page document and its pipeline batches pages through all 5 stages with the configured `DOCLING_PERF_PAGE_BATCH_SIZE`.

**Verdict**: Best for pipeline utilization. A 10-page document with `page_batch_size=16` processes all 10 pages through each stage as a batch. This is how docling is designed to be used.

**Caveat**: We need individual per-page markdown output. Docling returns one combined markdown for the whole document. We'd need to split the output by page markers (docling's markdown includes page breaks).

### Option C: RQ Workers (2 GPUs, Redis Queue)

```
┌──────────┐     ┌───────┐     ┌────────────────────────────┐
│ run_ocr  │────▶│ Redis │────▶│ RQ Worker (GPU 1)          │
│  client  │     │ Queue │     │  docling-serve rq-worker   │
└──────────┘     │       │     └────────────────────────────┘
      │          │       │     ┌────────────────────────────┐
      └─────────▶│       │────▶│ RQ Worker (GPU 2)          │
                 └───────┘     │  docling-serve rq-worker   │
                               └────────────────────────────┘
                 ┌────────────────────────────┐
                 │ API Server (CPU only)      │
                 │  docling-serve (port 5001) │
                 │  ENG_KIND=rq               │
                 └────────────────────────────┘
```

- **API server**: lightweight, no GPU, accepts requests and enqueues to Redis
- **Workers**: each pulls one task at a time, processes it with full GPU pipeline
- **Work stealing**: fast worker pulls next task as soon as done — handles uneven page sizes naturally
- **Single URL**: client sends everything to `:5001`, queue distributes

**docker-compose** (proposed, do not apply yet):

```yaml
services:
  redis:
    image: redis:latest
    command: ["redis-server", "--port", "6373", "--appendonly", "yes"]
    ports: ["6373:6373"]
    restart: unless-stopped

  docling-api:
    image: ghcr.io/docling-project/docling-serve-cu128:latest
    container_name: docling-api
    ports: ["5001:5001"]
    environment:
      DOCLING_SERVE_ENABLE_UI: "true"
      DOCLING_SERVE_ENG_KIND: "rq"
      DOCLING_SERVE_ENG_RQ_REDIS_URL: "redis://redis:6373/"
      DOCLING_SERVE_MAX_SYNC_WAIT: "1200"
      DOCLING_SERVE_MAX_DOCUMENT_TIMEOUT: "1200"
    restart: unless-stopped

  docling-worker-gpu1:
    image: ghcr.io/docling-project/docling-serve-cu128:latest
    container_name: docling-worker-gpu1
    command: ["docling-serve", "rq-worker"]
    environment:
      DOCLING_SERVE_ENG_KIND: "rq"
      DOCLING_SERVE_ENG_RQ_REDIS_URL: "redis://redis:6373/"
      DOCLING_SERVE_ALLOW_CUSTOM_PICTURE_DESCRIPTION_CONFIG: "true"
      DOCLING_DEVICE: "cuda"
      DOCLING_NUM_THREADS: "16"
      DOCLING_PERF_PAGE_BATCH_SIZE: "16"
      OMP_NUM_THREADS: "16"
      MKL_NUM_THREADS: "16"
      DOCLING_SERVE_LOAD_MODELS_AT_BOOT: "true"
      HF_HOME: "/cache/huggingface"
    volumes:
      - /home/baris/.cache/huggingface:/cache/huggingface
      - docling-cache:/opt/app-root/src/.cache/docling
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
    restart: unless-stopped

  docling-worker-gpu2:
    image: ghcr.io/docling-project/docling-serve-cu128:latest
    container_name: docling-worker-gpu2
    command: ["docling-serve", "rq-worker"]
    environment:
      DOCLING_SERVE_ENG_KIND: "rq"
      DOCLING_SERVE_ENG_RQ_REDIS_URL: "redis://redis:6373/"
      DOCLING_SERVE_ALLOW_CUSTOM_PICTURE_DESCRIPTION_CONFIG: "true"
      DOCLING_DEVICE: "cuda"
      DOCLING_NUM_THREADS: "16"
      DOCLING_PERF_PAGE_BATCH_SIZE: "16"
      OMP_NUM_THREADS: "16"
      MKL_NUM_THREADS: "16"
      DOCLING_SERVE_LOAD_MODELS_AT_BOOT: "true"
      HF_HOME: "/cache/huggingface"
      DOCLING_SERVE_MAX_DOCUMENT_TIMEOUT: "1200"
    volumes:
      - /home/baris/.cache/huggingface:/cache/huggingface
      - docling-cache:/opt/app-root/src/.cache/docling
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["2"]
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  docling-cache:
```

**Verdict**: Best for multi-GPU scaling. Combines naturally with Option B (send multi-page PDFs → each worker processes a full document with batched pipeline).

### Option D: Use Docling Python Library Directly (Skip HTTP)

Instead of going through the HTTP API, call `docling` as a library:

```python
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
results = converter.convert_all(sources)  # processes with full pipeline
```

**Pros**: Zero HTTP overhead, direct control over pipeline options, can process a batch of documents.
**Cons**: Must manage GPU assignment, model loading, and parallelism ourselves. Loses the queue/retry infrastructure.

**Verdict**: Best raw throughput for a one-shot batch job, but more engineering. Consider if RQ doesn't meet needs.

## Recommended Approach: Option B + C Combined

### Why

1. **Option B** (multi-page PDF per document) maximizes **per-task pipeline efficiency** — pages flow through the 5-stage pipeline as batches, keeping GPU saturated within each task.
2. **Option C** (RQ with 2 workers) maximizes **cross-task parallelism** — two workers on two GPUs process two documents simultaneously with work-stealing semantics.
3. **Combined**: each worker processes a full multi-page document with batched pipeline stages, and idle workers immediately pick up the next document.

### Expected Throughput Gain

| Bottleneck | Current | After |
|-----------|---------|-------|
| GPU utilization per task | ~10% (1 page, no batching) | ~80%+ (multi-page, batched stages) |
| GPU parallelism | 1 GPU | 2 GPUs |
| Task overhead | 1 HTTP req + poll per page | 1 HTTP req + poll per document |
| Work distribution | Client round-robin (uneven) | Queue work-stealing (balanced) |

Conservative estimate: **4-10x throughput improvement** from batching alone, **~2x** from second GPU → **8-20x total**.

## Tuning Knobs (for A100 80GB)

| Setting | Current | Recommended | Why |
|---------|---------|-------------|-----|
| `DOCLING_PERF_PAGE_BATCH_SIZE` | 16 | **32-64** | A100 has 80GB VRAM; batch more pages per stage. Only effective with multi-page documents. |
| `DOCLING_SERVE_OCR_BATCH_SIZE` | unset (default 4) | **16** | Batch OCR stage for GPU efficiency |
| `DOCLING_SERVE_LAYOUT_BATCH_SIZE` | unset (default 4) | **16** | Batch layout model inference |
| `DOCLING_SERVE_TABLE_BATCH_SIZE` | unset (default 4) | **8** | Table structure is memory-heavy |
| `DOCLING_SERVE_BATCH_POLLING_INTERVAL_SECONDS` | unset | **0.05** | Don't wait long to start processing a batch (low latency) |
| `DOCLING_SERVE_ENG_LOC_NUM_WORKERS` | 4 | **1** (with RQ) | RQ worker = single-threaded; internal pipeline handles parallelism |
| `DOCLING_PERF_ELEMENTS_BATCH_SIZE` | unset (default 8) | **16** | Batch enrichment (picture description) elements |

## run_ocr.py Redesign Sketch

The current script sends 1 page per request. A redesigned version would:

1. **Combine all page images into a multi-page PDF** per document (Pillow `save_all=True`)
2. **Send 1 request per document** (not per page) to the async endpoint
3. **Split the combined markdown output** by page markers to get per-page files
4. **Submit all documents upfront** (don't wait for one to finish before starting next)
5. **Poll in batch** — check multiple task IDs concurrently

```
Current:  25 docs × 10 pages = 250 HTTP requests, 250 tasks, 250 poll loops
Proposed: 25 docs = 25 HTTP requests, 25 tasks, 25 poll loops
          Each task processes 10 pages with batched GPU pipeline
```

## Picture Description: The Hidden Bottleneck

`granite-vision-3.3-2b` runs per-picture during the enrichment phase. This is likely the dominant cost per page:
- Layout/OCR: ~100ms/page on GPU
- Picture description: ~2-10s per picture depending on complexity
- A page with 3 figures = 6-30s in VLM inference alone

**Options to address**:
- `picture_description_area_threshold`: currently 0.05 (5% of page area). Increase to skip small decorative images.
- `DOCLING_PERF_ELEMENTS_BATCH_SIZE`: batch enrichment elements (may help if VLM supports batching internally).
- **Disable for text-heavy docs**: science papers, business reports rarely have informative figures. Only enable for infographics, posters.
- **External VLM**: Route picture description to a vLLM-served model that handles batched inference natively (our Qwen at :8927 could do this, but that's a separate integration).

## Profiling

Enable pipeline timing to identify the actual bottleneck before over-optimizing:

```
DOCLING_DEBUG_PROFILE_PIPELINE_TIMINGS=true
```

This will log per-stage timings, showing exactly which stage (OCR, layout, table, picture description) dominates.

## Sources

- [Docling GPU Support](https://docling-project.github.io/docling/usage/gpu/)
- [Docling RTX Tuning Guide](https://docling-project.github.io/docling/getting_started/rtx/)
- [Best Performance Settings (GitHub Discussion)](https://github.com/docling-project/docling/discussions/2516)
- [Multi-threading Discussion](https://github.com/docling-project/docling/issues/1256)
- [Parallel Parsing Discussion](https://github.com/docling-project/docling/discussions/2757)
- [Docling-Serve Configuration](https://docling-project.github.io/docling-serve/docs/configuration.md)
- [Docling-Serve RQ Workers Example](https://github.com/docling-project/docling-serve/blob/main/docs/deploy-examples/docling-serve-rq-workers.yaml)
- [Docling Technical Report](https://arxiv.org/html/2408.09869v4)
