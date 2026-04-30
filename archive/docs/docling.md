# How Docling & Docling-Serve Work

## Overview

**Docling** is an open-source (MIT) document conversion library by IBM that turns PDFs, images, Office files, and HTML into structured formats (Markdown, JSON, HTML). It uses AI models for layout analysis, OCR, table structure recognition, and picture description — all running locally with no data sent to external services.

**Docling-serve** wraps Docling in a production-ready FastAPI service, exposing its conversion pipeline through REST endpoints with async task management, an optional Gradio UI, and multiple compute backends.

## Docling Core: The Conversion Pipeline

### Three Main Concepts

1. **Parser backends** — read raw input formats. For PDFs: extract text + geometry and render page images. For markup formats (HTML, DOCX): directly create a `DoclingDocument`.
2. **Pipelines** — orchestration layer that iterates through pages, applies a chain of AI models to build and enrich the `DoclingDocument`.
3. **DoclingDocument** — the unified internal data model: a structured tree of text, tables, figures, reading order, and metadata.

### Pipeline Stages

```
Input → Format Detection → Backend Selection → Pipeline Execution → Export
                                                    │
                                        ┌───────────┼───────────┐
                                        ▼           ▼           ▼
                                      BUILD     ASSEMBLE     ENRICH
                                   (layout,     (reading    (picture
                                    OCR,        order,      description,
                                    tables)     assembly)   classification)
```

**BUILD phase** — per-page processing:
- **Layout model** (`granite-docling-258M` or heron/egret variants): detects document elements (headings, paragraphs, tables, figures, formulas) and their bounding boxes
- **OCR engine** (EasyOCR on GPU, Tesseract on CPU, or RapidOCR): extracts text from scanned/image content. Native PDFs skip OCR since text is embedded.
- **TableFormer**: recognizes table structure (rows, columns, spanning cells). Two modes: `FAST` and `ACCURATE`.

**ASSEMBLE phase** — document-level:
- Determines reading order across elements
- Merges multi-page structures
- Builds the DoclingDocument tree

**ENRICH phase** — optional post-processing:
- **Picture description**: a VLM (e.g., `granite-vision-3.3-2b`, SmolVLM) generates text descriptions of images/figures
- **Picture classification**: categorizes figure types
- **Code/formula recognition**: extracts code blocks and mathematical formulas

### Native vs. Scanned PDFs

- **Native PDFs**: text is already embedded — Docling extracts it directly from the PDF structure, no OCR needed. Layout model still runs to understand structure.
- **Scanned PDFs / Images**: the full pipeline runs — layout detection finds regions, OCR extracts text from each region, TableFormer processes table regions.

### Supported OCR Engines

| Engine | Class | Best for |
|--------|-------|----------|
| EasyOCR | `EasyOcrOptions` | GPU environments (default) |
| Tesseract | `TesseractOcrOptions` | CPU, fast |
| RapidOCR | `RapidOcrOptions` | Custom models |
| macOS Vision | `OcrMacOptions` | macOS native |

Auto-detection: Docling probes the environment at init and picks the best available engine.

## Docling-Serve: The API Service

### Architecture

```
Client  ──POST──▶  FastAPI App  ──▶  Orchestrator  ──▶  DocumentConverter (Docling)
                      │                    │
                      │              ┌─────┼─────┐
                      │              ▼     ▼     ▼
                      │           Local    RQ    KFP
                      │          (threads) (Redis) (K8s)
                      │
                      ├── /v1/convert/source       (sync)
                      ├── /v1/convert/source/async  (async + poll)
                      ├── /v1/convert/file          (sync, multipart upload)
                      ├── /v1/status/poll/{task_id} (poll task status)
                      ├── /v1/result/{task_id}      (get result)
                      └── /v1/chunk/...             (chunking endpoints)
```

### Execution Engines

| Engine | `ENG_KIND` | How it works |
|--------|-----------|-------------|
| **Local** | `local` | Tasks run in separate threads within the same process. `ENG_LOC_NUM_WORKERS` controls parallelism. Models can be shared across workers. |
| **RQ** | `rq` | Tasks queued in Redis, processed by separate `rq_worker` processes. Redis tracks task status via `RedisTaskStatusMixin`. Good for scaling. |
| **KFP** | `kfp` | Tasks submitted as Kubeflow Pipeline runs for Kubernetes-native orchestration. Experimental. |

### Async Task Lifecycle

1. **Submit**: `POST /v1/convert/source/async` → returns `task_id`
2. **Poll**: `GET /v1/status/poll/{task_id}` → returns status (`pending`, `started`, `success`, `failure`)
3. **Retrieve**: `GET /v1/result/{task_id}` → returns converted document

The sync endpoints (`/v1/convert/source`) internally do the same but poll in a loop up to `MAX_SYNC_WAIT` seconds.

### Key Configuration (Environment Variables)

| Variable | Default | What it does |
|----------|---------|-------------|
| `DOCLING_DEVICE` | auto | `cpu`, `cuda`, `cuda:N`, `mps` |
| `DOCLING_NUM_THREADS` | 4 | Torch CPU threads |
| `DOCLING_PERF_PAGE_BATCH_SIZE` | 4 | Pages processed per batch |
| `DOCLING_SERVE_ENG_LOC_NUM_WORKERS` | 2 | Parallel worker threads |
| `DOCLING_SERVE_MAX_SYNC_WAIT` | 120 | Sync endpoint timeout (seconds) |
| `DOCLING_SERVE_MAX_DOCUMENT_TIMEOUT` | 7 days | Max processing time |
| `DOCLING_SERVE_LOAD_MODELS_AT_BOOT` | true | Pre-load models on startup |
| `DOCLING_SERVE_ALLOW_CUSTOM_PICTURE_DESCRIPTION_CONFIG` | false | Allow custom VLM for picture description |

### Model Management

Models are loaded from `DOCLING_SERVE_ARTIFACTS_PATH`. If unset, they auto-download from HuggingFace on first use. Pre-download with:
```sh
docling-tools models download layout tableformer easyocr granite_vision
```

Models can be mounted via Docker volumes (our approach — see below).

## How We Use It: Our OCR Pipeline

### Docker Compose Setup (`docker-compose.yaml`)

```yaml
services:
  docling-serve:
    image: ghcr.io/docling-project/docling-serve-cu128:latest
    ports: ["5001:5001"]
    environment:
      DOCLING_SERVE_ENABLE_UI: "true"
      DOCLING_SERVE_ALLOW_CUSTOM_PICTURE_DESCRIPTION_CONFIG: "true"
      DOCLING_SERVE_ENG_LOC_NUM_WORKERS: "4"       # 4 parallel conversion threads
      DOCLING_DEVICE: "cuda"                         # GPU acceleration
      DOCLING_NUM_THREADS: "16"
      DOCLING_PERF_PAGE_BATCH_SIZE: "16"
      DOCLING_SERVE_LOAD_MODELS_AT_BOOT: "true"     # pre-load all models
    volumes:
      - ~/.cache/huggingface:/cache/huggingface      # shared HF model cache
      - docling-cache:/opt/app-root/src/.cache/docling
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["2"]                       # pinned to GPU 2
              capabilities: [gpu]
```

Key choices:
- **CUDA image** (`docling-serve-cu128`): GPU-accelerated layout model + OCR
- **Custom picture description enabled**: lets us pass custom VLM config per-request
- **4 workers**: processes 4 documents concurrently
- **Models at boot**: avoids cold-start latency on first request
- **GPU 2**: dedicated GPU, doesn't conflict with VLM serving on other GPUs

### OCR Script (`scripts/run_ocr.py`)

Pre-processes all DocVQA documents through docling-serve, saving markdown per page.

**Request payload** (per page image):
```python
{
    "options": {
        "to_formats": ["md"],           # output as markdown
        "from_formats": ["image"],      # input is a page image
        "do_ocr": True,                 # run OCR on image content
        "do_table_structure": True,     # recognize table layouts
        "do_picture_description": True, # describe embedded images
        "image_export_mode": "placeholder",
        "picture_description_area_threshold": 0.05,  # skip tiny images (<5% of page)
        "picture_description_local": {
            "repo_id": "ibm-granite/granite-vision-3.3-2b",  # VLM for descriptions
            "prompt": "Describe this image thoroughly...",
            "generation_config": {
                "max_new_tokens": 4096,
                "do_sample": False,
                "repetition_penalty": 1.2
            }
        }
    },
    "sources": [{"kind": "file", "base64_string": "<b64>", "filename": "page.png"}]
}
```

**Flow**:
1. Load dataset images from HuggingFace (`VLR-CVC/DocVQA-2026`)
2. For each document, check if cached OCR exists (`data/{split}/ocr/{doc_id}/page_*.md`)
3. Send each page as base64 PNG to `/v1/convert/source/async`
4. Poll `/v1/status/poll/{task_id}` until success (up to 10 min per page)
5. Retrieve result from `/v1/result/{task_id}`, extract `md_content`
6. Strip `<!-- image -->` placeholders, save as `page_N.md`
7. Pages processed concurrently (default 4-8 threads), round-robin across docling-serve instances

**Output structure**:
```
data/val/ocr/
├── business_report_1/
│   ├── page_0.md          # markdown OCR text for page 0
│   ├── page_1.md
│   └── metadata.json      # {doc_id, num_pages, ocr_time_seconds, pages: [...]}
├── infographics_1/
│   ├── page_0.md
│   └── metadata.json
└── ...
```

### What the Pipeline Produces

For each page, docling-serve returns **structured markdown** with:
- **Headings** preserved with `#` hierarchy
- **Body text** with reading order
- **Tables** as markdown tables (via TableFormer)
- **Picture descriptions** as text paragraphs (via granite-vision-3.3-2b)
- **Lists, captions, footnotes** in their semantic role

This markdown is then used by the RLM agent's BM25 search index for text retrieval during question answering.

## Sources

- [How Docling Works: Architecture and Design Decisions](https://www.codesota.com/ocr/docling/explanation)
- [Docling Technical Report](https://arxiv.org/html/2408.09869v4)
- [Docling GitHub](https://github.com/docling-project/docling)
- [Docling-Serve DeepWiki](https://deepwiki.com/docling-project/docling-serve)
- [Docling Pipeline Options Reference](https://docling-project.github.io/docling/reference/pipeline_options/)
- [IBM Granite-Docling Announcement](https://www.ibm.com/new/announcements/granite-docling-end-to-end-document-conversion)
