# OCR Infrastructure

## docling-serve (RQ Architecture)

Uses a Redis Queue setup with multiple GPU workers for high throughput.

- **Image**: `ghcr.io/docling-project/docling-serve-cu128:latest`
- **API server**: `localhost:5001` (CPU-only, enqueues to Redis)
- **Workers**: 6 total — 3 on GPU 1, 3 on GPU 2 (A100 80GB each)
- **Redis**: `localhost:6373` (task queue, ephemeral)
- **tmux**: `docvqa:docling-serve` (docker compose in foreground)
- **Source code**: `./tmp/docling-serve/` for reference
- **Components**: EasyOCR (text extraction) + granite-vision-3.3-2b (picture descriptions)
- **Docs**: See `docs/docling.md` for how docling works, `docs/docling-high-throughput.md` for scaling analysis

```bash
# Start/stop
docker compose up      # foreground (see logs)
docker compose up -d   # background
docker compose down
```

## OCR Pipeline Script

- **Script**: `scripts/run_ocr.py`
- **Pattern**: Submit all pending pages as async tasks upfront, then poll for results. Throughput scales with number of RQ workers.
- **Output**: `data/{split}/ocr/{doc_id}/page_N.md` + `metadata.json`
- **Resumable**: Skips pages with existing non-empty `.md` files unless `--force`

```bash
uv run python scripts/run_ocr.py                          # all val docs
uv run python scripts/run_ocr.py --split test             # test set
uv run python scripts/run_ocr.py --num-samples 5
uv run python scripts/run_ocr.py --doc-ids maps_1 comics_1
uv run python scripts/run_ocr.py --force                  # re-process all
```

## Picture Description
- **Model**: `ibm-granite/granite-vision-3.3-2b` (runs inside docling-serve workers)
- **Prompt**: "Describe this image thoroughly. Include all visible text, labels, numbers, and data. Describe the layout, any charts, tables, diagrams, or figures. Be specific about colors, positions, and relationships between elements."
- **Settings**: max_new_tokens=4096, do_sample=False, repetition_penalty=1.2, area_threshold=0.05

## OCR Quality by Document Type
- **Science papers/reports**: Good text extraction via EasyOCR
- **Comics**: Panel-by-panel dialogue extraction (175K chars for 36-page comic)
- **Maps/engineering drawings**: Limited text extraction — mostly image descriptions, actual labels require VLM tools
- **Business reports/slides**: Good for text-heavy content

## Data Organization
- `data/val/ocr/` — 25 docs (processing in progress)
- `data/test/ocr/` — 48 docs (not started)

## Integration
- **RLM agent**: Loaded by `src/docvqa/data.py` as `page_texts` on each Document
- **BM25 search**: Index auto-built per doc during eval from OCR text (`src/docvqa/search.py`)
