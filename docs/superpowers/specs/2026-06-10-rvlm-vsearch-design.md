# Design: `rvlm_vsearch` — OCR-free visual retrieval solver

**Date:** 2026-06-10
**Status:** approved (design review with BDS)

## Goal

A new solver that adds a **multimodal embedding search tool** to the rvlm
scaffold: within-document page retrieval over page images, where the query
can be **text or an image crop**. Fully **OCR-free** — this is the retrieval
extension without the OCR dependency, sitting beside `rvlm_ocr_ablation`
(lexical retrieval) as a clean ablation: same `search(query, k)` tool name
and prompt shape, different retrieval backend.

## Decisions made during design review

| Decision | Choice |
|---|---|
| Retrieval scope | Within-doc pages (doc is given per question; docs are 2–54 pages) |
| Embedding model | `ModernVBERT/colmodernvbert` — 250M, MIT, late-interaction, CPU-capable; within ~0.6 NDCG@5 of ColPali ([arXiv:2510.01149](https://arxiv.org/abs/2510.01149)) |
| Tooling | `colpali-engine` (MIT, v0.3.17; `ColModernVBert` / `ColModernVBertProcessor` confirmed in main) |
| Solver composition | `batch_look` + embedding retrieval, **no BM25** — isolates the visual-retrieval contribution |
| Sandbox tool name | `search(query, k)` — prompt parity with `rvlm_ocr_ablation` (D-007) |
| Solver name | `rvlm_vsearch` |
| Embedder lifecycle | In-process lazy singleton + per-doc on-disk cache (mirrors BM25 pattern); no precompute script, no serving infra |

Rejected alternatives: dense embedders (GME — weaker on visually-rich docs),
ColNomic-7B (better quality but needs a GPU slot; can swap later via config),
jina-embeddings-v4 (Qwen Research License, not open), cross-document
retrieval (different experimental setting; nothing in the pipeline needs it).

## Components

### 1. `src/docvqa/vsearch.py` (sibling of `search.py`)

- **Lazy global embedder**: ColModernVBERT loaded via colpali-engine on
  first use, behind a thread lock (runner is concurrent). Device: `cuda` if
  available else `cpu`, overridable via config. Note: model card uses
  `trust_remote_code=True`.
- `get_or_build_page_index(doc_id, images, vsearch_dir) -> PageIndex`:
  load cached per-page multi-vector embeddings or embed and cache.
  Cache at `data/{slug}/{split}/vsearch/{doc_id}/`:
  - `embeddings.pt` — float16 list-of-tensors (~200 KB/page)
  - `meta.json` — model name + page count; cache invalidated if the model
    name differs.
- `PageIndex.search(query: str | PIL.Image, k) -> [{page, score}]`:
  text query → `processor.process_texts`, image query →
  `processor.process_images`; MaxSim scoring via `processor.score`;
  returns top-k page indices + scores. **No text snippets** — the agent
  follows up with `pages[i]` / `batch_look`.

### 2. `src/docvqa/solvers/rvlm_vsearch_solver.py`

Cloned from `rvlm_ocr_ablation_solver.py` with the BM25 parts swapped:

- Tools: `_batch_look_impl` + `_search` closure over the `PageIndex`.
- Sandbox wrapper `search(query, k=5)` accepts a string **or a PIL image**
  (image saved to temp PNG and passed by path — same pattern as
  `batch_look`).
- `_TASK_BODY` documents `search()` as visual retrieval: query with text or
  with an image crop; returns page numbers ranked by visual relevance.
  Otherwise prompt-parity with the OCR variant (D-007).
- No `page_texts` / OCR anywhere in the solver.

### 3. `configs/solver/rvlm_vsearch.yaml`

Clone of `rvlm_ocr_ablation.yaml` with the new `_target_`, plus:

```yaml
vsearch_model: ModernVBERT/colmodernvbert
vsearch_device: null   # auto: cuda if available else cpu
```

### 4. Dependency

Add `colpali-engine` to `pyproject.toml`. This pulls `torch`/`transformers`
into the env for the first time (multi-GB install, uv.lock churn). One-time
cost; no alternative for local embeddings.

## Error handling

Embedder load failure or empty index → the tool returns an error string
into the REPL (same convention as existing tools); the agent sees it and
falls back to `batch_look` over all pages.

## Testing & validation

1. Unit test for `vsearch.py` round-trip: build → cache → reload → text
   query and image query on a tiny image set; marked slow (downloads
   weights on first run).
2. Smoke run on the fast test docs (`engineering_drawing_2`,
   `business_report_3`, `science_paper_2`).
3. n=1 val cell queued per D-008 (`rvlm-vsearch-val-t1`), compared against
   `rvlm` and `rvlm_ocr_ablation`.

**Expected-value framing:** our results show OCR search adds ≈0 on DocVQA
val (moderate docs; rvlm already sees all-page thumbnails) and pays off on
long-doc benchmarks. Visual retrieval likely follows the same pattern — its
differentiators are visual matching (charts, stamps, layout) and
image-as-query. Frame as the long-doc extension story, not a DocVQA-val win.

## Implementation notes

- Verify at install time that the released colpali-engine (0.3.17) ships
  `ColModernVBert` (confirmed in main branch; card's "not yet merged" note
  is stale from Oct 2025).
- Embedding a 54-page doc on CPU is a one-time per-doc cost amortized by
  the cache; if it proves slow, set `vsearch_device: cuda` — 250M is small
  enough to coexist with vllm.
