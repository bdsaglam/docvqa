# DocVQA 2026: Active Perception via RLM Agents

ICDAR 2026 DocVQA competition entry. RLM agents with active document perception — iteratively inspect pages via VLM tools from a Python REPL sandbox.

## Results

| Config | Val Score | Test Score |
|--------|-----------|------------|
| Flat Solo SC-8 (Qwen 3.6 27B) | **51.2%** | **43.75%** |
| Flat Solo SC-8 (Qwen 3.5 27B) | 51.2% | 41.0% |
| Flat Solo (Qwen 3.5 27B) | 48.8% | 35.6% |
| Official Gemini 3 Pro baseline | — | 37.5% |
| Official GPT-5.2 baseline | — | 35.0% |

Test scores are from the competition server (no public ground truth). Val scores use ANLS (Average Normalized Levenshtein Similarity).

## Setup

Install dependencies:

```bash
uv sync
```

Copy `.env.example` to `.env` and fill in the credentials for the provider(s) you plan to use:

```bash
cp .env.example .env
```

The dataset is loaded automatically from [HuggingFace](https://huggingface.co/datasets/VLR-CVC/DocVQA-2026) (`val` and `test` splits; no train split). First run caches OCR output under `data/{split}/ocr/`.

### Picking an LM / VLM backend

Every config in `configs/lm/` and `configs/vlm/` is named `<model>-<provider>.yaml`. Pick one of each via Hydra (`lm=...` and `vlm=...`) and supply the env vars that provider needs.

| Config suffix | Provider | Required env vars |
|---|---|---|
| `*-vertex` | Vertex AI (GCP) | `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`, `gcloud auth application-default login` |
| `*-studio` | Google AI Studio | `GEMINI_API_KEY` |
| `*-openrouter` | OpenRouter | `OPENROUTER_API_KEY` |
| `*-vllm-local` / `*-vllm-remote` | self-hosted vLLM | (optional) `HOSTED_VLLM_API_BASE` to override the hardcoded `http://localhost:8927/v1` / `:8928/v1` |

Available configs:

- `lm/gemini-3-flash-{vertex,studio,openrouter}` — Gemini 3 Flash Preview
- `lm/gemini-3_1-pro-{vertex,studio,openrouter}` — Gemini 3.1 Pro Preview
- `lm/qwen-3_5-27b-{vllm-local,vllm-remote,openrouter}` — Qwen 3.5 27B
- `vlm/gemini-3-flash-{vertex,studio,openrouter}` — Gemini 3 Flash Preview
- `vlm/gemini-3_1-pro-{vertex,studio,openrouter}` — Gemini 3.1 Pro Preview
- `vlm/qwen-3_5-27b-{vllm-local,vllm-remote,openrouter}` — Qwen 3.5 27B

The dataset download requires `HF_TOKEN` in `.env` (needed even for public datasets due to rate limits).

### OCR

Each document page is processed via [docling-serve](https://github.com/docling-project/docling-serve) and cached as markdown under `data/{split}/ocr/{doc_id}/page_*.md`. BM25 indexes for retrieval are cached under `data/{split}/bm25/`. You have two options to populate these caches:

**Option A — Download the pre-built bundle (recommended).** A ~13 MB zip with OCR markdown and BM25 indexes for all val + test documents:

```bash
# Download from Google Drive (file ID: 1LgLyEkDuDyl_roS2ZlXjhWFE2JhJPg9K)
uv run --with gdown gdown 1LgLyEkDuDyl_roS2ZlXjhWFE2JhJPg9K -O data.zip
unzip -o data.zip -d .  # extracts into ./data/{val,test}/{ocr,bm25}/
rm data.zip
```

Or download manually from [this link](https://drive.google.com/file/d/1LgLyEkDuDyl_roS2ZlXjhWFE2JhJPg9K/view?usp=drive_link) and unzip into the repo root.

**Option B — Run the OCR pipeline locally.** Reproducible from scratch but requires a GPU and takes a while:

```bash
# Start docling-serve locally (GPU-accelerated)
docker run --gpus '"device=0"' -p 5001:5001 quay.io/docling-project/docling-serve

# Extract OCR for val + test docs
uv run python scripts/run_ocr.py
uv run python scripts/run_ocr.py --split test
```

Pass `--docling-url http://host:port` if docling-serve runs elsewhere. If you skip this step entirely, OCR is also produced lazily on first eval run, but pre-populating the cache avoids OCR overhead bleeding into eval timings.

## Running Solvers

**Quickstart (OpenRouter — only one API key needed):**

```bash
uv run python evals.py \
  lm=gemini-3_1-pro-openrouter vlm=gemini-3-flash-openrouter \
  solver=flat_solo solver.rlm_type=lean \
  'data.doc_ids=[business_report_3,engineering_drawing_2,science_paper_2]' \
  max_concurrency=4 run_id=quick-test
```

**Best single-run solver (Flat Solo, local Qwen):**

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=flat_solo \
  data.split=val data.num_samples=null \
  max_concurrency=16 run_id=flat-solo-val
```

**Full val eval with Gemini via Vertex:**

```bash
uv run python evals.py \
  lm=gemini-3_1-pro-vertex vlm=gemini-3-flash-vertex \
  solver=flat_solo solver.rlm_type=lean \
  data.split=val data.num_samples=null \
  max_concurrency=8 run_id=flat-solo-val-vertex
```

All runs are resumable — re-running with the same `run_id` skips already-completed questions.

## Project Structure

```
evals.py                              # Hydra entry point
configs/
  config.yaml                         # Global defaults
  solver/                             # flat_solo, leanest_solo, lean_solo, flat_batch, routing, rvlm
  lm/                                 # gemini-3-flash-{vertex,studio,openrouter}, gemini-3_1-pro-{vertex,studio,openrouter}, qwen-3_5-27b-{vllm-local,vllm-remote,openrouter}
  vlm/                                # same set as lm/
src/docvqa/
  data.py                             # Dataset loading (HuggingFace), OCR cache
  runner.py                           # Eval runner -- concurrent, resumable
  metrics.py                          # ANLS evaluation
  search.py                           # BM25 index (bm25s)
  prompts.py                          # Per-category tips + answer formatting rules
  rlm/                                # RLM implementations
    base.py                           # RLM agent base (SubprocessInterpreter)
    lean.py                           # LeanRLM -- minimal prompt
    code.py                           # CodeRLM -- code-only output
    thinking.py                       # ThinkingRLM -- reasoning from thinking tokens
    subprocess_interpreter.py         # CPython subprocess REPL with IPC
  solvers/
    flat_solo_solver.py               # Best solo solver
    leanest_solo_solver.py            # Minimal tool solo solver
    lean_solo_solver.py               # Lean solo with BM25 search
    flat_batch_solver.py              # Batch solver baseline
    rvlm_solver.py                    # Multimodal agent (sees images via display())
    routing_solver.py                 # Category-based solver routing
scripts/
  report.py                           # Generate results reports
  prepare_submission.py               # Build competition submission JSON
docs/
  results.md                          # Full results with per-category breakdown
  solvers/                            # Per-solver documentation
```

## Competition

[ICDAR 2026 DocVQA](https://rrc.cvc.uab.es/?ch=34) challenges systems to answer questions about documents that require multimodal reasoning across 8 domains:

- Business reports
- Science papers
- Science posters
- Maps
- Comics
- Infographics
- Engineering drawings
- Presentation slides

Evaluation uses ANLS (Average Normalized Levenshtein Similarity). Dataset: [VLR-CVC/DocVQA-2026](https://huggingface.co/datasets/VLR-CVC/DocVQA-2026).

## Design

**RLM (Reasoning Language Model).** The LLM writes Python code in a subprocess REPL sandbox. It calls VLM tools iteratively -- `look(image, query)` for visual inspection, `search(query, k)` for BM25 retrieval -- deciding at each step what to examine next.

**Active perception.** Rather than passively processing whole pages, the agent decides what to look at: crops, zooms, multi-scale scans, BM25 search hits. This lets it focus computation on the regions that matter for each question.

**Solo over batch.** Answering one question at a time yields roughly 10 percentage points higher accuracy than batch processing. The agent can dedicate its full context window and iteration budget to a single question.

**Lean RLM.** A minimal prompt with no chain-of-thought tokens consistently outperforms verbose prompts with explicit reasoning. The code REPL itself serves as the reasoning scratchpad -- structured thinking happens implicitly through iterative tool calls.
