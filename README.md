# DocVQA 2026: Active Perception via RLM Agents

ICDAR 2026 DocVQA competition entry. RLM agents with active document perception — iteratively inspect pages via VLM tools from a Python REPL sandbox.

The proposed method is the **`rvlm`** solver: an OCR-free RLM that drives a recursive VLM sub-call (`batch_look`) from a code REPL. The core finding is a **visual context-budget** effect — mid-sized open VLMs are perception-budget-bound, not reasoning-bound, and recursive visual perception is the fix.

## Results

Headline comparison on the DocVQA-2026 val subset (25 docs / 80 questions, Qwen 3.5 27B as both LM and VLM, `n=8` trials, ANLS as mean ± std). `rvlm` leads three clean tiers — every gap is larger than the cross-trial std.

| Tier | Solver | Val (n=8) |
|---|---|---|
| **proposed** | **`rvlm`** — REPL + recursive VLM `batch_look` (OCR-free) | **39.38% ± 1.49** |
| + OCR extension | `rvlm_ocr_ablation` | 37.81% ± 3.12 |
| RL-target twin | `codeact` — append-only/MDP twin of `rvlm` | 36.74% ± 4.29 (n=23) |
| no recursion | `react_baseline` / `direct_vlm` / `raw_vlm_multi_baseline` | 20–25% |
| competition anchor | `official_baseline` (`MASTER_PROMPT`, no scaffold) | 17.81% ± 1.86 |
| OCR-only floor (no vision) | `rlm_ocr` | 13.91% ± 1.56 |

External official baselines (ICDAR 2026, for context): Gemini 3 Pro **37.5%** test, GPT-5.2 **35.0%** test.

`codeact` is `rvlm`'s twin — identical tools, prompt, and `batch_look` — but with a strictly **append-only** transcript (no LeanRLM compaction), making the trajectory a fully-observable MDP suited as an RL fine-tuning target. Dropping compaction is nearly free: pooled **36.74% ± 4.29** over a `max_iterations ∈ {24, 40, 56}` budget sweep (n=23), within noise of `rvlm` (−2.6pp, < the per-budget std) and in the same visual-recursive tier.

See [`docs/results.md`](docs/results.md) for the full cross-solver matrix (ablations, the document-length axis, and the model/perception sweeps) and [`docs/experiment-status.md`](docs/experiment-status.md) for run status.

## Competition submission

The numbers above are from the current `main` codebase. The original ICDAR 2026 competition entry used an earlier solver (`flat_solo`, since refactored away) and is preserved on the **[`docvqa-2026`](../../tree/docvqa-2026)** branch — a frozen snapshot of the code, configs, and prompts as submitted. To reproduce it:

```bash
git checkout docvqa-2026
```

| Config (submission) | Val | Test |
|---|---|---|
| Flat Solo SC-8 (Qwen 3.6 27B) | **51.2%** | **43.75%** |
| Flat Solo SC-8 (Qwen 3.5 27B) | 51.2% | 39.0% |
| Flat Solo (Qwen 3.5 27B) | 48.8% | 35.6% |

Test scores are from the competition server (no public ground truth). These predate the current code and framing and are **not** directly comparable to the `main`-branch results above (different solver, prompts, and per-call retry logic). The submission's method writeup is [`docs/submission-report/submission-summary.pdf`](docs/submission-report/submission-summary.pdf); legacy submission configs are kept under `archive/configs/solver/`.

## Setup

Install dependencies:

```bash
uv sync
```

Copy `.env.example` to `.env` and fill in the credentials for the provider(s) you plan to use:

```bash
cp .env.example .env
```

The dataset is loaded automatically from [HuggingFace](https://huggingface.co/datasets/VLR-CVC/DocVQA-2026) (`val` and `test` splits; no train split). First run caches OCR output under `data/docvqa-2026/{split}/ocr/`. Additional benchmarks (MP-DocVQA, MMLongBench-Doc) live under `data/{dataset-slug}/{split}/`.

### Picking an LM / VLM backend

Every config in `configs/lm/` and `configs/vlm/` is named `<model>-<provider>.yaml`. Pick one of each via Hydra (`lm=...` and `vlm=...`) and supply the env vars that provider needs.

| Config suffix | Provider | Required env vars |
|---|---|---|
| `*-vertex` | Vertex AI (GCP) | `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`, `gcloud auth application-default login` |
| `*-studio` | Google AI Studio | `GEMINI_API_KEY` |
| `*-openrouter` | OpenRouter | `OPENROUTER_API_KEY` |
| `*-vllm-local` / `*-vllm-remote` | self-hosted vLLM | (optional) `HOSTED_VLLM_API_BASE` to override the hardcoded `http://localhost:8927/v1` / `:8928/v1` |

Available configs (see `configs/lm/` and `configs/vlm/` for the full list):

- **Gemini** (closed): `gemini-3_1-pro-{vertex,studio,openrouter}`, `gemini-3-flash-{vertex,studio,openrouter}`
- **Qwen** (local vLLM): `qwen-3_5-{4b,9b,27b}-vllm-local`, `qwen-3_6-{27b,35b}-vllm-local`, `qwen-2_5-vl-72b-vllm-local`; `qwen-3_5-27b-openrouter`
- **Gemma** (local vLLM): `gemma-4-{e4b,31b}-vllm-local`

Headline runs use Qwen 3.5 27B for both `lm` and `vlm`.

The dataset download requires `HF_TOKEN` in `.env` (needed even for public datasets due to rate limits).

### OCR

Each document page is processed via [docling-serve](https://github.com/docling-project/docling-serve) and cached as markdown under `data/docvqa-2026/{split}/ocr/{doc_id}/page_*.md`. BM25 indexes for retrieval are cached under `data/docvqa-2026/{split}/bm25/`. You have two options to populate these caches:

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
  solver=rvlm \
  'data.doc_ids=[business_report_3,engineering_drawing_2,science_paper_2]' \
  max_concurrency=4 run_id=quick-test
```

**Proposed method (`rvlm`), local Qwen:**

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=rvlm \
  data.split=val data.num_samples=null \
  max_concurrency=16 run_id=rvlm-val
```

**Full val eval with Gemini via Vertex:**

```bash
uv run python evals.py \
  lm=gemini-3_1-pro-vertex vlm=gemini-3-flash-vertex \
  solver=rvlm \
  data.split=val data.num_samples=null \
  max_concurrency=8 run_id=rvlm-val-vertex
```

Swap `solver=` for a variant: `rvlm_ocr_ablation` (OCR extension), `codeact`, `direct_vlm`, `react_baseline`, `raw_vlm_multi_baseline`, `official_baseline`, `rlm_ocr`. See `docs/solvers/README.md` for the full map.

All runs are resumable — re-running with the same `run_id` skips already-completed questions.

## Project Structure

```
evals.py                              # Hydra entry point (default solver=rvlm)
configs/
  config.yaml                         # Global defaults
  solver/                             # rvlm (proposed), rvlm_ocr, rvlm_full, codeact, direct_vlm,
                                      #   react_baseline, raw_vlm_{multi,single}_baseline,
                                      #   official_baseline, rlm_ocr, + rvlm_* ablations
  lm/                                 # qwen-3_5-{4b,9b,27b}, qwen-3_6-{27b,35b}, gemma-4-{e4b,31b},
                                      #   gemini-3_1-pro / gemini-3-flash ({vertex,studio,openrouter})
  vlm/                                # same family set as lm/
src/docvqa/
  data.py                             # Dataset loading (HuggingFace), OCR cache
  runner.py                           # Eval runner -- concurrent, resumable
  metrics.py                          # ANLS evaluation
  search.py                           # BM25 index (bm25s)
  prompts.py                          # Per-category tips + answer formatting rules
  datasets/profile.py                 # DatasetProfile + get_profile() (per-dataset prompt/tips/scorer)
  judges/                             # LLM judges (e.g. Qwen judge for MMLongBench scoring)
  rlm/                                # RLM scaffold implementations
    base.py                           # RLM agent base (SubprocessInterpreter)
    lean.py                           # LeanRLM -- minimal prompt
    code.py                           # CodeRLM -- code-only output
    thinking.py                       # ThinkingRLM -- reasoning from thinking tokens
    multimodal.py                     # Multimodal RLM (in-context images)
    subprocess_interpreter.py         # CPython subprocess REPL with IPC
  solvers/                            # one *_solver.py per config above (all dataset-aware)
    rvlm_solver.py                    # Proposed method: REPL + recursive VLM batch_look (OCR-free)
    rlm_ocr_solver.py                 # REPL + OCR text + BM25, no vision (text-perception control)
    direct_vlm_solver.py             # Multimodal agent, in-context display() (no sub-call)
    react_baseline_solver.py          # dspy.ReAct + VLM tools, no REPL
    raw_vlm_multi_baseline_solver.py  # Raw multi-image, single VLM pass, no scaffold
    official_baseline_solver.py       # Competition MASTER_PROMPT, verbatim
    codeact_solver.py                 # Append-only/MDP twin of rvlm (RL-target form)
scripts/
  report.py                           # Generate results reports
  iter_stats.py                       # Per-run agent-iteration stats
  prepare_submission.py               # Build competition submission JSON
docs/
  results.md                          # Cross-solver results (single source of truth)
  experiment-status.md                # Done / in-progress / queued tracker
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

**RLM (Reasoning Language Model).** The LLM writes Python code in a subprocess REPL sandbox and drives perception from it, deciding at each step what to examine next. The proposed `rvlm` solver is **OCR-free**: its one perception primitive is a recursive VLM sub-call, `batch_look(image, query)`. OCR text + BM25 `search` are an extension (`rvlm_ocr_ablation`), not part of the core method.

**Recursive visual perception.** Both halves of the scaffold are load-bearing. Dropping the recursive VLM sub-call (`raw_vlm_multi_baseline`) or the REPL (`react_baseline`) both collapse the score to the no-recursion tier (~20–25% vs `rvlm`'s ~39%); swapping visual perception for OCR text (`rlm_ocr`) is the matrix floor (~14%). Recursive *visual* perception does work OCR text cannot replace.

**Active perception.** Rather than passively processing whole pages, the agent decides what to look at: crops, zooms, multi-scale scans. This focuses computation on the regions that matter for each question — the lever is sharpest on detail-dense categories (engineering drawings, science posters).

**Lean RLM.** A minimal prompt with no chain-of-thought tokens consistently outperforms verbose prompts with explicit reasoning. The code REPL itself serves as the reasoning scratchpad -- structured thinking happens implicitly through iterative tool calls.
