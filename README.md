# DocVQA 2026: Perceive-Reason-Code

Active perception for document visual question answering — the ICDAR 2026 DocVQA challenge entry that [jointly won the 8–35B model tier](https://rrc.cvc.uab.es/?com=news&view=data&id=83). An open Qwen 3.5 27B beats the challenge's bare-model baselines — the far larger Gemini 3 Pro and GPT-5.2 — on the held-out test set.

The method fits in one sentence: give a code-capable model a persistent Python REPL and a single perception primitive — an on-demand VLM sub-call pointed at any region of any page — and let it **direct** its own perception instead of reading whole pages at a fixed resolution. Three moves name the system, **Perceive-Reason-Code**: perceive through a VLM call, reason in language, act by writing code. The proposed solver is **`rvlm`** — an OCR-free RLM driving a recursive VLM sub-call (`batch_look`) from the REPL.

The reframe underneath: on documents, **perception is the constraint but the reasoner is the lever**. No model resolves a dense page in one fixed-resolution look, so what separates systems is how well the reasoner directing the looks spends its budget of them — scaling the reasoner moves accuracy about twice as much as scaling the VLM it looks through.

Full write-up: [**Perceive-Reason-Code: Active Perception for Document VQA**](https://barisdeniz.is-a.dev/posts/perceive-reason-code/).

## Results

### Held-out test set (competition)

The challenge scores one submitted answer per question on a private test set (self-consistency vote over 8 samples). Two entries are ours — a **tuned entry** fitted to this benchmark (DocVQA-specific prompts plus OCR and search) and the **general method** this repo is built around, which drops all of that. Both clear the challenge's official baselines, which are bare models reported with no agentic scaffold — a harnessed 27B against unharnessed frontier models.

| System (held-out test set) | ANLS |
|---|---|
| **Ours — tuned entry** (`flat_solo`, SC-8) | **43.75%** |
| **Ours — general method** (`rvlm`, SC-8) | **39.38%** |
| Gemini 3 Pro | 37.50% |
| GPT-5.2 | 35.00% |
| Gemini 3 Flash | 33.75% |
| GPT-5 Mini | 22.50% |

The general method's append-only twin, `codeact_chat`, scores **41.25%** on test — statistically tied with `rvlm` (the two submissions agree on 68% of answers). See [`docs/results.md`](docs/results.md) for the test-submission detail.

### Validation matrix (ablations)

Every ablation below runs on the DocVQA-2026 val subset (25 docs / 80 questions), Qwen 3.5 27B as both reasoner and VLM, `n=8` trials, reported as **avg@1** (single-trial ANLS, mean ± std). `rvlm` leads three clean tiers — every cross-tier gap is far larger than the within-cell std.

| Tier | Solver | Val avg@1 (n=8) |
|---|---|---|
| **proposed (full method)** | **`rvlm`** — REPL + recursive VLM `batch_look` (OCR-free) | **41.88% ± 5.79** |
| corrected MDP twin | `codeact_chat` — append-only chat-MDP twin (RL-target form) | 39.53% ± 2.83 |
| + general sub-agent | `rvlm_subagent_ablation` | 36.72% ± 2.75 |
| + OCR & search | `rvlm_ocr_ablation` | 36.56% ± 2.89 |
| no REPL | `react_baseline` | 27.19% ± 3.19 |
| pixels in-context (no sub-call) | `direct_vlm` | 22.34% ± 2.79 |
| raw multi-image (no scaffold) | `raw_vlm_multi_baseline` | 20.94% ± 1.60 |
| competition prompt (no scaffold) | `official_baseline` (`MASTER_PROMPT`) | 18.91% ± 1.94 |
| OCR-only floor (no vision) | `rlm_ocr` | 14.69% ± 2.19 |

Both halves of the scaffold are load-bearing: dropping the REPL (`react_baseline`) or the recursive sub-call (`raw_vlm_multi_baseline`, `direct_vlm`) collapses the score to the no-recursion tier (~21–27%), and swapping visual perception for OCR text (`rlm_ocr`) is the matrix floor. The enrichments that *don't* help — a general sub-agent, OCR on top, richer trajectory management — cost accuracy or buy nothing.

`codeact_chat` is `rvlm`'s twin: identical tools, prompt, and `batch_look`, but conditioning on a strictly **append-only** chat transcript (no RLM-style compaction), which keeps the trajectory a growing-prefix MDP suited as an RL fine-tuning target. The append-only form costs essentially nothing — **39.53% ± 2.83** (n=8), statistically tied with `rvlm` (which edges +2.35pp, within combined std). (An earlier dspy-based `codeact` solver — single-turn `dspy.Predict` re-rendering history into a string field, a POMDP-shaped approximation — is **deprecated** in favor of this corrected chat-MDP version; its budget-sweep writeup is archived.)

See [`docs/results.md`](docs/results.md) for the full cross-solver matrix (the reasoner × VLM size matrix, the document-length axis, and the model/family sweeps) and [`docs/experiment-status.md`](docs/experiment-status.md) for run status.

## Reproducing the competition entries

The **general method** (`rvlm`) is the current `main` codebase — run it directly (see below). The **tuned entry** used an earlier solver (`flat_solo`, since refactored away) and is preserved on the **[`docvqa-2026`](../../tree/docvqa-2026)** branch — a frozen snapshot of the code, configs, and prompts as submitted:

```bash
git checkout docvqa-2026
```

| Tuned entry (submission) | Val | Test |
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
| `*-vllm-local` | self-hosted vLLM | (optional) `HOSTED_VLLM_API_BASE` to override the hardcoded `http://localhost:8927/v1` |

Available configs (see `configs/lm/` and `configs/vlm/` for the full list):

- **Gemini** (closed): `gemini-3_1-pro-{vertex,studio,openrouter}`, `gemini-3-flash-{vertex,studio,openrouter}`
- **Qwen** (local vLLM): `qwen-3_5-{4b,9b,27b}-vllm-local`, `qwen-3_6-{27b,35b}-vllm-local`, `qwen-2_5-vl-72b-vllm-local`; `qwen-3_5-27b-openrouter`
- **Gemma** (local vLLM): `gemma-4-{e4b,31b}-vllm-local`

Headline runs use Qwen 3.5 27B for both `lm` and `vlm`.

The dataset download requires `HF_TOKEN` in `.env` (needed even for public datasets due to rate limits).

### OCR

The proposed `rvlm` method is OCR-free, so OCR is only needed for the OCR solvers (`rvlm_ocr_ablation`, `rlm_ocr`). Each document page is processed via [docling-serve](https://github.com/docling-project/docling-serve) and cached as markdown under `data/docvqa-2026/{split}/ocr/{doc_id}/page_*.md`; BM25 indexes for retrieval are cached under `data/docvqa-2026/{split}/bm25/`. OCR is produced lazily on first eval run, or you can pre-populate the cache (avoids OCR overhead bleeding into eval timings):

```bash
# Start docling-serve locally (GPU-accelerated)
docker run --gpus '"device=0"' -p 5001:5001 quay.io/docling-project/docling-serve

# Extract OCR for val + test docs
uv run python scripts/run_ocr.py
uv run python scripts/run_ocr.py --split test
```

Pass `--docling-url http://host:port` if docling-serve runs elsewhere.

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

Swap `solver=` for a variant: `rvlm_ocr_ablation` (OCR extension), `codeact_chat` (MDP/RL-target twin), `direct_vlm`, `react_baseline`, `raw_vlm_multi_baseline`, `official_baseline`, `rlm_ocr`. See `docs/solvers/README.md` for the full map.

All runs are resumable — re-running with the same `run_id` skips already-completed questions.

## Project Structure

```
evals.py                              # Hydra entry point (default solver=rvlm)
configs/
  config.yaml                         # Global defaults
  solver/                             # rvlm (proposed), rvlm_ocr, rvlm_full, codeact_chat, direct_vlm,
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
    codeact_chat_solver.py            # Append-only chat-MDP twin of rvlm (RL-target form)
    codeact_solver.py                 # DEPRECATED dspy-based codeact (superseded by codeact_chat)
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

**RLM (Recursive Language Model).** The LLM writes Python code in a subprocess REPL sandbox and drives perception from it, deciding at each step what to examine next. The proposed `rvlm` solver is **OCR-free**: its one perception primitive is a recursive VLM sub-call, `batch_look(image, query)`. OCR text + BM25 `search` are an extension (`rvlm_ocr_ablation`), not part of the core method.

**Recursive visual perception.** Both halves of the scaffold are load-bearing. Dropping the recursive VLM sub-call (`raw_vlm_multi_baseline`, `direct_vlm`) or the REPL (`react_baseline`) both collapse the score to the no-recursion tier (~21–27% vs `rvlm`'s ~42%); swapping visual perception for OCR text (`rlm_ocr`) is the matrix floor (~15%). Recursive *visual* perception does work OCR text cannot replace — and it has to arrive as a **compact-text sub-call**, not raw pixels poured into the reasoner's own context (`direct_vlm` does exactly that, pins the iteration cap, and collapses too).

**Active perception.** Rather than passively processing whole pages, the agent decides what to look at: crops, zooms, multi-scale scans. This focuses computation on the regions that matter for each question — the advantage is sharpest on detail-dense categories (engineering drawings, science posters).

**The reasoner is the lever.** Perception binds in the moment — no single look resolves a dense page — but the leverage sits with the reasoner directing the looks. At a fixed 27B VLM, scaling the reasoner (4B→27B) adds ~+20pp; at a fixed 27B reasoner, scaling the VLM adds ~+9pp. The same extra reasoning capacity is worth 3–4× as much inside the loop as in a whole-page ReAct agent, and a strong reasoner peering through a weak 4B VLM (32.8%) beats a weak reasoner given the full 27B VLM (21.1%). Between better eyes and a better director, buy the director.

**Lean RLM.** A minimal prompt with no chain-of-thought tokens consistently outperforms verbose prompts with explicit reasoning. The code REPL itself serves as the reasoning scratchpad — structured thinking happens implicitly through iterative tool calls.
