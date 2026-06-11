# rvlm_vsearch — Qwen 3.5 27B (val)

> **Status: PROVISIONAL — n=6, paused before t7/t8 (2026-06-11).**
> Numbers below are the mean of 6 trials; t7/t8 are queued but not run
> (user paused the chain). Do not cite as a locked n=8 headline. The
> `rvlm` (39.38 ± 1.49) and `rvlm_ocr_ablation` (37.81 ± 3.12) references
> are n=8.

## Hypothesis / question

`rvlm_vsearch` is `rvlm` + a **multimodal embedding `search` tool** —
within-document page retrieval over page *images* (no OCR). The query can
be text or an image crop; pages are embedded with **ColModernVBERT**
(250M, MIT, late-interaction) via `colpali-engine`, scored by MaxSim, and
`search(query, k)` returns the top-k page indices. It is the **OCR-free
retrieval extension**: same role as `rvlm_ocr_ablation` (which adds BM25
+ `page_texts`), but retrieval is visual-semantic instead of lexical, and
there is no OCR anywhere.

Question: does a visual page retriever help over plain `rvlm` (which
already sees every page via `batch_look`)? Design framing
(`docs/superpowers/specs/2026-06-10-rvlm-vsearch-design.md`): expected
≈ neutral on moderate-length DocVQA val (same as OCR search), with the
real payoff on long-doc benchmarks where surveying every page is
infeasible.

## Setup

- Solver: `rvlm_vsearch` (`batch_look` + visual `search`, OCR-free)
- Retriever: `ModernVBERT/colmodernvbert` on `cuda:1` (placement only)
- Model: Qwen 3.5 27B local vllm 8927 (lm + vlm), `enable_thinking=false`
- Profile: DocVQA-2026 (default); val 25 docs / 80 Qs
- max_concurrency: 24
- Per-doc page embeddings cached at `data/docvqa-2026/val/vsearch/<doc_id>/`

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_vsearch solver.vsearch_device=cuda:1 \
  data.split=val data.num_samples=null \
  max_concurrency=24 \
  run_id=rvlm-vsearch-val-tN
```

## Results (n=6, provisional)

| Trial | Score | Notes |
|---|---|---|
| t1 | 40.00% | |
| t2 | 33.75% | recovered — science_paper_1 q6 batch_look wedge |
| t3 | 33.75% | |
| t4 | 38.75% | recovered — comics_4 q2 batch_look wedge |
| t5 | 37.50% | recovered — comics_2 batch_look wedge |
| t6 | 36.25% | recovered — science_paper_1 wedge; resume ran under the fix (no wedge) |
| **mean** | **36.67% ± 2.36** | n=6 |

### Comparison

| Solver | Val | Δ vs `rvlm` |
|---|---|---|
| `rvlm` (n=8) | 39.38% ± 1.49 | — |
| `rvlm_ocr_ablation` (n=8) | 37.81% ± 3.12 | −1.56pp |
| **`rvlm_vsearch` (n=6, prov.)** | **36.67% ± 2.36** | **−2.71pp** |

`rvlm_vsearch` lands ≈ at the OCR-retrieval ablation and ~2.7pp below
`rvlm` — within the combined std, so **not a real difference** at this n.
Read: **visual retrieval adds nothing measurable over already-sees-all-
pages recursive vision on moderate val docs**, mirroring the OCR-search
result. The motivating test is long-doc benchmarks (MP-DocVQA /
MMLongBench-Doc), where surveying every page isn't feasible — pending.

## Tool-use behavior (logfire spans, this run)

- The agent **uses `search` actively**: ~3 calls/question (≈600 logged),
  alongside ~20 `batch_look` calls/question. Pattern is as designed —
  `search` to locate pages, `batch_look` to read them. Queries are
  well-formed and content-targeted (e.g. "share repurchases",
  "PARTIAL VIEW schematic").
- **Text-only in practice:** all ~600 `search` calls were text queries;
  the agent issued **0 image-crop queries**. The image-query path is
  wired and unit-tested but **latent** — the prompt's parenthetical
  mention isn't enough to elicit it; a worked example would be needed.

## Efficiency — NOT a turn-saver

| Solver | avg iters/Q | median | %@cap |
|---|---|---|---|
| `rvlm` (n=8) | 13.0 | 11 | 1% |
| `rvlm_ocr_ablation` (n=8) | 12.0 | 10 | 1% |
| **`rvlm_vsearch`** | **~14.7** | ~12 | ~4% |

Counter to the "locate cheaply → fewer turns" intuition, vsearch takes
**slightly more** steps than `rvlm` (and pins the cap ~4×). `search` is
used as an *extra* exploration step, not a replacement for surveying;
its page-level, approximate results still require a `batch_look` to
verify, and on hard/unanswerable questions cheap re-search invites longer
exhaustive loops. (Lexical `rvlm_ocr` *did* shave turns, 12.0 < 13.0 —
visual search did not.)

## Design notes

- **Retrieval is page-level; the result is a page index, not an image or
  region.** Scoring is patch-level MaxSim over a page's full multi-vector
  set (all tiles) — any single strongly-matching patch surfaces the whole
  page. `search` returns `[{page, score}]`; the agent then reads
  `pages[i]` with `batch_look`. The matching tile's bbox is known
  internally but **not surfaced** — a future enhancement (return the
  region to crop) that could help on large dense pages.
- **Image tiling is automatic (processor-side).** ColModernVBERT's
  Idefics3-style processor resizes each page to longest-edge ≤ 2048 then
  splits into 512px tiles + a global thumbnail (≈9–17 sub-images/page).
  Large/dense pages are preserved as many patch vectors rather than one
  thumbnail — the late-interaction advantage. Ceiling: the 2048 longest-
  edge cap softens very-fine print on huge scans.

## The batch_look wedge (and fix)

**Symptom.** 3 of 6 trials hit a CPU-bound runaway (science_paper_1 ×2,
comics_4, comics_2 — all large multi-page docs): a question froze its log
while burning ~75% CPU for ~90 min, uncaught by the iteration cap. Each
was manually recovered by kill + relaunch (resumable; only the stuck doc
re-runs).

**Root cause.** On large docs the agent does big multi-page `batch_look`
surveys (10–15 pages). Under shared-server load (2 trials × c=24 = 48-way
on a single-GPU vllm, see below) some VLM calls hit the 600s litellm
timeout and retry 5× (~50 min each), and `batch_look`'s blocking
`ThreadPoolExecutor` shutdown waited on every straggler → the whole
question wedged.

**Fix (commit `0bd0f80`, scoped to `rvlm_vsearch_solver.py`):**
1. `batch_look` is now bounded by a wall-clock budget (120s/wave) with a
   non-blocking pool shutdown; unfinished slots return a
   `[VLM call timed out]` sentinel so the agent continues.
2. The perception sub-call is capped at `timeout=150 / num_retries=2`
   (was 600/5) so a stalled image fails fast.

Normal calls (<30s) never hit either limit, so scores stay comparable.
**Validated on t6:** relaunched under the fix, it cleared
science_paper_1 (the doc that wedged it at ~82% CPU) at ~12% CPU with 0
timeout-sentinels — no wedge.

> The same unbounded `batch_look` exists in `rvlm` / `rvlm_ocr_ablation`
> (they share the pattern); they've been lucky on comics so far.
> Propagating the bound to the shared helper is a recommended follow-up.

## Operational note

The 27B vllm (`qwen35-27b`) ran `--data-parallel-size 1`,
`CUDA_VISIBLE_DEVICES=0` — **single GPU**. GPU 0 pegged at ~98% serving
all 48 concurrent requests while GPUs 1–2 sat idle. This both slowed the
trials (~2 h each) and amplified the timeout-driven wedges. A DP=3
restart (~3× throughput, fewer timeouts) was recommended before resuming
t7/t8 — deferred to the user.
