# Handover — second-benchmark evaluations (MP-DocVQA + MMLongBench-Doc)

**Created:** 2026-05-11
**Updated:** 2026-05-11 23:50 — refreshed for resume-from-here.
**Context:** Continuing the §B "benchmark generality" claim from
`docs/paper/experiment-plan.md`. Picks: MP-DocVQA + MMLongBench-Doc
(see "Picks rationale" section at bottom for why these two).

## Resume-from-here checklist (top-of-mind for next agent)

1. **Finish MP-DocVQA prep** (loader exists, sample file + README missing
   — see "MP-DocVQA prep status" section). ~10-min Python script.
2. **Wire dataset dispatch** in `src/docvqa/data.py` `load_documents()`
   so Hydra `data.dataset=lmms-lab/MP-DocVQA` and
   `data.dataset=yubo2333/MMLongBench-Doc` route to the new loaders
   in `src/docvqa/datasets/`. The DocVQA-2026 default path stays
   unchanged. ~30 lines.
3. **Calibrate Qwen judge** on ~10 hand-marked MMLongBench-Doc triples
   before running cells (≤30 min). If agreement <70%, iterate the
   prompt in `src/docvqa/judges/qwen_judge.py` first.
4. **Run cells** — see "Eval execution plan" section. 4 cells × n=3 =
   12 trials. Use both endpoints in parallel (~11h total wall).
5. **Write per-dataset experiment files** as cells complete:
   `docs/experiments/mp-docvqa-qwen27b.md` and
   `docs/experiments/mmlongbench-doc-qwen27b.md`. Template = any
   existing `docs/experiments/*-baseline-scaffold.md`.
6. **Update §B in `docs/paper/experiment-plan.md`** + index entries
   in `docs/experiments/README.md` after both datasets done.

## Current state at resume

- **vllm 8927 (local) — UP** as of 2026-05-11 23:29:55:
  `Qwen/Qwen3.5-27B`, max_model_len 131072, DP=4 across all 4 GPUs
  (prefix caching, async-scheduling, reasoning-parser qwen3).
  Container `vllm-qwen-27b`, tmux `vllm:qwen27b`. **Verified idle**
  — no eval processes running.
- **vllm 8928 (other host) — UP:** `Qwen/Qwen3.5-27B`, max_model_len
  131072, responding cleanly to `/v1/models`. Ready to use.
- Two endpoints → run two trials in parallel, each pinned to one
  endpoint via Hydra config (no code change needed).
- **Configs already exist** for both endpoints:
  - `configs/lm/qwen-3_5-27b-vllm-local.yaml` → port 8927
  - `configs/lm/qwen-3_5-27b-vllm-remote.yaml` → port 8928
  - Same names under `configs/vlm/`.
- **No agents currently running.** The two prep sub-agents both
  returned earlier today: MMLongBench-Doc agent finished its full
  brief; MP-DocVQA agent returned partial deliverables (loader file
  written, but sample file + README skipped).
- **Most recent commit:** `e35d0d5` (this handover doc + loaders +
  judge + data layout migration).

## Decisions already made (do not re-litigate)

- **Sample size:** 200 questions per dataset, fixed seed=42. Stratified
  sampling where applicable. MMLongBench-Doc sample is committed at
  `data/mmlongbench-doc/val/sample_200q_doc_ids.txt` (22 docs, 207 Qs,
  all 5 answer formats represented).
- **Solvers per dataset:** `no_loop_multi` (matched raw-VLM baseline)
  + `leanest_solo` (OCR-free agent loop with VLM `look()` tool). No
  OCR is required for either solver — MP-DocVQA and MMLongBench-Doc
  do not need pre-extracted OCR.
- **Trials per cell:** n=3 (plan's headline floor). Two solvers × two
  datasets × n=3 = 12 trials total. Estimated wall ~22h on one
  endpoint, ~11h with both endpoints used in parallel.
- **MMLongBench-Doc metric:** Qwen 27B as local judge (no GPT-4o
  budget). Judge is at `src/docvqa/judges/qwen_judge.py`. Prompt is
  adapted from MMLongBench-Doc's official `prompt_for_answer_extraction.md`
  + `eval/eval_score.py` rules. **Calibration step recommended before
  trusting numbers** — run the judge on ~10 hand-marked
  (question, gt, pred) triples to verify it matches official
  human/GPT-4o judgements within reasonable agreement.
- **Per-dataset experiment file** at `docs/experiments/{dataset-slug}-qwen27b.md`
  with the same structure as the existing files (see
  `docs/experiments/qwen-9b-baseline-scaffold.md` for the template).
- **On prep failure:** push through with whichever dataset works.
- **Failure isolation:** if MMLongBench-Doc judge produces flaky
  results, fall back to ANLS-as-proxy with explicit caveat in the
  experiment file. Do not block on judge issues.

## Data folder convention (newly established, layout-migration COMPLETE)

```
data/
├── docvqa-2026/                      # moved here from data/{val,test}/ on 2026-05-11
│   ├── val/
│   │   └── ocr/{doc_id}/page_*.md
│   └── test/
├── mp-docvqa/
│   ├── val/
│   │   └── sample_200q_doc_ids.txt   # ⚠ NOT YET CREATED (see "MP-DocVQA prep status" below)
│   └── README.md                     # ⚠ NOT YET CREATED
└── mmlongbench-doc/
    ├── val/
    │   ├── sample_200q_doc_ids.txt   # ✓ 22 docs / 207 Qs
    │   └── pages/{doc_id}/page_{i}.png   # ✓ cached PDF→PNG renders
    └── README.md                     # ✓ written
```

**Code/docs updated for the new layout (commit before running evals):**
- `src/docvqa/data.py` — `_ocr_dir_for_split` now returns `data/docvqa-2026/{split}/ocr`
- `src/docvqa/search.py` — `DEFAULT_BM25_DIR = data/docvqa-2026/val/bm25`
- `scripts/run_ocr.py` — default `ocr_dir = data/docvqa-2026/{split}/ocr`
- `tests/test_search.py` — `OCR_DIR = data/docvqa-2026/val/ocr`
- `CLAUDE.md` — path mention updated
- `README.md` — both OCR/BM25 path mentions updated
- `docs/dataset.md` — OCR path mention updated

All new datasets go under `data/{dataset-slug}/{split}/`. The pattern
`data/{dataset-slug}` is the new convention; the layout migration is
documented inline in `src/docvqa/data.py:_ocr_dir_for_split` docstring.

## Loader interface

Every dataset loader exposes `load_<dataset>_documents(split, num_samples,
doc_ids, ocr_dir=None) -> list[Document]`. `Document` and `Question`
dataclasses are in `src/docvqa/data.py:14-31` — do not modify those
unless forced. Loaders go under `src/docvqa/datasets/{name}.py`.

The runner's existing entry point `evals.py` calls
`docvqa.data.load_documents(dataset_name, split, ...)` which currently
hardcodes the DocVQA-2026 schema. **You will need to add a dispatch**
in `load_documents` (or in evals.py) that routes by `dataset_name`:

```python
# Pseudo-code for the dispatch
if dataset_name == "lmms-lab/MP-DocVQA":
    from docvqa.datasets.mp_docvqa import load_mp_docvqa_documents
    return load_mp_docvqa_documents(split, num_samples, doc_ids, ocr_dir)
elif dataset_name == "yubo2333/MMLongBench-Doc":
    from docvqa.datasets.mmlongbench_doc import load_mmlongbench_doc_documents
    return load_mmlongbench_doc_documents(split, num_samples, doc_ids, ocr_dir)
else:  # default DocVQA-2026 path
    ...
```

Then the existing eval invocation pattern works unchanged:

```bash
uv run python evals.py \
  data.dataset=yubo2333/MMLongBench-Doc \
  data.split=val \
  'data.doc_ids=[<paste from sample file>]' \
  lm=qwen-3_5-27b-vllm-local \
  vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=leanest_solo \
  max_concurrency=16 \
  run_id=leanest-solo-mmlb-val-t1
```

For 200-Q subsetting, prefer passing `data.doc_ids=[...]` from the
sample file rather than `num_samples` — gives reproducibility and
matches what's in the sample text file.

## MMLongBench-Doc prep — DONE (details from sub-agent report)

**Files created:**
- `src/docvqa/datasets/mmlongbench_doc.py` — loader. Function:
  `load_mmlongbench_doc_documents(split, num_samples, doc_ids, ocr_dir=None, *, dpi=150, max_pages=80, pages_dir=None)`.
- `src/docvqa/judges/qwen_judge.py` — Qwen judge. Function:
  `qwen_judge(question, ground_truth, prediction, answer_format) -> (is_correct, judge_response)`.
  Defaults to `http://localhost:8928/v1` and `Qwen/Qwen3.5-27B`,
  overridable via env (`QWEN_JUDGE_BASE_URL/MODEL/API_KEY`) or kwargs.
- `data/mmlongbench-doc/val/sample_200q_doc_ids.txt` — 22 docs, 207 Qs,
  stratified across `{Int=23.19%, Float=22.22%, Str=19.81%, None=18.84%, List=15.94%}`.
- `data/mmlongbench-doc/val/pages/{doc_id}/page_{i}.png` — rendered PDF
  caches at 150 DPI (warning: hundreds of MB at sample size, several
  GB at full dataset).
- `data/mmlongbench-doc/README.md` — full schema notes.

**Implementation notes from the sub-agent:**
- Pulls 1091-row HF parquet, groups by `doc_id`, downloads each PDF
  from `documents/<doc_id>` via `huggingface_hub`.
- Renders each PDF once with `pypdfium2` at 150 DPI (cap 80 pages).
- `Document` objects produced with each `Question` getting an extra
  `mmlb` attribute (`MMLBQuestionMeta` with `answer_format`,
  `evidence_pages`, `evidence_sources`, `doc_type`).
- `question_id` is synthesized as `<doc_id>::<first 60 chars of question>`
  (HF dataset has no native qid).
- Loader auto-uses `data/mmlongbench-doc/{split}/sample_200q_doc_ids.txt`
  if present.
- New dep added: `pypdfium2==5.8.0` (chosen over `pdf2image` to avoid
  poppler system dep).

**Judge contract (Qwen 27B):**
```
Extracted answer: <normalized>
Answer format: <Integer|Float|String|List|Not answerable>
Verdict: <correct|incorrect>
```

Single-call adaptation of the official two-stage extraction+score
protocol. Folds in deterministic scoring rules from
`eval/eval_score.py` (Int exact, Float ~1% tolerance with %-equivalence,
Str/None ANLS, List set/order rules, "Not answerable" handling).

**Open issues flagged by sub-agent:**
1. **Judge calibration not done.** Run ~10 hand-marked triples first.
2. **Page cap (80) may exclude evidence pages** for 150-200pp docs.
   For unbiased reporting, do one ceiling pass at `max_pages=200`.
3. **Pyright diagnostics** in `mmlongbench_doc.py` (cosmetic, not
   blocking):
   - `line 33`: unused `import os`.
   - `line 132`: `dpi: float` passed to `render(scale=...)` which
     expects int. Cast to int.
   - `line 193, 267, 268`: HF row indexing typed as Sized — pyright
     thinks `row["key"]` is a slice. Need `# type: ignore[index]` or a
     cast.
   - `line 253`: PIL `ImageFile` not assignable to `list[Image]`.
     Switch annotation to `list[ImageFile.ImageFile]` or cast.
   None of these break runtime — fix at your discretion.

## MP-DocVQA prep status — PARTIAL (loader exists, sample + README missing)

The sub-agent (`a6184cdc6680fd858`) returned but did NOT complete the
deliverables. Current state:

- ✓ `src/docvqa/datasets/mp_docvqa.py` exists (268 lines). Function:
  `load_mp_docvqa_documents(split, num_samples, doc_ids, ocr_dir=None)`.
  The loader auto-reads `data/mp-docvqa/{split}/sample_200q_doc_ids.txt`
  via `_split_doc_ids_file(split)` (lines 103-106 of the loader).
- ✗ `data/mp-docvqa/val/sample_200q_doc_ids.txt` — NOT WRITTEN. The
  directory `data/mp-docvqa/` doesn't even exist.
- ✗ `data/mp-docvqa/README.md` — NOT WRITTEN.
- Pyright cosmetics on `mp_docvqa.py` (lines 167, 173, 244, 248):
  HF row indexing typed as Sized — cast or `# type: ignore[index]`.

**Action for the new agent:** complete prep by running this snippet
(or have a sub-agent do it):

```python
# Quick fix: load MP-DocVQA val, sample ~25-30 docs accumulating ~200 Qs,
# write sample_200q_doc_ids.txt
import json, random
from datasets import load_dataset

random.seed(42)
ds = load_dataset("lmms-lab/MP-DocVQA", split="val")
# Inspect: likely one row per (doc, question) or (doc, qlist).
# Group by doc_id, count questions per doc, sample docs greedily
# until cumulative questions ≥ 200.
# ... (see existing mmlongbench_doc.py sampling logic for a template)
```

Then write a short `data/mp-docvqa/README.md` (~30 lines) covering:
source (lmms-lab/MP-DocVQA), splits + answer availability, schema
brief, sample stats (#docs, #qs), license.

## Eval execution plan

Execute one dataset at a time, ideally splitting trials between the
two endpoints (8927 local, 8928 remote) for ~2× wall speedup:

1. **Wire dataset dispatch** in `data.load_documents` (see §"Loader
   interface" above) to support both new datasets.
2. **MP-DocVQA cells** (after dataset prep is in):
   - Trial 1 of each solver on 8927; trial 2 on 8928; trial 3 on
     whichever finishes first (or sequential).
   - Run `no_loop_multi` first (faster, ~30-45 min/trial @ 200Q).
   - Then `leanest_solo` (~3 hours/trial @ 200Q).
   - Write `docs/experiments/mp-docvqa-qwen27b.md` after the first
     trial of each solver completes (template = qwen-9b-baseline-scaffold.md).
3. **MMLongBench-Doc cells** (same structure):
   - First call the judge on ~10 hand-marked triples to calibrate.
     If agreement <70%, fix the judge prompt before running cells.
   - Run baseline + scaffold same as above.
   - Use `qwen_judge` for scoring.
   - Write `docs/experiments/mmlongbench-doc-qwen27b.md`.
4. **After each cell completes**: send a notification with
   `$HOME/dotfiles/tools/notify.sh "$NTFY_TOPIC" "DocVQA cell done: <name>" "<mean ± std, lift, sandbox-error count>"`.
5. **Update §B in `docs/paper/experiment-plan.md`** with the new
   numbers when both datasets are done. Also add experiment-file
   entries to `docs/experiments/README.md`.

Recommended Hydra invocation template:

```bash
# baseline (no_loop_multi+tips), 1 trial
uv run python evals.py \
  data.dataset=yubo2333/MMLongBench-Doc \
  data.split=val \
  data.num_samples=null \
  'data.doc_ids_file=data/mmlongbench-doc/val/sample_200q_doc_ids.txt' \
  lm=qwen-3_5-27b-vllm-local \
  vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=no_loop_multi \
  max_concurrency=16 \
  run_id=no-loop-multi-tips-mmlb-val-t1
```

(`data.doc_ids_file` doesn't exist yet — add it to the Hydra data
config so it's loaded in addition to / instead of `data.doc_ids` list.
Or just inline the doc_ids list in Hydra `'data.doc_ids=[...]'`.)

## Picks rationale (for context)

From the conversation that set this up:
- Filtered out InfographicVQA / SlideVQA / DocVQA-original because
  DocVQA-2026 reuses their data.
- ChartQA: highest-cited remaining (1432) but adjacent (no chart
  category in DocVQA-2026 means generality isn't tested cleanly, and
  OCR-centric scaffold may underperform).
- **MP-DocVQA** (Pattern Recognition 2023, Tier-B established): direct
  ANLS metric match, clean baseline anchor (Hi-VT5), multi-page,
  industry doc pool independent of DocVQA-2026.
- **MMLongBench-Doc** (NeurIPS 2024 D&B Track, Tier-C modern): top-tier
  venue, multi-page (47pp avg), unanswerable subset matches our
  `Unknown` rule.

## Notifications

```bash
$HOME/dotfiles/tools/notify.sh "$NTFY_TOPIC" "<title>" "<summary>"
# NTFY_TOPIC env var is set in the user's shell.
```

Send on:
- Each cell completion (with mean, std, lift, sandbox errors)
- Critical blocker (vllm crash you can't recover from, dataset prep
  failure, judge calibration failure)
- Final wrap-up

## File / task pointers

- Spec / numbers history: `docs/paper/experiment-plan.md`
- Existing experiment-file template: `docs/experiments/qwen-9b-baseline-scaffold.md`
- Memory entries: `~/.claude/projects/-home-baris-repos-docvqa/memory/`
  — relevant: `feedback_scaffold_lift_scales_with_model_size.md`
- Prior chain scripts: `scripts/run_qwen9b_chain.sh`,
  `scripts/run_gemma_chain.sh`, `scripts/run_scaffold_chain.sh` —
  reuse pattern when writing chain script for the new datasets.

## User notes

- Only notify on important events: cell completions (with mean ± std,
  lift, sandbox-error count), blockers you can't resolve, final wrap-up.
  Not routine progress.
- User said "go autonomous, use your judgement and continue rather than
  wait, unless you are absolutely stuck."
- On dataset-prep failure: push through with whichever dataset works
  rather than blocking.
- Cell results live in `output/runs/<run_id>/results.json`. Use the
  existing report scripts in `scripts/report.py` to summarize.

## Tasks (cleaned up at resume)

The session-local task tracker is empty by design — recreate as you go.
Suggested initial set:

1. Finish MP-DocVQA prep (sample file + README).
2. Wire `data.load_documents` dispatch.
3. Calibrate Qwen judge.
4. Run MP-DocVQA × 2 solvers × n=3.
5. Run MMLongBench-Doc × 2 solvers × n=3.
6. Update §B + experiments index, commit, notify wrap-up.
