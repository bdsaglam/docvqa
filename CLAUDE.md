# DocVQA 2026

ICDAR 2026 DocVQA competition. RLM agents with active document perception.

> **Paper framing pivot (2026-05-27, D-006).** The paper is now framed
> around a **visual context-budget hypothesis** — mid-sized open VLMs
> are perception-budget-bound, not reasoning-bound; recursive perception
> (RLM with VLM sub-call) is the fix. **Proposed method = OCR-free RLM**
> (currently `leanest_solo`). **OCR/search is an extension**, requires
> a new solver (clean fork of leanest, distinct from `flat_solo` which
> conflates OCR with a `look()` ergonomic wrapper). See
> `docs/paper/decisions.md` (D-006/D-007/D-008) and
> `docs/paper/README.md` for the full framing.
>
> Operating principles flowing from this:
> - **Prompt parity (D-007).** Every paper solver passes the same
>   prompt audit standard.
> - **Semantic-per-profile, tool-routing-per-solver (D-009, refines
>   D-007).** Tool-agnostic per-category content lives in the dataset
>   profile (`datasets/profile.py`); tool-routing lives in the solver
>   (`TASK_INSTRUCTIONS` + optional per-category overlay). All paper
>   solvers are dataset-aware by default — `solo`/`_da` pairs are
>   being merged.
> - **Trial-budget escalation (D-008).** New cells: n=1 → n=2 → n=8
>   only after the paper headline locks.
> - **No prompt-iteration narrative in the paper.** No v1/v2/scrub
>   history; no engineering solver names. Readers see the end-state.

## Two-host coordination convention

Two hosts share this repo:

- **amax7** = **adaptive** host. Runs critical-path experiments whose
  result might change the experiment plan. One cell at a time; replan
  after each result. Has Qwen 3.5 27B vllm at `localhost:8927`.
- **amax1** = **throughput** host. Runs side-track experiments where
  the direction is already known and we just need to lock numbers. No
  adaptive iteration; brings up its own per-model vllm containers as
  needed.

### File layout

```
coordination/
├── README.md       # full protocol details
├── amax7.md        # adaptive host's queue
└── amax1.md        # throughput host's queue
```

Each host file has three sections: `## In progress`, `## Queued`,
`## Done`. Each cell is one experiment (one solver × one split × n=1).

### Per-cell workflow

A "unit of work" is one experiment cell. For each cell:

1. **Pull latest:** `git pull --rebase`
2. **Pick** the first `[ ]` queued cell in your host file. Don't claim
   cells you can't start immediately — one in-progress cell per host
   at a time.
3. **Mark `[→]`** with an ISO timestamp (and tmux session if running
   in background).
4. **Commit + push** the host file. This advertises the lock so the
   other host doesn't accidentally duplicate.
5. **Run the experiment.**
6. **Mark `[✓]`** with the run_id and a one-line result, or **`[✗]`**
   with the failure mode.
7. **Commit + push** the updated host file + any new run artifacts to
   share (run dirs in `output/runs/` are gitignored; share via the
   one-line result, plus submission JSONs and experiment docs when
   applicable).
8. **Loop.**

### Which host runs what

- **Critical-path** (amax7): cells where a surprising result would
  trigger a paper-framing revision. Examples: the unified-tips ablation
  (could change the default method); the M+OCR clean cell (locks the
  paper's OCR-extension number); the direct_vlm cell (alt-architecture
  evidence for prediction 3).
- **Side-tracks** (amax1): cells where the direction is robust and we
  just need clean magnitudes. Examples: model-axis re-runs (Gemma E4B,
  Qwen 9B, Gemma 31B) on clean prompts — original n=3 data showed the
  lift; the re-runs just lock magnitudes.
- **If amax1 hits a surprise:** halt the queue and append a
  `## NOTE FOR AMAX7` section to `coordination/amax1.md`. amax7 reads
  on its next pull and decides whether to redirect.

### Conventions

- **Status legend:** `[ ]` queued, `[→]` in progress, `[✓]` done,
  `[✗]` failed, `[~]` deferred.
- **Trial budget per D-008:** cells default to n=1. If n=1 matches
  expectations, the queue owner files n=2 as a follow-up cell at file
  time. n=8 only after the paper headline locks.
- **Run IDs use new solver names** (`rvlm-*`, `rvlm-ocr-*`,
  `raw-vlm-multi-*`, etc.). Historical IDs (`leanest-*`, `flat-solo-*`)
  stay as they were; D-010 doesn't backfill.
- **Commit messages for coord changes:** `coord: <host> <action>
  <cell-name>`, e.g., `coord: amax1 done gemma-e4b-baseline-val-t1`.
- **Never claim or in-progress two cells at once on one host.** That
  invites push/pull conflicts on the host file.

### Orchestration: prefer heartbeat over chain scripts

**Don't write chain orchestrator scripts** (`scripts/<x>_chain.py`,
`scripts/<x>_post_orch.py` etc.) that pre-script "run trial N → wait
for sentinel → run trial N+1 → ...". Use a **heartbeat cron** that
polls run state and launches the next thing when its condition is
met. Concretely:

- A trial launch is one line: `tmux new-session -d -s <name> bash -c
  "cd /repo && exec uv run python evals.py ... run_id=<id>"`. There's
  no value in wrapping six of these in a script.
- The heartbeat checks state on each tick — `ls output/runs/<id>/
  tasks/*/result.json | wc -l`, `tmux ls`, `ls /tmp/<sentinel>` — and
  decides what to launch. Stateless and self-healing: if the
  heartbeat misses a tick or the previous launcher died, the next
  tick just sees the same state and recovers.

Why this matters:
- **Brittle.** Chain scripts are single processes that can silently
  die mid-chain (we lost a chain at 14:33 on 2026-05-30 this way).
  Heartbeats are session-resident and re-fire from clean state.
- **Decisions get hidden.** A chain that auto-launches the next
  trial after the prior lands removes the chance to look at the
  result first. amax7 is supposed to be the **adaptive** host —
  pre-committed chains undermine that.
- **Triggers are locked in.** A chain script with a 22/25 overlap
  trigger can't react to a stuck-at-18/25 long-tail trial. A
  heartbeat can apply any condition — overlap trigger, stuck-
  detection, GPU-idle, etc. — without rewriting the orchestrator.
- **Early launches.** Heartbeat can launch the next trial *before*
  the previous one fully completes (e.g., when the previous is at
  18/25 and stuck on a long-tail doc). A chain locked to "wait for
  sentinel" can't do that.

What's OK: single-purpose launcher scripts (`tmux new-session ...
evals.py ...`), the eval runner itself, utility scripts. Not
multi-step orchestrators.

## Best Results

| Config | Val | Test |
|--------|-----|------|
| Flat Solo SC-8 (lean, no-think, Qwen 3.6 27B / Qwen 3.6 27B) | **51.2%** | **43.75%** |
| Flat Solo SC-8 (lean, no-think, Qwen 3.5 27B / Qwen 3.5 27B) | 51.2% | 39.0% |
| Flat Solo single run (lean, no-think, Qwen 3.5) | 48.8% | 35.6% |
| Flat Batch (Pro+Flash) | 55.0% | — |
| Gemini 3 Pro baseline | 37.5% | 37.5% |

## Best Config Per Solver

| Solver | Config Override | Best Val |
|--------|----------------|----------|
| **Flat Solo** | `solver=flat_solo solver.rlm_type=lean` | **48.8%** |
| Leanest Solo | `solver=leanest_solo` | 43.8% |
| Lean Solo | `solver=lean_solo` | 42.5% |
| Flat Batch | `solver=flat_batch` | 37.5% |
| Ensemble | `solver=ensemble_lean_solo` | — |

**IMPORTANT**: `flat_solo` yaml default is `rlm_type=code` (~40%). ALWAYS override to `lean`.

## Infrastructure

- **LLM**: `vertex_ai/gemini-3-pro-preview` (best) or `qwen-3_5-27b` (local)
- **VLM**: Qwen/Qwen3.5-27B at localhost:8927 (3x A100 GPUs)
- **OCR data**: `data/docvqa-2026/{split}/ocr/{doc_id}/page_*.md` (new dataset layout: `data/{dataset-slug}/{split}/...`)
- **BM25 indexes**: auto-built per doc during eval

## Key Commands

```bash
# Best single-run solver
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=flat_solo solver.rlm_type=lean \
  data.split=val data.num_samples=null max_concurrency=16 run_id=flat-solo-val

# Ensemble (5x lean solo)
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local lm.enable_thinking=false \
  solver=ensemble_lean_solo data.split=val data.num_samples=null \
  max_concurrency=15 run_id=ens-val

# Generate report
python scripts/report.py --all --min-questions 80 --recent 7
```

## Key Findings

1. **Solo >> Batch**: ~10pp gap — one question at a time is much better
2. **Lean RLM > Code RLM** for solo: lean+nothink = 46.2%, code+think = 40.0%
3. **Thinking hurts lean**: 38.8% (think) vs 41.6% mean (nothink)
4. **High variance**: ~3-4% std across trials — always run 3+ trials
5. **Per-category tips** in `src/docvqa/prompts.py` help precision-heavy categories

## Cross-benchmark methodology rule (critical)

When reporting baseline-vs-scaffold on **any benchmark other than
DocVQA-2026**, the baseline must use a **dataset-aware profile** and
a **fair page budget**, or the scaffold lift double-counts prompt fit
+ truncation as scaffold capability.

Concretely:

- Use `*_da` solver variants (`solver=no_loop_multi_da`,
  `solver=flat_solo_da`, `solver=leanest_solo_da`). They pull from
  `docvqa.datasets.profile.get_profile(dataset)` for prompt, tips,
  per-question hint, and scorer.
- Pass `data.use_profile_scoring=true` so the runner uses the
  profile's `score_fn` (e.g. Qwen judge for MMLongBench) instead of
  ANLS.
- On long-doc benchmarks, override `solver.max_pages=80` (or the
  loader's `DEFAULT_MAX_PAGES`) so the raw-VLM baseline can see the
  evidence pages. The default `max_pages=10` is fine for short-doc
  benchmarks.

**Empirical evidence (2026-05-14, Qwen 3.5 27B, 200Q val samples):**

| Benchmark | Legacy lift | Fair lift (DA + pages) | Δ from baseline crippling |
|---|---|---|---|
| MP-DocVQA (ANLS) | −4.88pp (leanest "regresses") | **~0pp** | +5pp |
| MMLongBench-Doc (judge) | +26.43pp (leanest) | **+14.81pp** | +11.6pp |
| MMLongBench-Doc (judge) | +26.60pp (flat_solo) | **+16.84pp** | +9.8pp |

About half the MMLongBench legacy headline came from baseline
crippling (+5pp from max_pages=10→80, +8pp from DocVQA-2026 prompt →
MMLongBench profile). The MP-DocVQA legacy "regression" was 100%
prompt mismatch.

See `docs/experiments/mp-docvqa-qwen27b.md` and
`docs/experiments/mmlongbench-doc-qwen27b.md` for the full closed-loop
numbers and `src/docvqa/datasets/profile.py` for the registered
profiles.

## Project Structure

| File | Purpose |
|------|---------|
| `evals.py` | Hydra entry point |
| `src/docvqa/solvers/` | Solver implementations |
| `src/docvqa/solvers/*_da_solver.py` | Dataset-aware variants (profile-driven) |
| `src/docvqa/datasets/profile.py` | `DatasetProfile` + `get_profile(dataset_id)` |
| `src/docvqa/prompts.py` | DocVQA-2026 answer-formatting rules + per-category tips |
| `src/docvqa/data.py` | Dataset loading, OCR integration |
| `src/docvqa/runner.py` | Eval runner (concurrent, resumable; accepts profile `score_fn`) |
| `src/docvqa/metrics.py` | ANLS evaluation |
| `scripts/report.py` | Results report from run IDs |

## GCP Credits

- EDU Credit: ~41K/43.8K remaining (94%) — expires March 2027
- Gen App Builder: 41.5K — expires October 2026
