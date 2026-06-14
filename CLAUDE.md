# DocVQA 2026

ICDAR 2026 DocVQA competition. RLM agents with active document perception.

> **Paper framing pivot (2026-05-27, D-006).** The paper is framed
> around a **visual context-budget hypothesis** — mid-sized open VLMs
> are perception-budget-bound, not reasoning-bound; recursive perception
> (RLM with a VLM sub-call) is the fix. **Proposed method = OCR-free RLM**
> = the **`rvlm`** solver (REPL + recursive VLM `batch_look`, no OCR).
> **OCR/search is an extension** (`rvlm_ocr` / `rvlm_ocr_ablation`), kept
> clean and distinct from `rvlm_full`, which bundles OCR with a `look()`
> ergonomic wrapper. See `docs/paper/decisions.md` (esp. D-006/D-009/D-010)
> and `docs/paper/README.md` for the full framing; `docs/solvers/README.md`
> for the solver↔paper-role map.
>
> Operating principles flowing from this:
> - **Prompt parity (D-007).** Every paper solver passes the same
>   prompt audit standard.
> - **Semantic-per-profile, tool-routing-per-solver (D-009).** Tool-agnostic
>   per-category content lives in the dataset profile
>   (`datasets/profile.py`); tool-routing lives in the solver
>   (`TASK_INSTRUCTIONS` + optional per-category overlay). **All solvers
>   are dataset-aware by default** — the old `solo`/`_da` split is gone
>   (D-010); there are no more `*_da_solver.py` files.
> - **Trial-budget escalation (D-008).** New cells: n=1 → n=2 → n=8
>   only after the paper headline locks.
> - **No prompt-iteration narrative in the paper.** No v1/v2/scrub
>   history; no engineering solver names. Readers see the end-state.

## Two-host coordination convention

Two hosts share this repo: **amax7** and **amax1**. Both are used
dynamically — whichever host has free GPU takes the next experiment;
there's no fixed role split. Each owns one queue file; one in-progress
cell per host at a time. amax7 has Qwen 3.5 27B vllm at `localhost:8927`;
amax1 brings up its own per-model vllm containers as needed.

### File layout

```
coordination/
├── README.md       # full protocol details
├── amax7.md        # amax7's queue
└── amax1.md        # amax1's queue
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

### Cross-host escalation

If a host hits an unexpected result the other host should know about,
halt the queue and append a `## NOTE FOR <other-host>` section to its
own queue file (e.g. `## NOTE FOR AMAX7` in `coordination/amax1.md`).
The other host reads it on its next pull and decides whether to
redirect.

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
  result first — pre-committed chains undermine adaptive replanning.
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

## Results — where to look

**`docs/results.md`** is the single source of truth for cross-solver
numbers; **`docs/experiment-status.md`** for what's done / in progress /
queued; `docs/experiments/{solver}-{model}.md` for per-cell detail.
**`docs/pass-at-k.md`** holds the pass@k (oracle) / SC@k (self-consistency)
diagnostic per cell (script: `scripts/pass_at_k.py`; headline stays mean±std
per D-003). ⚠ The published `*-cmp-val` headline matrix has **no retained
per-trial artifacts** (deleted both hosts) → its pass@k/SC@k need a re-run.

> ⚠ **Numbers moved on 2026-06-01** (prompt scrub + per-call retry change).
> Pre-change numbers are archived under `archive/` and are **no longer
> valid** — don't cite them. The competition-submission headline
> (legacy `flat_solo` SC-8: val 51.2% / test 43.75% on Qwen 3.6 27B) lives
> in the public `README.md` and the submission report; it predates the
> current code/framing.

Current headline (Qwen 3.5 27B, val 25-doc/80-Q subset, `n=8`, current
code — three clean tiers, every gap ≫ the std):

| Tier | Solver | Val (n=8) |
|---|---|---|
| **proposed** | **`rvlm`** (REPL + recursive VLM `batch_look`, OCR-free) | **39.38% ± 1.49** |
| +OCR extension | `rvlm_ocr_ablation` | 37.81% ± 3.12 |
| no recursion | `react_baseline` / `direct_vlm` / `raw_vlm_multi_baseline` | 20–25% |
| OCR-only floor (no vision) | `rlm_ocr` | 13.91% ± 1.56 |
| competition anchor | `official_baseline` (MASTER_PROMPT, no scaffold) | 17.81% ± 1.86 |

Official ICDAR baselines (external): Gemini 3 Pro 37.5% test, GPT-5.2 35.0% test.

## Infrastructure

- **LLM / VLM configs**: `configs/{lm,vlm}/<model>-<provider>.yaml`. Headline
  runs use Qwen 3.5 27B for both. Model axis: `qwen-3_5-{4b,9b,27b}`,
  `qwen-3_6-{27b,35b}`, `gemma-4-{e4b,31b}`; closed: `gemini-3_1-pro`,
  `gemini-3-flash` (`-vertex` / `-studio` / `-openrouter`).
- **Local VLM server**: Qwen 3.5 27B vllm at `localhost:8927`. amax1 is the
  active host; brings up per-model containers as needed (always keep a 27B up).
- **OCR data**: `data/docvqa-2026/{split}/ocr/{doc_id}/page_*.md`
  (layout `data/{dataset-slug}/{split}/...`). BM25 indexes auto-built per doc.

## Key Commands

```bash
# Proposed method (rvlm), 27B homog, single trial. rvlm is the config default.
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false solver=rvlm \
  data.split=val data.num_samples=null max_concurrency=16 run_id=rvlm-val-t1
# swap solver= for a variant: codeact | rvlm_ocr_ablation | rvlm_hybrid_ablation |
#   rvlm_nocrop_ablation | rvlm_subagent_ablation | rvlm_subagent_full | rvlm_rationale |
#   react_baseline | raw_vlm_multi_baseline | direct_vlm | official_baseline | rlm_ocr
# cross-model: set lm=/vlm= to a config above (e.g. lm=qwen-3_5-9b-vllm-local vlm=qwen-3_5-27b-vllm-local)

# Reports
python scripts/report.py --all --min-questions 80 --recent 7   # results report
python scripts/iter_stats.py '<run_id_glob>'                    # per-run agent iterations
```

Concurrency: `c=16–24` on a healthy 27B; lower (`c=4–8`) for heavy/nested
solvers (`subagent_full`, `codeact` on long docs) and small/slow servers.

## Key Findings (current code; see `docs/results.md`)

1. **Three clean tiers**: visual-recursive (`rvlm` ~39%) ≫ no-recursion
   (`react`/`direct_vlm`/`raw_vlm_multi` 20–25%) ≫ OCR-only floor (`rlm_ocr` 14%).
2. **OCR-free is decisive**: swapping visual perception for OCR text
   (`rlm_ocr`) is the matrix floor, **−25.5pp** vs `rvlm`. Adding OCR *on top*
   of vision (`rvlm_ocr`) buys ≈ 0 on moderate-doc DocVQA; it pays off on
   long-doc benchmarks (the OCR extension's job).
3. **Both halves of the scaffold are load-bearing**: dropping the recursive
   sub-call (`raw_vlm_multi`) or the REPL (`react`) both collapse to the
   no-recursion tier.
4. **Enriching the sub-call doesn't help on DocVQA val** — generality
   (`subagent`), full agency (`subagent_full`), rationale channel
   (`rvlm_rationale`) are all ≈ parity; the minimal `batch_look` is sufficient.
5. **Perception-budget-bound**: fixing the reasoner and swapping only the VLM
   →27B lifts ~8pp at 9B/4B — the signature of a perception (not reasoning)
   bottleneck (supports D-006).
6. **High variance**: ~3pp std across trials — always run ≥3 (headline n=8).
7. **Per-category tips** live in `src/docvqa/prompts.py` + `datasets/profile.py`.

## Cross-benchmark methodology rule (critical)

When reporting baseline-vs-scaffold on **any benchmark other than
DocVQA-2026**, the baseline must use a **dataset-aware profile** and
a **fair page budget**, or the scaffold lift double-counts prompt fit
+ truncation as scaffold capability.

Concretely:

- All solvers are dataset-aware by default — pass `data.dataset=<id>` and
  the right solver (`solver=rvlm`, `rvlm_ocr_ablation`,
  `raw_vlm_multi_baseline`, …). They pull from
  `docvqa.datasets.profile.get_profile(dataset)` for prompt, tips,
  per-question hint, and scorer. (The old `*_da` variants no longer exist —
  D-010 merged them.)
- Pass `data.use_profile_scoring=true` so the runner uses the profile's
  `score_fn` (e.g. Qwen judge for MMLongBench) instead of ANLS.
- On long-doc benchmarks, raise the page budget (`solver.max_pages`, or the
  loader's `DEFAULT_MAX_PAGES`) so the raw-VLM baseline can see the evidence
  pages. The default is fine for short-doc benchmarks.

> ⚠ **The earlier MP-DocVQA / MMLongBench-Doc numbers are invalid** — they
> were run on pre-2026-06-01 prompts/retry logic (archived under
> `archive/experiments/`). The *mechanism* (lift sign + magnitude scale with
> the benchmark's page-count distribution) is robust, but the magnitudes need
> a current-code re-run before citing. See `docs/results.md`
> ("Document-length axis") — **pending**.

Registered profiles: `src/docvqa/datasets/profile.py`
(+ `mmlongbench_doc.py`, `mp_docvqa.py`).

## Project Structure

| File | Purpose |
|------|---------|
| `evals.py` | Hydra entry point (defaults: `solver=rvlm`, Qwen 3.5 27B lm+vlm) |
| `src/docvqa/solvers/` | Solver implementations (all dataset-aware; see `docs/solvers/README.md`) |
| `src/docvqa/rlm/` | LeanRLM / CodeRLM scaffold + subprocess REPL interpreter |
| `src/docvqa/datasets/profile.py` | `DatasetProfile` + `get_profile(dataset_id)` |
| `src/docvqa/prompts.py` | DocVQA-2026 answer-formatting rules + per-category tips |
| `src/docvqa/data.py` | Dataset loading, OCR integration |
| `src/docvqa/search.py` | BM25 index (bm25s) |
| `src/docvqa/judges/` | LLM judges (e.g. Qwen judge for MMLongBench scoring) |
| `src/docvqa/runner.py` | Eval runner (concurrent, resumable; accepts profile `score_fn`) |
| `src/docvqa/metrics.py` | ANLS evaluation |
| `scripts/report.py` | Results report from run IDs |
| `scripts/iter_stats.py` | Per-run agent-iteration stats |

## GCP Credits

- EDU Credit: ~41K/43.8K remaining (94%) — expires March 2027
- Gen App Builder: 41.5K — expires October 2026
