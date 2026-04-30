# GEPA Optimization for Flat-Solo DocVQA — Design

**Date:** 2026-04-29
**Status:** Draft (awaiting user review)
**Goal:** Raise the test-set ANLS score above 43.75% by automatically optimizing the prompts that drive the production `flat_solo` solver, using GEPA's `optimize_anything`.

## Why now

`flat_solo` (Qwen 3.5/3.6 27B / Qwen 3.5/3.6 27B, lean RLM, no thinking) is the strongest pipeline we've shipped: 51.2% val and 43.75% test with SC-8 voting. The prompts that drive it have been hand-tuned over several iterations (`prompts.py:V1_ORIGINAL → V2_LEAN → V3` plus a bespoke `TASK_INSTRUCTIONS` in `flat_solo_solver.py`, plus 8 per-category tip blocks in `prompts.CATEGORY_TIPS`). A previous attempt at GEPA (`archive/scripts/optimize_flat_batch.py`) targeted the weaker `flat_batch` solver, used Gemini Pro as both student and reflection model, and only allowed 50 metric calls. It did not produce a usable artifact.

This run targets the right solver, uses the cheap student (local Qwen) with an expensive teacher (Pro reflection), and gives GEPA enough budget to round-robin through every component multiple times.

## Optimization scope (Scope A — focused)

Nine optimizable text components:

1. `task_instructions` — main RLM agent prompt (~3 KB, currently `TASK_INSTRUCTIONS` in `flat_solo_solver.py:37`).
2. `tip_business_report`, `tip_comics`, `tip_engineering_drawing`, `tip_infographics`, `tip_maps`, `tip_science_paper`, `tip_science_poster`, `tip_slide` — the 8 entries from `prompts.CATEGORY_TIPS`, each ~600–1200 chars.

**Explicitly out of scope:** `ANSWER_FORMATTING_RULES` (hand-tuned, scoring strict, mutating risks regressions), the VLM `look()` prompt (small leverage on Qwen), and `LeanRLM.DEFAULT_ACTION_INSTRUCTIONS` (REPL mechanics — easy to break the agent if mutated).

## Models & infrastructure

| Role | Model | Endpoint |
|------|-------|----------|
| Student LLM (RLM) | `hosted_vllm/Qwen/Qwen3.5-27B` | `localhost:8927/v1` (2× A100 GPUs, tensor-parallel) |
| Student VLM (`look`/`batch_look`) | `hosted_vllm/Qwen/Qwen3.5-27B` | same endpoint |
| Reflection LM | `vertex_ai/gemini-3_1-pro-preview` | Vertex AI |

The 3rd A100 hosts Qwen 3.6 27B for unrelated workloads — it stays untouched.

## Train / val split

The dataset has only `val` (25 docs / 80 questions) and `test` (no public ground truth). We split val:

- **Train (17 docs, ~55 questions)** — used by GEPA's reflection minibatch sampler.
- **Held-out val (8 docs, ~25 questions)** — used by GEPA's Pareto frontier and final candidate ranking. **Stratified: exactly 1 doc per category.**

Selection is deterministic given a seed (default `42`). The val docs are pinned in the optimization script for reproducibility (the same 8 doc IDs every run).

After optimization, the top-3 candidates are re-evaluated on the **full 25-doc val** before going to test.

## Component architecture

### Files created

```
src/docvqa/solvers/flat_solo_gepa_solver.py    # candidate-injectable copy of flat_solo_solver
configs/solver/flat_solo_gepa.yaml             # Hydra config that takes candidate_path
scripts/optimize_flat_solo.py                  # GEPA driver (replaces archive version)
```

### Files NOT modified

`src/docvqa/solvers/flat_solo_solver.py` is left untouched — it remains the frozen "official" competition solver. All GEPA-related work (optimization, candidate evaluation, test submission) goes through `flat_solo_gepa_solver.py`.

### How the seed candidate equals production

The smoke test "Solver equivalence" (below) confirms that `flat_solo_gepa_solver` with the seed candidate (= current `TASK_INSTRUCTIONS` + `CATEGORY_TIPS`) produces identical answers to `flat_solo_solver`. This makes `flat_solo_gepa_solver` a strict superset: same behavior on the seed, configurable on optimized candidates.

### `flat_solo_gepa_solver.py` — diff from `flat_solo_solver.py`

1. `FlatSoloGepaProgram.__init__` adds three required parameters:
   - `task_instructions: str` (no fallback to module constant — caller must pass).
   - `tips_overrides: dict[str, str]` (per-category tips; empty dict = no tips).
   - `vlm_prompt: str | None = None` (forward-compat hook; unused in Scope A).
2. `solve_document(doc, precomputed=None)` adds a `precomputed` keyword:
   - `precomputed = {"page_dir": str, "search_index": SearchIndex | None, "page_texts_formatted": list[str]}` lets the GEPA evaluator skip the per-call `tempfile.TemporaryDirectory()` + image-saving + BM25-build (those are done once during dataset prep). When `precomputed=None`, falls back to the original behavior.
3. Uses `self.task_instructions` and `self.tips_overrides.get(cat, "")` directly — no `from docvqa.prompts import` at module scope for these.
4. No factory function, no Hydra config — only the optimization script imports it.

Everything else (tools, sandbox code, RLM construction, retry logic, scoring) is identical to `flat_solo_solver.py`.

### `scripts/optimize_flat_solo.py` — sections

1. **Setup** — load `.env`, init observability, `litellm.drop_params=True`, `request_timeout=300`.
2. **Dataset prep** — load val docs with ground truth via `load_documents("VLR-CVC/DocVQA-2026", "val")`. Stratified split into train (17 docs) and val (8 docs, 1 per category, deterministic by `doc_id`). For each doc, pre-compute `{document, page_dir, search_index, page_texts_formatted}` once: save page PNGs to a per-doc tempdir and build the BM25 index. These survive the entire optimization run.
3. **Seed candidate** — read `TASK_INSTRUCTIONS` from `flat_solo_solver.py` and the 8 entries from `prompts.CATEGORY_TIPS`. Build:
   ```python
   seed = {
       "task_instructions": TASK_INSTRUCTIONS,
       "tip_business_report": CATEGORY_TIPS["business_report"],
       # ... 7 more
   }
   ```
4. **Evaluator** — see "Evaluator" section below.
5. **GEPA config** — see "GEPA configuration" section below.
6. **Run + save** — call `optimize_anything(...)`. Save `best_candidate.json` to `run_dir`. Print top-3 candidates with their val scores.

### `configs/solver/flat_solo_gepa.yaml`

Hydra config to evaluate any candidate (seed or optimized) via `evals.py`:

```yaml
_target_: docvqa.solvers.flat_solo_gepa_solver.create_flat_solo_gepa_program
candidate_path: null      # null = use seed (current TASK_INSTRUCTIONS + CATEGORY_TIPS)
max_iterations: 30
vlm: ${vlm}
rlm_type: lean
page_factor: 1.5
question_concurrency: 4
```

Usage:
```
# Identity check (should match flat_solo)
uv run python evals.py solver=flat_solo_gepa data.split=val data.num_samples=null run_id=gepa-seed-id

# Apply optimized candidate
uv run python evals.py solver=flat_solo_gepa \
  solver.candidate_path=output/gepa-flat-solo/<run_id>/best_candidate.json \
  data.split=test data.num_samples=null run_id=gepa-best-test
```

## Evaluator

```python
def evaluate(candidate: dict[str, str], example: dict) -> tuple[float, dict]:
    # example = {"document": Document, "page_dir": str, "search_index": ..., "doc_id": str}
    # candidate = {"task_instructions": str, "tip_<category>": str, ...}
```

### Per-call work

1. Build `tips_overrides = {cat: candidate[f"tip_{cat}"] for cat in CATEGORIES}`.
2. Construct `FlatSoloGepaProgram(vlm_lm, task_instructions=candidate["task_instructions"], tips_overrides=tips_overrides, max_iterations=30, rlm_type="lean", question_concurrency=4)` — same defaults as production flat-solo.
3. Call `program.solve_document(doc, precomputed=...)` → `predictions, trajectories`.
4. Score: `accuracy = correct_count / scored_count` via `evaluate_prediction()`. ANLS via `get_anls()` for continuous tagging.
5. Return `(accuracy, side_info)`.

### ASI structure (the gradient — most important detail)

```python
{
    "scores": {"accuracy": 0.50},          # Pareto frontier metric
    "Feedback": (
        "Document: maps_3 (category: maps, 1 page, 4 questions)\n"
        "Score: 2/4 (50%)\n\n"
        "  CORRECT: Q='How many highways enter Rome?' -> '7'\n"
        "  WRONG: Q='What is the population of grid C-3?' "
            "predicted='Unknown' expected='234,500' — gave up (answer exists)\n"
        "    Last reasoning: \"Couldn't find C-3 in legend, marking unknown\"\n"
        "  CLOSE: Q='What road type connects A-2 to B-4?' "
            "predicted='secondary' expected='Secondary' (ANLS=0.89) — case mismatch\n"
        "  WRONG: Q='Which monument is south of Pantheon?' "
            "predicted='Colosseum' expected='Circus Maximus' (ANLS=0.18)\n"
    ),
    "tip_maps_specific_info": {
        "Feedback": "Category: maps\nScore: 2/4 with current tips\n<same per-question lines>"
    }
}
```

Three deliberate ASI choices:

1. **Per-component focused feedback.** Only the `tip_<category>` matching the doc's category gets a `tip_<category>_specific_info`. So when GEPA's round-robin selector reflects on `tip_maps`, it sees only maps-doc results — tight signal. When it reflects on `task_instructions`, it sees the global `Feedback` (cross-category patterns).
2. **Failure tagging.** Each wrong answer is tagged `gave up` / `close (case/format)` / `wrong (semantic)`. Helps reflection LM decide what to fix (formatting rule vs. retrieval strategy vs. category-specific knowledge).
3. **Last-reasoning snippet (failures only).** For wrong answers we include the last 200–300 chars of the agent's final reasoning before SUBMIT. CORRECT trajectories are omitted to save reflection LM context.

## GEPA configuration

```python
GEPAConfig(
    engine=EngineConfig(
        max_metric_calls=150,
        run_dir="output/gepa-flat-solo/<run_id>",
        seed=42,
        parallel=True,
        max_workers=8,                 # GEPA-level: 8 docs evaluated in parallel
        display_progress_bar=True,
        use_cloudpickle=True,
        cache_evaluation=True,         # disk cache, since run_dir is set
        raise_on_exception=False,      # don't kill the run on a single doc crash
    ),
    reflection=ReflectionConfig(
        reflection_lm="vertex_ai/gemini-3_1-pro-preview",
        reflection_minibatch_size=3,
        module_selector="round_robin", # rotates through the 9 components
    ),
    tracking=TrackingConfig(
        use_wandb=True,
        wandb_init_kwargs={"project": "docvqa", "entity": "bdsaglam"},
    ),
)
```

### Concurrency

- **GEPA `max_workers=8`** × **flat-solo `question_concurrency=4`** ⇒ up to ~24 in-flight questions (most docs have 2–4 questions each).
- Realistic Qwen load: ~16–24 concurrent question-streams. Production has run at `max_concurrency=16` successfully against the same vLLM endpoint.
- Fallback if we see vLLM overload / 429s: drop GEPA `max_workers` to 4.

### Caching

`cache_evaluation=True` with disk storage at `run_dir`. Critical because GEPA frequently re-evaluates `(seed_candidate, doc)` pairs across iterations. GEPA uses per-key locking, so concurrent writes on the same key are safe.

### Stopping

- `max_metric_calls=150` (primary).
- `<run_dir>/gepa.stop` file — manual kill switch.
- Custom callback: optional, not used in v1.

### Budget math

- Initial seed eval over train (17 docs) ≈ 17 metric calls. Plus initial val eval over 8 docs ≈ 8 metric calls.
- Per iteration: minibatch eval (3 docs) + occasional full val eval (8 docs) + 1 reflection LM call.
- 150 total ⇒ ~30 iterations after seed. With round-robin across 9 components, that's ~3 cycles per component.

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Reflection LM proposes a `task_instructions` that drops `SUBMIT(...)` syntax | Pareto frontier preserves prior best; ASI surfaces 0/N scores; round-robin moves on |
| Qwen vLLM crash mid-run | `raise_on_exception=False` keeps run alive; resume by restarting vLLM and rerunning with same `run_id` (disk cache preserves prior results) |
| Variance hides real gains (~3% std on single trial) | Pareto across 8 val docs preserves any wins; re-eval top-3 on full 25-doc val before test |
| Pro 429 rate-limits | ~30 reflection calls over 10h is well under Pro limits; if hit, fallback to `gemini-3-flash-preview` for reflection (documented quality drop) |
| Subprocess REPL leaks | Existing `_interpreter_context` has `try/finally: repl.shutdown()` (`lean.py:290`) — verified |
| Cache hash sensitivity | Never mutate candidate dict inside evaluator; always read |

## Smoke tests before the long run

1. **Solver equivalence** — confirm `FlatSoloGepaProgram(task_instructions=TASK_INSTRUCTIONS, tips_overrides=CATEGORY_TIPS)` produces identical answers to `FlatSoloProgram` on 3 docs (`engineering_drawing_2`, `business_report_3`, `science_paper_2`). ~10 min wall.
2. **GEPA wiring** — `max_metric_calls=10`, 5 train docs, 3 val docs. Verify reflection LM proposes a candidate, scoring runs end-to-end, ASI reaches reflection. ~30 min wall.
3. **Real run** — `max_metric_calls=150`, full split. Overnight (~8–10 hours).

## Post-optimization flow

1. Optimization completes → `output/gepa-flat-solo/<run_id>/best_candidate.json`, full Pareto frontier, run logs, wandb run.
2. **Re-eval top-3 candidates on full 25-doc val** via `solver=flat_solo_gepa solver.candidate_path=...`. ~2 hours each at `max_concurrency=8`. Decision rule: best candidate must beat seed val by ≥ 2pp.
3. **Single-shot test eval** with the winning candidate (~2 hours). Decision rule: must beat production test (43.75%).
4. **SC-8 on test** — only if step 3 wins. ~60h wall total (8 trials × ~7h, can run in parallel where capacity allows).

## Decision: extend or stop

Per the user's instruction: if the medium budget run improves at least one sample beyond seed, that's the signal to try a higher budget (300+ metric calls). Otherwise, treat it as a negative result and document.

## Out of scope (explicitly)

- New solver architecture (separate solver, not refactor).
- Optimizing VLM prompts, formatting rules, or RLM action instructions.
- SC voting strategy (orthogonal — applied at test time on the winning prompts).
- Optimizing for Pro / Flash student (this run targets local Qwen 3.5 27B specifically).
