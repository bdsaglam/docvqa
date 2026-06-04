# amax1 queue

Experiment queue for the amax1 host. If a cell shows an unexpected
direction worth the other host's attention, append a `## NOTE FOR
AMAX7` section at the bottom so it's seen on the next pull.

**vllm:** brings up its own per-model containers as needed. Set
`MODEL_TAG` per cell.

## In progress

### `[→]` solver-comparison re-run (val, n=3) — started 2026-06-01T11:03

Running the locked 7-solver × n=3 matrix below. **c=24**, overlap-the-
tail (launch next when current ≥80% docs = 20/25, cap 2 concurrent),
driven by an OS-crontab heartbeat
(`tmp/workspace/solver-cmp/heartbeat.sh`, every 10 min) + queue
(`queue.txt`). run_ids use a `-cmp-` tag to avoid resuming the kept
prior-session `rvlm-val-*` / `react-val-*` dirs. t1 launched:
`rvlm-cmp-val-t1`. Results land in
`docs/experiments/{solver}-qwen-3_5-27b.md` as trials complete.

## Queued

### ★ solver-comparison re-run (val, n=3) — LOCKED PLAN, 2026-06-01

**Why:** retry logic changed (whole-agent `@retry` removed; per-call
`num_retries=5` is the only retry layer now) AND prompts were minimized
+ parity-stripped vs the `rvlm` reference (`bc20ba8`). Old numbers
aren't comparable → re-run all comparison solvers under current code.
This SUPERSEDES the earlier "pending re-val" list.

**Matrix:** n=3, **val** split, **Qwen 3.5 27B** (lm+vlm local,
`lm.enable_thinking=false`).

**Solvers (7, LOCKED — `rvlm_full` + `raw_vlm_single` excluded):**
- method: `rvlm`
- ablations: `rvlm_ocr_ablation`, `rvlm_hybrid_ablation`
  (`rvlm_full` deferred — `rvlm_ocr` already covers the OCR-extension
  insight; the only delta is the extra `look()` tool, expected
  immaterial per user hunch)
- baselines: `raw_vlm_multi_baseline`, `react_baseline`, `direct_vlm`
  (`raw_vlm_single` excluded — `raw_vlm_multi` is the stronger baseline)

**Orchestration (NOT chained):** launch each trial individually; when a
run reaches its long tail (~21/25 docs), launch the next in parallel
(overlap-the-tail), cap ~2 concurrent so vllm 8927 isn't double-
saturated. 21 runs total. `direct_vlm` uses `solver.max_iterations`
default=40.

**Results → `docs/experiments/{solver}-qwen-3_5-27b.md`** (per the
README naming convention); cross-solver head-to-head table in
`docs/results.md`.

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=<solver> \
  data.split=val data.num_samples=null \
  max_concurrency=16 \
  run_id=<solver>-val-tN
```

### Test submission (deferred)

`rvlm` n=8 on **test** (48 docs / 160 Qs) + SC-8 majority-vote
submission, under current code. The earlier `rvlm-minimal-test` chain
was stopped to re-run on final code. SC-vote script: for each
`(doc_id, question_id)`, majority-vote the literal answer string across
trials, tie-break by trial-with-highest-val-score → write
`output/submissions/rvlm-test-sc8.json`. User uploads to the
competition server manually.

### Model-axis re-runs (clean prompts) — deferred

Gemma-4 E4B / Gemma-4 31B baseline+scaffold n=1 val on clean prompts.
Direction robust from the original n=3 (lift sign preserved); just
locking magnitudes. Per-model vllm bringup is unscripted — Gemma-4 needs
the right tool-call parser (wrong parser → silent tool-call failure →
fake-low scores), and 31B as-written wants TP=4 (amax1 has 3 GPUs).
vllm template + ports in
`tmp/workspace/amax1-model-axis/vllm-bringup-notes.md`. (Qwen 3.5 9B
model-axis is claimed by amax7.)

## Done (coordination log — findings live in docs/experiments/)

- **Solver minimization + rename cascade** (2026-06-01, `feae419` +
  `693c0c9`): minimal→canonical (`rvlm`), baselines get `_baseline`,
  variants get `_ablation`; `direct_vlm` minimized in place; deleted old
  heavy `rvlm` / `rvlm_unified` / `direct_vlm_minimal`. Shared helpers
  (`_build_signature` / `_create_tools` / `_build_sandbox_code`) now live
  in `rvlm_solver.py`. All old `solver=` names kept as TEMP aliases —
  see NOTE FOR AMAX7 below.
- **direct_vlm il_n / compaction / max_messages investigation**
  (2026-05-29 → 31, parked): il_n=3 ≈ 35% ± 5pp (non-comics); the 43.2%
  was a high-variance draw, not a regression. Full story (il_n sweep,
  >64-image crash + per-Q catch, compaction & max_messages failures,
  prompt-vs-variance, old-build rerun) →
  `docs/experiments/direct-vlm-il_n-and-prompt-variance.md`.
- **ReAct baseline n=8 val** (`react-val-t1..t8`): REPL is load-bearing,
  Δ = −10.47pp vs `rvlm`. → `docs/experiments/react-baseline.md`.
- **rvlm n=8 val** (`rvlm-val-t1..t8`): unified-tips paired anchor,
  Δ = 0.00pp vs `rvlm_unified`. →
  `docs/experiments/unified-category-tips-ablation.md`.

## NOTE FOR AMAX7 — solver rename/cleanup DONE (2026-06-01)

The full solver minimization + rename cascade is committed & pushed
(`feae419` minimal→canonical + baselines; `693c0c9` `_ablation` infix).
**All old `solver=` names are kept as TEMP aliases, so your existing
commands keep working** — nothing breaks. Switch to the new names when
convenient.

New canonical names:
- **`rvlm`** ← was `rvlm_minimal` (the proposed method).
- Baselines get `_baseline`: `react_baseline`,
  `raw_vlm_single_baseline`, `raw_vlm_multi_baseline`.
- Variants/ablations get `_ablation`: `rvlm_skeletal_ablation`,
  `rvlm_naked_ablation`, `rvlm_hybrid_ablation`, `rvlm_full_ablation`,
  `rvlm_ocr_ablation`, `rvlm_gepa_ablation`.
- `direct_vlm` minimized in place (kept name).

⚠ **Do NOT delete `configs/solver/rvlm_minimal.yaml`** — your live
model-axis sweep invokes `solver=rvlm_minimal`. All TEMP aliases get
removed only once both hosts reference new names.

## NOTE FOR AMAX7 (2026-06-04): 27B-only directive — v3 deferred, CodeAct-27B reuses b40

Per-user (2026-06-04): amax1 keeps its hosted 27B up (shared with other
experiments — do NOT restart it) and runs **only Qwen-27B experiments**;
other model families deferred for now. Effect on your handoffs:

- **v3 (LLM=27B / VLM=9B) RLM/CodeAct/ReAct — DEFERRED.** Needs a 9B VLM
  container = other family. Synced partials' state on amax1:
  `codeact-27b-llm-9b-vlm-val-t4` 25/25 (done), `react-27b-llm-9b-vlm-val-t8`
  25/25 (done), `rvlm-minimal-27b-llm-9b-vlm-val-t3` 24/25 (1 doc left,
  needs 9B to finish). Picks back up once a 9B server is allowed.
- **CodeAct 27B/27B n=8 (`codeact-3_5-27b-val-*`) — satisfied by amax1's
  `codeact-b40` budget-sweep arm, NOT re-run.** `b40` is config-identical
  (`solver=codeact max_iterations=40`, lm=vlm=27B) and amax1 already took
  it to n=8 (mean ~37, see `docs/experiments/codeact-qwen-3_5-27b.md`).
  Per-user we reuse b40 n=8 as the harness-types "27B CodeAct anchor"
  rather than burn ~2-3h re-running t6/t7/t8. Your locked t1-t5
  (35.54/38.61/37.48/35.99/30.28, mean ~35.6) are consistent with b40.

amax1 continues its own approved 27B queue to exhaustion: codeact 3-budget
sweep (b24 ✓, b56 ✓, b40 finishing) + rvlm_nocrop_ablation n=8 +
rvlm_subagent_ablation n=8.
