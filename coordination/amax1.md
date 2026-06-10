# amax1 queue

Experiment queue for the amax1 host. If a cell shows an unexpected
direction worth the other host's attention, append a `## NOTE FOR
AMAX7` section at the bottom so it's seen on the next pull.

**vllm:** brings up its own per-model containers as needed. Set
`MODEL_TAG` per cell.

## In progress

### `[→]` rvlm_vsearch n=8 (val) — started 2026-06-10T19:40

New solver: rvlm + multimodal embedding `search` (ColModernVBERT via
colpali-engine, OCR-free; see `docs/superpowers/specs/2026-06-10-rvlm-vsearch-design.md`).
Per-user: straight to n=8, c=24. Sequential t1→t8 with overlap-the-tail
(≥21/25, cap 2 concurrent), session-cron heartbeat. `solver.vsearch_device=cuda:1`
(embedder on the idle GPU; placement only). run_ids `rvlm-vsearch-val-t1..t8`,
tmux `rvlm-vsearch-val-tN`. Results → `docs/experiments/rvlm-vsearch-qwen-3_5-27b.md`.

```bash
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false solver=rvlm_vsearch solver.vsearch_device=cuda:1 \
  data.split=val data.num_samples=null max_concurrency=24 run_id=rvlm-vsearch-val-tN
```

_(The 2026-06-01 solver-comparison re-run that used to sit here is **done** —
all 7 solvers locked at n≥3 in `docs/results.md` / `docs/experiments/
{solver}-qwen-3_5-27b.md`. Removed from In progress 2026-06-10.)_

## Queued

### Test submission (deferred)

`rvlm` n=8 on **test** (48 docs / 160 Qs) + SC-8 majority-vote
submission, under current code. The earlier `rvlm-minimal-test` chain
was stopped to re-run on final code. SC-vote script: for each
`(doc_id, question_id)`, majority-vote the literal answer string across
trials, tie-break by trial-with-highest-val-score → write
`output/submissions/rvlm-test-sc8.json`. User uploads to the
competition server manually.

### Dataset / document-length axis — deferred, needs go-ahead

rvlm / codeact / `raw_vlm_multi_baseline` / `official_baseline`
(±`rvlm_ocr_ablation`) on **MP-DocVQA + MMLongBench-Doc**, Qwen 3.5 27B.
MANDATORY cross-benchmark rules: `data.dataset=<id>`,
`data.use_profile_scoring=true`, and **raise `solver.max_pages`** so the
raw-VLM baseline sees evidence pages (else the lift double-counts
truncation). Earlier MP-DocVQA/MMLongBench numbers are pre-2026-06-01 /
invalid. n=1 → escalate. Plan: `tmp/workspace/solver-cmp/DATASET_AXIS_QUEUE.md`.

## Done (coordination log — findings live in docs/experiments/)

- **Gemma n=8 model-axis sweep + harness-lift** (2026-06-10, `7f5a320`):
  E4B + 31B homog, all 3 harnesses (rvlm/react/codeact) + 2 no-scaffold
  baselines (raw_vlm_multi, official). Per-model harness-LIFT table:
  31B every harness ≫ both baselines (rvlm +21.4); E4B no lift (4B
  negative control). 31B codeact n=5 (stopped early per user; gemma4-31B
  codeact operationally fragile — 8 shm-crashes + degenerate/max-iter
  runaways). → `docs/experiments/gemma-4-{e4b,31b}.md`,
  `harness-axis-summary.md` (Finding 5). Supersedes the old "model-axis
  re-runs n=1" cell.
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

~~⚠ Do NOT delete `configs/solver/rvlm_minimal.yaml` — your live
model-axis sweep invokes `solver=rvlm_minimal`.~~ **RESOLVED (2026-06-07):**
the model-axis sweeps completed (all cells locked in `docs/results.md`),
so the `rvlm_minimal.yaml` TEMP alias was deleted. Canonical name is
`solver=rvlm`. Other TEMP aliases (if any remain) get removed once both
hosts reference new names.

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

## NOTE FOR AMAX7 (2026-06-06): GPU split EXECUTED, v3 running on amax1

Per user, did the GPU split + picked up v3. **27B is now DP=2** (container
qwen35-27b, `--gpus all -e CUDA_VISIBLE_DEVICES=0,1`, `--data-parallel-size 2`,
@8927) on GPUs 0,1; **9B brought up** (container qwen35-9b, CVD=2, DP=1,
@8909) on GPU 2. Both verified serving; v3 wiring (27B-LLM reasons → 9B-VLM
batch_look) confirmed end-to-end. ⚠ The DP=2 restart RECOMPILES (fresh
torch_compile cache key) → ~5min startup, not instant.

Running v3 (lm=27B@8927 / vlm=9B@8909, c=8, 2 concurrent): RLM
`rvlm-minimal-27b-llm-9b-vlm-val-t3..t8` (t3 resuming 24/25; t1=32.60,
t2=34.74 are yours), CodeAct `codeact-27b-llm-9b-vlm-val-t5..t8` (t4 synced
done; t1=27.87/t2=35.24/t3=32.81 yours). ReAct v3 = 8/8 already (your t1-t7 +
synced t8) — its writeup row needs YOUR per-trial numbers to lock (only t8 is
local here). Managed by cron; will finalize the RLM/CodeAct v3 n=8 into
harness-axis-summary.md.

Earlier full-agent sub-agent experiment (rvlm_subagent_full) concluded
NEGATIVE at n=8 (pilot crossover was noise) — see its experiment doc.

## NOTE FOR AMAX7 (2026-06-06): v3 DONE — locked n=8 in harness-types

v3 (27B-LM/9B-VLM) finished on amax1. Locked n=8 in
`docs/experiments/harness-axis-summary.md`:
- RLM 34.82±3.01 (+10.3 vs v2 24.54) — reasoning-bound
- CodeAct 30.43±2.86 (+6.2 vs v2 24.26) — reasoning-bound
- ReAct 17.96±3.94 (−3.05) — perception-bound (your t1-t7 + synced t8)
Headline: the REPL crop/zoom loop converts reasoning→perception (RLM/CodeAct
gain from a stronger reasoner even on weaker VLM); ReAct has no actuator → loses.
27B still DP=2 + 9B @8909 up (idle now v3 is done).
