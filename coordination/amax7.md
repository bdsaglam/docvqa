# amax7 queue

Experiment queue for the amax7 host. If a cell shows an unexpected
direction worth the other host's attention, append a `## NOTE FOR
AMAX1` section at the bottom so it's seen on the next pull.

**vllm:** Qwen 3.5 27B at `localhost:8928`.

## In progress

### `[→]` ReAct VLM-axis sweep (val, n=8) — started 2026-06-02

Per-user direction 2026-06-02: repeat the VLM-axis design with the
`react_baseline` solver (FT target is a ReAct agent, not RLM — RLM's
hidden Python-namespace state makes it a POMDP; ReAct's trajectory is a
clean observable MDP). Cells (n=8 each): R1 8B v2, R2 9B v2, R3 4B v2,
R4 9B v1, R5 4B v1 (8B text-only → v2 only). **Then CodeAct sweep
(C1-C5, `solver=codeact`)** — same VLM-axis cells, run after ReAct
(per-user 2026-06-02). Sequential, all GPUs to the running config;
27B VLM @8927 stays up for all v2 cells then torn down for v1 (DP=4).
Heartbeat-driven. **amax7 holds all 4 GPUs.** Goal: RLM-vs-ReAct-vs-
CodeAct on the model-size axis (FT target is an append-only code agent).
run_ids `react-{8b,9b,4b}-llm-27b-vlm-val-t*` (v2) +
`react-3_5-{9b,4b}-val-t*` (v1). Local driver-state in
`tmp/workspace/qwen-9b-vlm-axis/driver-state.md`; writeup target
`docs/experiments/react-vlm-axis.md`.

**RLM (`rvlm`) VLM-axis sweep — ✅ DONE (2026-06-02):** 9B v1/v2
16.67/24.54 (Δ+7.87 sig), 4B 12.49/21.09 (Δ+8.60 sig), 8B-text-only v2
11.73. Perception-budget-bound at both clean sizes (D-006).

(superseded RLM in-progress notes below, kept for history)

### `[done]` Qwen 3.5 9B model-axis: rvlm_minimal n=8 escalation (val)

Per-user direction 2026-05-31 ~21:20: escalate both VLM-axis variants
to **n=8**. t1 = the original run (renamed `*-val-t1`). t2 launched,
t3-t8 driven by heartbeat cron (one trial per variant at a time,
c=16, stall-detect + resume). Full writeup:
`docs/experiments/qwen-9b-rvlm-minimal-vlm-axis.md`.

- run_ids: `rvlm-minimal-3_5-9b-val-t{1..8}`,
  `rvlm-minimal-9b-llm-27b-vlm-val-t{1..8}`.
- tmux `eval-9b`; 9B@:8909, 27B@:8928. Early-launch rule active
  (next trial starts in a new window once current ≥22/25).
- **RE-RUN in progress (per-user 2026-06-01):** old-code trials (ran
  under the whole-agent `@retry`, before `895851f`) are being re-run
  under new code — v1 t1/t2/t3 + v2 t1/t2. Old dirs preserved as
  `*-tN-oldcode`. Motivation: whole-agent retry may have *artificially
  boosted* old-code trials (old > new in BOTH arms: v1 22.1 vs 17.8,
  v2 25.0 vs 23.8). Re-run is also the hypothesis test — re-run should
  land at the new-code level if the boost was real.
- **n=8 finding (preliminary, reverses the n=1 read):** n=1 had both
  at 21.2% ("0.0pp lift"), but n=8 trends **v2 (9B/27B) > v1 (9B/9B)
  by ~+5pp** — the stronger VLM *does* help the 9B reasoner. Lock the
  paired Δ + t-stat after the re-run completes a homogeneous-code n=8.
- On both arms clean-n=8: paired Δ → update
  `docs/experiments/qwen-9b-rvlm-minimal-vlm-axis.md`, move to Done.

### `[ ]` Qwen 3.5 4B model-axis: rvlm_minimal n=8 (val) — PHASE 2, after 9B

Per-user direction 2026-05-31 ~22:25: after the 9B n=8 completes,
extend the model-size axis downward to **Qwen 3.5 4B**, same two VLM
variants, n=8. Same heartbeat drives it (phase 2).

- **GPU handoff (forced sequential):** 9B (DP=4 @ :8909, tmux
  `vllm:qwen9b`, container `vllm-qwen9b`) holds all 4 GPUs; 4B needs
  them. Heartbeat tears down 9B (`docker stop vllm-qwen9b`) then brings
  up Qwen3.5-4B w/ vision (DP=4) @ :8904 in tmux `vllm:qwen4b`. 4B is
  multimodal (`image-text-to-text`), ~8 GB, **not cached** → first
  start downloads it.
- run_ids: `rvlm-minimal-3_5-4b-val-t{1..8}` (4B/4B),
  `rvlm-minimal-4b-llm-27b-vlm-val-t{1..8}` (4B LLM + 27B VLM @ :8928).
- configs: `configs/{lm,vlm}/qwen-3_5-4b-vllm-local.yaml` (port 8904).
- Reuse tmux `eval-9b` windows for the evals. On t8 complete: n=8
  mean±std + paired Δ → new doc `docs/experiments/
  qwen-4b-rvlm-minimal-vlm-axis.md`, move this cell to Done.

### `[ ]` Qwen 3 8B (text-only): rvlm_minimal n=8, variant 2 only — PHASE 3, after 4B

Per-user direction 2026-05-31 ~22:40. **Qwen3-8B is text-only**
(`text-generation`, no vision) → variant 1 (LLM=VLM) is impossible;
**run variant 2 only** (8B LLM + 27B VLM @ :8928). Cross-FAMILY point
(Qwen3, not Qwen3.5): a pure text reasoner that can only perceive via
the VLM tool — no image-reading fallback.

- **GPU handoff:** after 4B n=8, heartbeat tears down 4B
  (`docker stop vllm-qwen4b`) and brings up Qwen3-8B (text-only, DP=4)
  @ :8908 in tmux `vllm:qwen3-8b`, container `vllm-qwen3-8b`. Model is
  **cached** (no download).
- run_id: `rvlm-minimal-qwen3-8b-llm-27b-vlm-val-t{1..8}`.
- config: `configs/lm/qwen-3-8b-vllm-local.yaml` (port 8908). No vlm
  config needed (vlm=qwen-3_5-27b-vllm-local, api_base→:8928).
- On t8 complete: n=8 mean±std → new doc `docs/experiments/
  qwen3-8b-rvlm-minimal.md`, move this cell to Done. (No paired Δ —
  single variant.)

### `[✓]` Remove agent-level @retry from ALL solvers — DONE 2026-06-01 (commit `0a04aee`)

**Done:** after amax1's rename cascade landed (`feae419`), removed the
whole-agent `@retry` from all 12 remaining solvers (`0a04aee`; `rvlm`
already clean from `895851f`). Pure deletions (decorator + unused
tenacity/`is_retryable_lm_error` imports); all 13 solvers import OK;
grep for the retry pattern is now zero across `src/docvqa/solvers/`.
Per-call dspy `num_retries=5` (global in `types.py`) is the only retry
layer everywhere. Watcher cron retired.

---

Per-user direction 2026-05-31 ~23:00 (updated: amax1 is doing the
rename/reorg NOW, not deferred). `rvlm_minimal` already done (commit
`895851f`: dropped whole-agent `@retry`, per-call `num_retries` 3→5).
The other 15 solvers still wrap the whole agent in
`@retry(is_retryable_lm_error, stop_after_attempt(4), ...)`, which
restarts from iteration 0 on any transient error, discarding all
completed iterations. Remove it everywhere so per-call dspy retries
(`num_retries=5`, already global in `types.py`) are the only layer;
a call that still fails → exception propagates → doc fails (runner
returns None → re-run on resume).

**amax7 is polling every 30 min** for amax1's rename/delete cascade to
land + stabilize, then does the removal (name-agnostic):
```bash
grep -rl "stop_after_attempt\|is_retryable_lm_error" src/docvqa/solvers/
```
For each hit, apply the `895851f` transform: replace the
`@retry`-wrapped `_solve_one()`/equivalent with a direct model call,
drop unused tenacity + `is_retryable_lm_error` imports, confirm import,
commit. (If amax1 centralized retry into a shared helper, remove it
there instead.)

**⚠ amax1 — sweep protection:** amax7 has a LIVE model-axis sweep that
launches `solver=rvlm_minimal` (4B + Qwen3-8B phases not yet started).
The rename `rvlm_minimal`→`rvlm` + delete-old-`rvlm` will break those
launches. **Please keep a `rvlm_minimal.yaml` alias** (thin copy of the
renamed `rvlm.yaml`) until amax7 marks the model-axis sweep done — or
ping; amax7 will otherwise repoint its heartbeat to `solver=rvlm` when
the rename lands.

## Queued

*(All queued cells below moved to amax1 per user direction 2026-05-31
— see "★ HANDOFF FROM AMAX7" at top of `coordination/amax1.md` for
the new ordering and rsync prerequisites. Kept here for reference
only; do NOT run on amax7. Exception: cell 1 moved back to amax7.)*

### 1. `[ ]` rvlm_skeletal n=8 refill t6 + t7 (post-hybrid) — MOVED BACK FROM AMAX1 (2026-05-31)

Resumability needs the partial run dirs (t6=22/25, t7=23/25), which
are local on amax7, so the refill returns here. t3/t4/t5 already
refilled to 25/25. Only t6 (3 docs) + t7 (2 docs) remain. Strict
serial, low concurrency.

```bash
for T in 6 7; do
  uv run python evals.py \
    lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
    lm.enable_thinking=false \
    solver=rvlm_skeletal \
    data.split=val data.num_samples=null \
    max_concurrency=4 \
    run_id=rvlm-skeletal-val-t${T}
done
```

Acceptable to skip — current paired Δ (−1.63pp [−5.67, +2.41] n.s.)
already lands minimal as proposed method; refill only refines σ. Not
a gate on the submission path. → docs/experiments/strip-chain-naked-hybrid.md

### 2. `[→moved]` rvlm_minimal n=8 test + SC-vote submission

Paper-method submission cell. Test = 48 docs / 160 Qs. Run minimal
n=8 on test, then self-consistency vote across 8 trials → one
submission JSON. Decision gate: if skeletal Δ in noise band, method
is `rvlm_minimal`; if skeletal beats minimal by >+1pp, swap to
`rvlm_skeletal`.

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_minimal \
  data.split=test data.num_samples=null \
  max_concurrency=16 \
  run_id=rvlm-minimal-test-t${T}
```

Strict serial, c=16. Chain wall ~28-32h.

### 3. `[→moved]` Prompt-minimize other solvers (post-test gate)

Runs only after test n=8 confirms `rvlm_minimal`. Apply skeletal-style
minimization (drop doc-shape patterns; keep DATA / TOOLS / APPROACH /
VERIFY / OUTPUT) to: `rvlm_ocr_solver.py`, `raw_vlm_multi_solver.py`,
`direct_vlm_solver.py` (TOOL_HINTS audit, task #27).
`official_baseline_solver.py` already minimal — no change.

### 4. `[→moved]` rvlm_ocr n=1 val (task #14, post-minimization)

Locks the clean OCR-extension number. Runs after prompt minimization
so the lift comparison isn't confounded.

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_ocr \
  data.split=val data.num_samples=null \
  max_concurrency=16 \
  run_id=rvlm-ocr-val-t1
```

Compare to `rvlm` val (OCR-free); expect approximately equal on
DocVQA-2026. Decision rule: rvlm_ocr ≈ rvlm on val → OCR neutral on
moderate-length docs (paper §B doc-length-axis claim holds), push to
MMLongBench-Doc next.

### 5. `[→moved]` direct_vlm n=1 val (task #19) — MOVED to amax1 (2026-05-29)

Moved to amax1 to run paired with `direct_vlm_minimal` at
`max_iterations=40` (the minimal cell surfaced cap=20 is binding —
~56/80 questions hit 20/20). Caveat: `direct_vlm` will be cap=40 but
the `rvlm` n=8 headline was cap=20 — interpret the direct_vlm-vs-rvlm
comparison only after confirming the cap difference doesn't drive it
(may need an rvlm cap=40 spot-check). Decision rules: direct_vlm <
rvlm by 5+pp → recursive sub-call load-bearing (paper §C prediction 3
supported); direct_vlm ≈ rvlm → architecture-agnostic, reframing
needed (context-rationing, not the sub-call, may be load-bearing).

## Done

### `[✓]` Qwen 3.5 9B model-axis: rvlm_minimal, two VLM variants (val n=1) — 2026-05-31
Both variants 17/80 = 21.2%; VLM 9B→27B = 0.0pp at n=1 (perception slack for 9B reasoner). Superseded by the n=8 escalation in In progress. → docs/experiments/qwen-9b-rvlm-minimal-vlm-axis.md

### `[✓]` rvlm_skeletal + rvlm_naked + rvlm_hybrid n=1 val (tasks #32/#33/#35) — 2026-05-29
skeletal +0.00pp, naked −10.00pp (~4.3σ, shelved), hybrid −7.50pp; display:ask_vlm = 1397:706. → docs/experiments/strip-chain-naked-hybrid.md

### `[✓]` rvlm_skeletal + rvlm_hybrid n=2 val (task #36) — 2026-05-29
Paired (common 68-Q): skeletal −2.94pp, hybrid −8.09pp. → docs/experiments/strip-chain-naked-hybrid.md

### `[✓]` rvlm_hybrid w/ images_for_last_n=1 n=1 val (task #37) — 2026-05-30
imgN1 = 20.00% (−15pp vs hybrid baseline, −22.5pp vs minimal); display fragile to visual-window, ask_vlm robust. → docs/experiments/strip-chain-naked-hybrid.md

### `[✓]` rvlm_skeletal n=8 val — paired vs minimal (task #38) — 2026-05-30
Δ skeletal−minimal = −1.63pp [−5.67, +2.41] (n.s.); minimal stays proposed method (same headline, tighter σ). → docs/experiments/strip-chain-naked-hybrid.md

### `[✓ PARTIAL]` rvlm_skeletal n=8 refill — t3/t4/t5 (task #38) — 2026-05-31
t3/t4/t5 refilled to 25/25 at c=4; t6/t7 remain (see Queued #1). Root cause = c=32 overlap load contention, not a solver bug. → docs/experiments/strip-chain-naked-hybrid.md

### `[✓]` rvlm_hybrid n=8 val — paired vs minimal (task #39) — 2026-05-31
Δ hybrid−minimal = −5.31pp [−8.92, −1.70], t(7)=−3.48 (significant); hybrid shelved. → docs/experiments/strip-chain-naked-hybrid.md

### `[✓]` rvlm_minimal n=8 val — generality test (task #31) — 2026-05-29
n=8 mean 42.03% ± 2.21pp; paired vs unified Δ = +1.09pp [−3.14, +5.33] (n.s.) → rvlm_minimal is the proposed method. → docs/experiments/rvlm-minimal-generality.md

### `[✓]` unified-tips n=8 val (tasks #25/#28) — 2026-05-28
unified n=8 mean 40.94% ± 4.05pp; paired vs rvlm Δ ≈ 0pp [−3.91, +3.91] → promote unified to default. → docs/experiments/unified-category-tips-ablation.md

## NOTE FOR AMAX1 — experiments folder cleanup + docs refresh (2026-06-01, amax7)

Per user ("old ones are no longer valid — we changed prompts + solver
retry logic"), I cleaned up `docs/experiments/`:

- **Archived 9 pre-change writeups** (invalid numbers) to
  `archive/experiments/`: `react-baseline.md`,
  `official-baseline-qwen27b.md`, `no-loop-baseline.md`,
  `no-loop-multi-image.md`, `split-calibration-no-loop-multi.md`,
  `strip-chain-naked-hybrid.md`, `direct-vlm-il_n-and-prompt-variance.md`,
  `rvlm-minimal-generality.md`, `unified-category-tips-ablation.md`.
  (git mv — history preserved; indexed in `archive/experiments/README.md`.)
- **Untouched:** your live `{solver}-qwen-3_5-27b.md` re-run files — those
  are the current source of truth.
- **Rewrote `docs/experiments/README.md`** → high-level results view
  (7-solver matrix with current partial n + the VLM-axis table).
- **Rewrote `docs/results.md`** → drops the invalid pre-change anchors,
  points at your per-solver files, marks the doc-length (MP-DocVQA /
  MMLongBench) cells as pending-rerun.

Numbers in both docs are pulled from your in-progress cells (most n=2/3) —
when you lock cells at n=3, refresh the matrix rows. No action needed from
you unless you disagree with the archive set.

## NOTE FOR AMAX1 (2026-06-03): take over CodeAct 27B/27B (n=8)

Per-user: amax7 finishes its in-flight CA-27B trials (t4/t5/t6 → 6/8)
then pivots to smaller-model cells. **amax1: please complete CodeAct
27B/27B to n=8** — run the remaining **t7, t8** (and re-run any of
t1-t6 that dropped a doc, i.e. <25/25, to finalize).

- Solver/cmd (amax1 brings up its own 27B; lm=vlm=27B both on it):
  ```
  uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
    lm.enable_thinking=false solver=codeact data.split=val data.num_samples=null \
    max_concurrency=16 run_id=codeact-3_5-27b-val-tN
  ```
  (point both lm + vlm api_base at amax1's 27B; cap ~2 — 27B serves both
  reasoner + batch_look, heavy.)
- Locked so far (amax7): t1=35.54, t2=38.61, t3=37.48 (mean ~37.2%,
  per-question micro-avg, 25 docs/80 Qs). t4/t5/t6 finishing on amax7.
- ⚠ **maps_2 reliably IPC-deadlocks the batch_look bridge** (hung 3×:
  9B-t2, 27B-t1, 27B-t4). Signature: CodeAct step frozen (often at
  max-iter) + byte-frozen log + interp ~70-100% CPU, on its last/maps_2
  doc, past the 600s HTTP timeout. Fix: kill the run_id's procs +
  relaunch same run_id (resumes only maps_2; a fresh attempt clears it).
- This 27B/27B CodeAct cell is the **headline** (CodeAct overtakes all
  loops at scale): 4B 15.66 → 9B 24.26 → 27B ~37. Worth locking cleanly.
- Writeup to append: `docs/experiments/react-vlm-axis.md` (v1-homog /
  27B-anchor table row "Qwen3.5 27B homog | CodeAct").

## NOTE FOR AMAX1 (2026-06-03, update): CA-27B also needs t6 finalize
amax7 locked t1-t5 (35.54/38.61/37.48/35.99/30.28). **t6 stuck at 24/25**
— its last doc `engineering_drawing_1` IPC-deadlocked the batch_look
bridge TWICE (kill+resume re-hit it); amax7 killed it and pivoted to
smaller models. **amax1: resume run_id `codeact-3_5-27b-val-t6` (re-runs
only engineering_drawing_1) + run t7, t8** → CA-27B n=8. Crop-heavy docs
(engineering_drawing_1, maps_2) are the bridge-hang triggers; a fresh
resume usually clears them, but if engineering_drawing_1 keeps hanging
on t6, consider it a known-bad doc for that seed and move on / re-seed.
