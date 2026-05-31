# amax7 queue (adaptive host)

amax7 is the adaptive host — runs critical-path experiments where the
result might change the experiment plan. Tighter feedback loops; one
cell at a time; replan after each result.

**vllm:** Qwen 3.5 27B at `localhost:8927`.

## In progress

### `[→]` Qwen 3.5 9B model-axis: rvlm_minimal n=8 escalation (val)

Per-user direction 2026-05-31 ~21:20: escalate both VLM-axis variants
to **n=8**. t1 = the original run (renamed `*-val-t1`). t2 launched,
t3-t8 driven by heartbeat cron (one trial per variant at a time,
c=16, stall-detect + resume). Full writeup:
`docs/experiments/qwen-9b-rvlm-minimal-vlm-axis.md`.

- run_ids: `rvlm-minimal-3_5-9b-val-t{1..8}`,
  `rvlm-minimal-9b-llm-27b-vlm-val-t{1..8}`.
- tmux `eval-9b` (`v1-homog` + `v2-mixed`); 9B@:8909, 27B@:8928.
- On t8 complete: n=8 mean±std + paired per-trial Δ, update doc + this
  cell.

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

### `[→]` Remove agent-level @retry from ALL solvers — amax7 POLLING (amax1 refactor in progress now)

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

## Done

### `[✓]` Qwen 3.5 9B model-axis: rvlm_minimal, two VLM variants (val n=1) — 2026-05-31

(Superseded by the n=8 escalation above; t1 numbers retained.)

**Result (both n=1, val, 80 Q / 25 docs):**
- Variant 1 — homogeneous 9B (LLM=VLM=9B): `rvlm-minimal-3_5-9b-val-t1`
  → **17/80 = 21.2% ANLS**
- Variant 2 — 9B LLM + 27B VLM: `rvlm-minimal-9b-llm-27b-vlm-val-t1`
  → **17/80 = 21.2% ANLS**

**Takeaway:** upgrading the VLM 9B→27B moved the headline by **0.0pp**
at n=1 — for the 9B reasoner under `rvlm_minimal`, perception is not
the binding constraint; the 9B orchestrator/LLM is. (Same correct
*count* (17), not necessarily the same questions — n=1, ~3–4% trial
σ, so treat as "no detectable lift" rather than literally identical.)
Consistent with the small-model caveat (scaffold/perception lift
scales with model size). To make this paper-grade would need n≥3 on
both arms.

Ops note: variant 2 hung once on a non-returning `maps_2_q5` model
call (~1h50m, never logged iter 1 — I/O stall, not a loop); killed +
resumed (resumable re-ran only maps_2), q5 then CORRECT.

Claimed by amax7 2026-05-31 18:38 (tmux `eval-9b`). Per-user
direction — superseded the 9B model-axis cell queued for amax1
(amax1 task #4). vllm: Qwen3.5-9B w/ vision (DP=4) @ :8909, 27B @ :8928.

<details><summary>commands</summary>

Two variants, both `solver=rvlm_minimal`, `max_concurrency=16`,
`lm.enable_thinking=false`, `data.split=val data.num_samples=null`:

```bash
# Variant 1 — homogeneous 9B (LLM=VLM=9B), 9B vllm w/ vision @ :8909
uv run python evals.py lm=qwen-3_5-9b-vllm-local vlm=qwen-3_5-9b-vllm-local \
  lm.enable_thinking=false solver=rvlm_minimal \
  data.split=val data.num_samples=null max_concurrency=16 \
  run_id=rvlm-minimal-3_5-9b-val

# Variant 2 — mixed: 9B LLM @ :8909, 27B VLM @ :8928
uv run python evals.py lm=qwen-3_5-9b-vllm-local \
  vlm=qwen-3_5-27b-vllm-local vlm.api_base=http://localhost:8928/v1 \
  lm.enable_thinking=false solver=rvlm_minimal \
  data.split=val data.num_samples=null max_concurrency=16 \
  run_id=rvlm-minimal-9b-llm-27b-vlm-val
```

- Scope: rvlm_minimal only (not the run_gemma_chain baseline+scaffold
  pair). If a clean baseline lift is wanted later, add raw_vlm_multi 9B.

</details>

## Queued

*(All queued cells below moved to amax1 per user direction 2026-05-31
— see "★ HANDOFF FROM AMAX7" at top of `coordination/amax1.md` for
the new ordering and rsync prerequisites. Kept here for reference
only; do NOT run on amax7.)*

### 1. `[ ]` rvlm_skeletal n=8 refill t6 + t7 (post-hybrid) — MOVED BACK FROM AMAX1 (2026-05-31)

**Moved back from amax1 per user direction 2026-05-31:** amax1 cannot
run this refill — resumability needs the partial run dirs (t6=22/25,
t7=23/25), which are **local on amax7**. The rsync-to-amax1 prerequisite
was the blocker, so the refill returns to amax7 where the data lives.

t3/t4/t5 were already refilled to clean 25/25 by amax7 (see the
`[✓ PARTIAL]` cell in Done). Only **t6 (3 docs) + t7 (2 docs)** remain.

**Strict serial — one trial at a time, low concurrency to avoid
recreating the load problem that caused the original timeouts.**

Per-trial refill (resumable via run_id — only re-runs missing docs;
no rsync needed, partial dirs already on amax7):
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

- Refill scope: t6=3 docs, t7=2 docs → 5 docs.
- Wall: ~3-4h at sequential c=4.
- After refill, re-run paired Δ skeletal-minimal at n=8 (per-trial
  intersection; see `git show 14e4784` for the script pattern) to get
  the clean σ for the paper. Update the `[✓ PARTIAL]` cell in Done with
  the refined number. Decision rule: Δ < +0.5pp or CI95 includes 0 →
  **minimal** is proposed method (default); Δ > +1pp → **skeletal**.
- **Note (acceptable to skip):** current paired Δ on the intersection
  (−1.63pp [−5.67, +2.41] n.s.) already lands minimal as proposed
  method, so this refill only refines σ — it is not a gate on the
  submission path.

### 2. `[→moved]` rvlm_minimal n=8 test + SC-vote submission

Paper-method submission cell. Test set = **48 docs / 160 Qs** (≈2×
val scope). Run minimal n=8 on test, then self-consistency vote
across 8 trials to produce one submission JSON for upload.

**Decision gate (S4 in heartbeat):** if refilled skeletal Δ < +0.5pp
or in the noise band, the proposed method is `rvlm_minimal` and this
cell runs. If skeletal beats minimal by >+1pp (unlikely from current
trajectory), swap `rvlm_minimal` → `rvlm_skeletal` here.

Per-trial:
```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_minimal \
  data.split=test data.num_samples=null \
  max_concurrency=16 \
  run_id=rvlm-minimal-test-t${T}
```

- Strict serial — one trial at a time, no overlap. c=16 (vs val
  c=32) gives test trials more vllm headroom and reduces timeout
  risk on the larger test set.
- Trial wall: ~3-4h. **Chain wall: ~28-32h sequential.**
- After all 8 land: per-question majority vote across 8 prediction
  files → single submission JSON. User uploads to competition server
  manually.

### 3. `[→moved]` Prompt-minimize other solvers (post-test gate)

**Decision gate:** runs only after test n=8 confirms `rvlm_minimal`
as the proposed method.

If skeletal-style minimization gave `rvlm_minimal` equal mean +
tighter σ, the same minimization principle (drop doc-shape
patterns, keep only DATA / TOOLS / APPROACH / VERIFY / OUTPUT) should
apply to other solvers before they get evaluated. Otherwise their
prompts are confounded with extra DocVQA-2026-tuned content that the
proposed-method case shows is unnecessary.

Solvers to minimize:
- `rvlm_ocr_solver.py` — currently mirrors `rvlm_unified` doc-shape
  patterns + OCR-specific tips. Strip to minimal + OCR-tool docs.
- `raw_vlm_multi_solver.py` — currently has category tips appended.
  Strip to bare task + format rules.
- `direct_vlm_solver.py` — currently has TOOL_HINTS section. Audit
  parity with skeletal-style minimization (task #27).
- `official_baseline_solver.py` — already minimal (verbatim
  competition prompt); no change.

Apply same skeletal-style edit (sub-agent if not trivial), then
proceed to cell 4.

### 4. `[→moved]` rvlm_ocr n=1 val (task #14, post-minimization)

Locks the clean OCR-extension number. Runs AFTER prompt minimization
above so the lift comparison isn't confounded with DocVQA-tuned
content stripped from `rvlm_minimal`.

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_ocr \
  data.split=val data.num_samples=null \
  max_concurrency=16 \
  run_id=rvlm-ocr-val-t1
```

- Expected wall: ~50min
- Compare to: `rvlm` val (OCR-free); expect approximately equal on
  DocVQA-2026 (moderate-length docs).

### 2. `[→moved]` direct_vlm n=1 val (task #19) — MOVED to amax1 (2026-05-29)

Moved to `coordination/amax1.md` to run paired with `direct_vlm_minimal`
on the throughput host, at **`max_iterations=40`** (the minimal cell
surfaced that cap=20 is binding — ~56/80 questions hit 20/20). Running
both direct-VLM-architecture cells at the same cap keeps the
prompt-stripping (TOOL_HINTS) comparison clean.

**Caveat for the paper claim (decision rules below):** `direct_vlm`
will now be cap=40, but the `rvlm` headline (n=8) was cap=20. The
`direct_vlm < rvlm by 5+pp` / `≈ rvlm` rules below assume equal
iteration budget — interpret the direct_vlm-vs-rvlm comparison only
after confirming the cap difference doesn't drive it (an rvlm cap=40
spot-check may be needed).

## Done

### `[✓ PARTIAL]` rvlm_skeletal n=8 refill — t3, t4, t5 only (task #38) — 2026-05-31

amax7 refilled t3 (7 missing→0), t4 (3→0), t5 (5→0) at c=4 strict
serial. Then released for user's other experiments per 13:xx
direction. **t6 (3 missing) and t7 (2 missing) handed off to
amax1** — see "★ HANDOFF FROM AMAX7" cell in
`coordination/amax1.md`.

| trial | pre-refill | post-amax7-refill |
|---|---:|---:|
| t3 | 18/25 | **25/25** ✓ |
| t4 | 22/25 | **25/25** ✓ |
| t5 | 20/25 | **25/25** ✓ |
| t6 | 22/25 | 22/25 (→ amax1) |
| t7 | 23/25 | 23/25 (→ amax1) |

amax1 needs to rsync t6/t7 partial run dirs to use resumability.
Skipping refill is also acceptable — current paired Δ
(−1.63pp [−5.67, +2.41] n.s.) already lands minimal as proposed
method.

Refill confirmed the per-doc timeout root cause: at c=4 (vs the
original c=32 overlap chain), every long-tail doc — including
science_paper_1 — completed within the 4h task_timeout. The
overlap pattern's load contention was the timeout driver, not any
solver-specific bug.

### `[✓]` rvlm_hybrid n=8 val — paired vs minimal (task #39) — 2026-05-31

run_ids: `rvlm-hybrid-val-t{1..8}` on Qwen 3.5 27B local, c=32.
Chain wall: ~20h start to sentinel (~12:09 → 08:30 next day).

**Paired Δ hybrid − minimal at n=8 (per-trial intersection):**
**Δ = −5.31pp ± 4.32pp sd**, SE 1.53pp, 95% CI [−8.92, −1.70]pp,
t(7) = −3.48 (**significant** — CI excludes 0). Confirms the n=2
signal (Δ = −8pp paired) tightened to −5.31pp at n=8.

Per-trial table (clean — all 8 trials hit 80/80 Qs, unlike skeletal):

| trial | n | hybrid | minimal | Δ |
|---|---:|---:|---:|---:|
| t1 | 80 | 35.00% | 42.50% | −7.50pp |
| t2 | 80 | 35.00% | 42.50% | −7.50pp |
| t3 | 80 | 31.25% | 40.00% | −8.75pp |
| t4 | 80 | 38.75% | 41.25% | −2.50pp |
| t5 | 80 | 43.75% | 45.00% | −1.25pp |
| t6 | 80 | 37.50% | 38.75% | −1.25pp |
| t7 | 80 | 40.00% | 41.25% | −1.25pp |
| t8 | 80 | 32.50% | 45.00% | −12.50pp |

Marginals: hybrid 36.72% ± 4.12pp (range 31.25–43.75); minimal
42.03% ± 2.21pp. Hybrid is ~2× noisier across trials AND ~5pp
behind on mean.

**Paper reading.** Hybrid is the "we tried it and it didn't work"
cell. Adding a second perception channel (`display()`) on top of
`ask_vlm()` degrades performance at n=8 with statistical
significance, despite the agent's revealed 2:1 preference for
`display()` over `ask_vlm()` (counted at n=1: 1397 vs 706 calls).
The agent's revealed preference for direct perception is the
**wrong** preference under this regime — forcing delegation through
`ask_vlm` is the right design choice for the rvlm family. Strong
evidence for the paper's discussion of why recursive perception >
direct perception when the LM is identical.

**Methodological footnote.** Hybrid was the only chain in the
rvlm-* family with NO doc-timeouts (8/8 trials clean 80-Q
intersections). The display() channel appears to sidestep the
agent-loop hang mode that catches `science_paper_1` in
`rvlm_minimal` and `rvlm_skeletal`. Worth a parenthetical in the
paper: when measuring an ablation we care about *more* than just
mean — variance, completion, and failure mode all matter.

Naked **shelved** (n=1 −10.00pp, ~4.3σ outside minimal's noise band;
pre-set "−5pp or worse → load-bearing" rule triggered — no n=2/n=8
needed). Full writeup in
`docs/experiments/strip-chain-naked-hybrid.md`.

### `[✓]` rvlm_skeletal n=8 val — paired vs minimal (task #38) — 2026-05-30

run_ids: `rvlm-skeletal-val-t{1..8}` on Qwen 3.5 27B local, c=32.
Chain wall: ~9.5h (orch + 8 trials w/ overlap).

**Paired Δ skeletal − minimal at n=8 (per-trial intersection):**
**Δ = −1.63pp ± 4.83pp sd**, SE 1.71pp, 95% CI [−5.67, +2.41]pp,
t(7) = −0.954 (n.s.). Lands cleanly in the "≈ 0pp / within paired
noise" band.

Per-trial paired table (n_common varies; skeletal lost more docs to
the long-tail science_paper_1 hang — pass-2 refill not run for the
short-tail trials, but pair-on-intersection is robust to that):

| trial | n_common | skeletal | minimal | Δ |
|---|---:|---:|---:|---:|
| t1 | 80 | 42.50% | 42.50% | +0.00pp |
| t2 | 80 | 38.75% | 42.50% | −3.75pp |
| t3 | 49 | 38.78% | 36.73% | +2.04pp |
| t4 | 63 | 38.10% | 39.68% | −1.59pp |
| t5 | 59 | 42.37% | 47.46% | −5.08pp |
| t6 | 65 | 43.08% | 35.38% | +7.69pp |
| t7 | 68 | 30.88% | 38.24% | −7.35pp |
| t8 | 80 | 40.00% | 45.00% | −5.00pp |

Marginals: skeletal n=8 = 39.31% ± 3.92pp (range 30.88–43.08);
minimal n=8 = 42.03% ± 2.21pp (range 38.75–45.00). Skeletal is ~1.8×
noisier across trials.

**Reading.** The 3 doc-shape patterns in `rvlm_minimal`
(high-density single page, many-page document, counting /
superlatives) are **not load-bearing for the headline score** —
paired Δ is well inside noise. But they tighten the trial-to-trial
variance: with the patterns, σ drops from 3.92pp → 2.21pp. The
patterns give the agent a more consistent reading discipline rather
than a better one.

**Promotion decision.** Both viable. Keep `rvlm_minimal` as the
proposed method — same headline, tighter variance is a free win for
the paper's "method is stable" story. Skeletal stays as the
ablation cell: "dropping the 3 doc-shape pattern bullets does not
change the headline but doubles σ."

### `[✓]` rvlm_hybrid w/ images_for_last_n=1 n=1 val (task #37) — 2026-05-30

Tested the "visual window eviction" hypothesis for hybrid's −8pp deficit.
Overrode `solver.images_for_last_n=1` (vs yaml default 3). Hypothesis
was that hybrid's deficit might come from stale-image confusion across
multi-turn display() calls (tighter window helps) OR from helpful
multi-image context being too small (tighter window hurts).

**Result: imgN1 = 20.00% (16/80) — clean 25/25.**
- vs hybrid baseline (images_for_last_n=3): **−15.00pp**
- vs minimal: **−22.50pp**

The "multi-image context helps" branch, much more extreme than
predicted. The display() strategy heavily depends on accumulating
visual context across turns; collapsing the window to 1 cripples it.

Tool-usage ratio essentially unchanged:
- baseline hybrid (n=3): display 1397 / ask_vlm 706 (66:34)
- imgN1: display 1486 / ask_vlm 706 (68:32)
- ask_vlm count is *literally identical* (706:706)
- agent didn't adapt strategy to the smaller window — kept using
  display() at the same rate, just with broken context retention.

**Implication for the paper.** This is direct evidence that
display()-based perception is fragile to context-window choice in a
way that ask_vlm() (recursive sub-VLM) is not — sub-VLM text answers
persist in the trajectory regardless of how many later display()
calls evict images. The forced-delegation design of the rvlm family
sidesteps a real failure mode of multimodal-LM-in-the-loop solvers.

Open next-cell options if we want to keep digging:
- images_for_last_n=8 (or 12): does pushing the window up recover
  hybrid? If yes, locks the "display needs lots of context" story.
- per-question correlation: are hybrid's wrongs concentrated on docs
  where the agent did many display() calls (cumulative eviction)?

### `[✓]` rvlm_skeletal + rvlm_hybrid n=2 val (task #36) — 2026-05-29

n=2 follow-up to confirm n=1 reads. Skeletal-t2 lost 4 docs to
long-tail timeout (refill pending); hybrid-t2 was clean 25/25.

**Paired Δ on common 21-doc / 68-Q subset (clean 4-trial compare):**

| trial | score | Δ vs minimal-t* |
|---|---|---|
| minimal-t1 | 44.12% | — |
| minimal-t2 | 42.65% | — |
| skeletal-t1 | 41.18% | **−2.94pp** |
| skeletal-t2 | 39.71% | **−2.94pp** |
| hybrid-t1 | 35.29% | **−8.82pp** |
| hybrid-t2 | 35.29% | **−7.35pp** |

**n=2 paired mean Δ (common 68-Q):**
- **skeletal − minimal = −2.94pp** (eerily consistent across both trials)
- **hybrid − minimal = −8.09pp** (also tight: −8.82, −7.35)

**Updated reads (revising n=1):**
- **skeletal ≈ minimal but slightly below** (−2.94pp paired n=2 on
  common subset). The n=1 full-set "tie" was on different doc sets
  per trial. Clean paired comparison shows skeletal is consistently
  ~3pp below minimal — close to the pre-set ±2pp threshold but not
  cleanly inside it. n=8 needed to call this a tie vs a small cost.
- **hybrid ~8pp below minimal at n=2.** Both trials agree the drop
  is real (not n=1 noise). The "agent prefers display, scores
  lower" finding is confirmed. n=8 not needed to call hybrid worse;
  the question for the paper is *why*.

**Hybrid identical-score curiosity.** hybrid-t1 and hybrid-t2 both
scored 24/68 on the common subset. With temperature=1.0 across 4
parallel sub-VLMs, exact agreement is suspicious but plausible at
small denominators. Worth a per-question check if we publish this.

### `[✓]` rvlm_skeletal + rvlm_naked + rvlm_hybrid n=1 val (tasks #32 #33 #35) — 2026-05-29

Strip-chain n=1, Qwen 3.5 27B local, c=32. All three refilled to clean
25/25 docs / 80 questions. Reference: rvlm_minimal n=8 mean **42.03%
SD 2.21pp** (t1 = 42.50%).

**Locked full-set numbers (80 Q each):**

| Solver | n=1 score | vs minimal-t1 | vs minimal n=8 mean |
|---|---|---|---|
| minimal-t1 | 42.50% (34/80) | — | within mean |
| **skeletal-t1** | **42.50% (34/80)** | **+0.00pp** | within mean |
| naked-t1 | 32.50% (26/80) | **−10.00pp** | **−9.53pp (~4.3σ)** |
| hybrid-t1 | 35.00% (28/80) | **−7.50pp** | **−7.03pp (~3.2σ)** |

**Reads (n=1):**
1. **skeletal ≡ minimal at n=1** (literal 34/80 tie). The 3 doc-shape
   patterns (high-density, many-page, counting) don't carry the
   method. Strong promote-to-default candidate. Run n=2 → n=8 to
   confirm at the headline level.
2. **naked drops 10pp** — well outside minimal's σ (≈4.3σ at the
   point estimate). Removing the APPROACH steps + the verify-under-
   VLM-stochasticity principle costs real points. **APPROACH +
   verify are load-bearing.** Naked is a step too far; don't escalate.
3. **hybrid drops 7.5pp** — also outside minimal's σ. Adding a second
   perception channel (display) on top of ask_vlm hurts in n=1.
   Worth n=2 to confirm the magnitude.

**Tool-preference insight from hybrid (paper-worthy).** Counted
code-block tool calls across the full hybrid trial:
- `display(...)`: **1397 calls**
- `ask_vlm(...)`: **706 calls**

Agent strongly preferred `display()` over delegation (~2:1).
*Given the choice, the agent picked direct perception. And the score
dropped 7.5pp.* This is direct evidence that the rvlm family's
forced-delegation pattern is the right one — the agent's revealed
preference points toward seeing-itself, but seeing-itself produces
worse answers. A nice find for the paper's discussion of why
recursive perception > direct perception when the LM is identical.

**Operational note.** The new `rvlm_overlap_orch.py` pattern kept
the GPU warm across the n=1 chain (no idle on long-tails). Refill
needed for skeletal (4 docs) + naked (1) — sequential pass at full
14400s timeout. Hybrid was already 25/25 (no long-tail; possibly
because MultimodalRLM's inline-image channel sidesteps the
agent-loop hang mode that catches science_paper_1 in the rvlm
family).

### `[✓]` rvlm_minimal n=8 val — generality test (task #31) — 2026-05-29

run_ids: `rvlm-minimal-val-t{1..8}` · **n=8 mean 42.03% ± 2.21pp**
(range 38.75–45.00) on Qwen 3.5 27B local, c=32.

Paired vs `rvlm_unified` t1..t8: **Δ = +1.09pp [CI95: −3.14, +5.33]**,
paired t = 0.611, df = 7 (n.s.). Lands cleanly in the pre-set
"≈ 0pp / within paired noise" band → **`rvlm_minimal` is the proposed
method.** The 10.7 kB of hand-crafted DocVQA-2026 category tips and
the engineering-drawing-specific VLM sub-call signature are not
load-bearing; the recursive-perception mechanism is.

Variance note worth surfacing for the paper: minimal σ = 2.21pp,
unified σ = 4.05pp — almost 2× tighter trial-to-trial. Plausibly the
agent stops being yanked between competing per-category prescriptions
when the dispatch guess is wrong. Per-trial table + paired analysis:
`docs/experiments/rvlm-minimal-generality.md`.

Refill notes: 4/8 trials needed pass-2 refill — `science_paper_1`
systematically hits the 4h task_timeout (~50% per-attempt success
rate, agent-loop / vllm-hang specific to this doc). All cleared on
the sequential refill pass.

### `[✓]` unified-tips n=8 val (tasks #25 + #28) — 2026-05-28

run_ids: `rvlm-unified-val-t{1..8}` · **n=8 mean 40.94% ± 4.05pp**
(range 35.0–47.5) on Qwen 3.5 27B local, c=32.

Paired vs amax1's `rvlm` baseline at t1..t7: Δ mean = +0.71pp,
SE 1.72pp, 95% CI [−3.50, +4.93]pp — well inside noise. Per-trial
table + paired analysis: `docs/experiments/unified-category-tips-ablation.md`.

Variance asymmetry worth flagging for the promote-to-default
decision: unified σ=4.05pp vs rvlm σ=2.38pp (~1.7×). Final paired
analysis lands when amax1's t8 commits.

## Decision rules (set in advance)

- **unified-tips Δ ≈ 0pp** → promote unified to default; replace rvlm
  cells with rvlm_unified in subsequent cells.
- **rvlm_ocr ≈ rvlm on val** → OCR neutral on moderate-length docs;
  paper's §B doc-length-axis claim holds. Push to MMLongBench-Doc next.
- **direct_vlm < rvlm by 5+pp** → recursive sub-call is load-bearing;
  paper §C prediction 3 supported.
- **direct_vlm ≈ rvlm** → architecture-agnostic; reframing needed
  (the sub-call may not be the load-bearing piece — context-rationing
  is, regardless of architecture).
