# amax1 queue (throughput host)

amax1 is the throughput host — runs side-track experiments where the
direction is already known and we just need to lock numbers. No adaptive
iteration; if a cell shows an unexpected direction, **halt and append a
`## NOTE FOR AMAX7`** section at the bottom of this file.

**vllm:** brings up its own per-model containers as needed. Set
`MODEL_TAG` per cell.

## In progress

(none — compaction/window experiments concluded 2026-05-30; see Done
"compaction + max_messages findings" below. GPU idle; next step is a
design decision, see NOTE.)

## Queued

### ★ HANDOFF FROM AMAX7 — paper submission path (2026-05-31)

User freed amax7 for other experiments mid-skeletal-refill. amax7 is
finishing its current `rvlm-skeletal-val-t5` refill (5 docs), then
stopping. Everything below moves here.

**Background (read first):**
- Hybrid n=8 done: **paired Δ hybrid−minimal = −5.31pp [−8.92, −1.70]
  significant** (`8113887`). Hybrid shelved (paper: "tried it, didn't
  work").
- Skeletal n=8 incomplete (only t1/t2/t8 clean; t3-t7 had timeout
  dropouts under the 22/25 overlap chain). Current paired Δ on
  intersection: **−1.63pp [−5.67, +2.41]pp t(7)=−0.954 (n.s.)** —
  inside the "minimal as proposed method" band already.
- Minimal n=8 reference: 42.03% ± 2.21pp marginal, σ tighter than
  skeletal (3.92pp) → minimal wins on stability even if mean ties.

**Order:**

1. ~~**Skeletal refill t6 + t7**~~ → **MOVED BACK TO AMAX7**
   (2026-05-31): amax1 can't refill — resumability needs the partial
   run dirs (t6=22/25, t7=23/25) which live on amax7; the rsync
   prerequisite was the blocker. See amax7.md Queued cell #1.
2. ~~**Refined paired Δ skeletal-minimal**~~ → **MOVED TO AMAX7**
   (depends on the refill data, which now lands on amax7). Default
   outcome stays **minimal** as proposed method.
3. **Minimal n=8 test + SC-vote submission** (paper headline cell;
   user uploads JSON to competition server manually) — **amax1 starts
   here.** Gate from step 2 defaults to `rvlm_minimal`; no need to wait
   on amax7's refined Δ (current intersection Δ already lands minimal).
4. **Prompt-minimize other solvers** (post-test gate, see
   amax7.md cell 3)
5. **rvlm_ocr n=1 val** (post-minimization)

#### Step 1: skeletal refill t6 + t7 — `[→amax7]` MOVED BACK TO AMAX7 (2026-05-31)

**Do NOT run on amax1.** Resumability needs the partial run dirs
(t6=22/25, t7=23/25) and they are **local on amax7** — the rsync-to-
amax1 prerequisite was the blocker, so per user direction this refill
returns to amax7 where the data lives. See amax7.md Queued cell #1 for
the (rsync-free) commands.

#### Step 2: refined paired Δ — `[→amax7]` MOVED TO AMAX7 (2026-05-31)

Follows the refill, so it moves with it. amax7 re-runs the per-trial-
intersection paired analysis after its refill and updates its own
"Done" section. Decision rule (unchanged): Δ < +0.5pp or CI95 includes
0 → **minimal**; Δ > +1pp → **skeletal**. Default outcome: **minimal**.

#### Step 3: minimal n=8 test + SC-vote submission — amax1 STARTS HERE

Test set = **48 docs / 160 Qs**. Strict serial, one trial at a
time, c=16:

```bash
for T in 1 2 3 4 5 6 7 8; do
  uv run python evals.py \
    lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
    lm.enable_thinking=false \
    solver=rvlm_minimal \
    data.split=test data.num_samples=null \
    max_concurrency=16 \
    run_id=rvlm-minimal-test-t${T}
done
```
(Or `solver=rvlm_skeletal` if Step 2 lands the skeletal verdict.)

- Trial wall: ~3-4h. Chain wall: ~28-32h.
- Apply long-tail rule: if a trial at ≤8 docs remaining stalls
  ≥45 min, launch the next trial in parallel (vllm has headroom
  for one overlap at c=16).

#### Step 4: SC-vote submission

After all 8 test trials land, write `scripts/sc_vote_submission.py`:
for each `(doc_id, question_id)`, load prediction from each trial's
`output/runs/rvlm-minimal-test-tN/submission.json`, majority-vote
literal answer string, tie-break by trial-with-highest-val-score.

Output: `output/submissions/rvlm-minimal-test-sc8.json`. User
uploads to competition server manually.

#### Steps 5-6 (post-submission)

See `coordination/amax7.md` cells 3 & 4 for prompt-minimization
+ rvlm_ocr. Run on whichever host has time.

---

### ★ SOLVER MINIMIZATION + NAMING CLEANUP (2026-05-31, user-directed)

Principle: **minimal prompts everywhere** (the n=8 ablation proved
hand-crafted per-category tips aren't load-bearing). Keep baselines but
minimize their prompts; remove OUR redundant heavy-prompt method
solvers; rename so minimal is canonical and baselines carry a
`_baseline` suffix.

**DONE — prompt minimization (committed, all stay dataset-aware):**
- Method extensions: `rvlm_full`, `rvlm_ocr` (c82b174)
- Baselines: `raw_vlm_single`, `raw_vlm_multi` (0ea5068);
  `react`, `repl_only` (febbcf8)
- Already-minimal (no change): `rvlm_minimal`, `rvlm_skeletal`,
  `rvlm_naked`, `rvlm_hybrid`, `direct_vlm_minimal`.

**PENDING re-val** (minimized solvers' numbers changed — re-run after
the test chain frees the GPU): `rvlm_full`, `rvlm_ocr`, `react`,
`repl_only`, `raw_vlm_single`, `raw_vlm_multi`. (rvlm_ocr re-val was
already handoff Step 5.)

**DEFERRED — rename + delete (do AFTER: (1) the rvlm_minimal TEST chain
finishes — it references `solver=rvlm_minimal` by name, so renaming now
breaks the cron relaunch; (2) re-val above):**

Rename map:
| current | →                | role |
|---|---|---|
| `rvlm_minimal` | `rvlm`           | proposed method (canonical) |
| `direct_vlm_minimal` | `direct_vlm` | alt-architecture method (see direct_vlm note) |
| `react` | `react_baseline`        | baseline |
| `repl_only` | `repl_only_baseline` | baseline |
| `raw_vlm_single` | `raw_vlm_single_baseline` | baseline |
| `raw_vlm_multi` | `raw_vlm_multi_baseline` | baseline |
| `rvlm_skeletal/naked/hybrid/full/ocr` | (unchanged) | method variants |
| `official_baseline` | (already suffixed) | baseline |

Delete (OUR old heavy-prompt method solvers, superseded by minimal;
git-tag `pre-solver-cleanup` first so ablations stay reproducible):
- `rvlm` (base, heavy category tips) — DELETE; frees the `rvlm` name.
- **OPEN — `rvlm_unified`** (heavy `_UNIFIED_TIPS`; the ablation control
  that proved tips aren't load-bearing). Recommend DELETE (result is in
  `docs/experiments/unified-category-tips-ablation.md` + git tag).
- **OPEN — `rvlm_gepa`** (GEPA prompt-optimization; separate research
  line, not just a heavy duplicate). Recommend KEEP unless retired.
- **OPEN — `direct_vlm` (heavy)**: it holds the il_n + legacy_prompt
  infra (documented in `docs/experiments/direct-vlm-il_n-and-prompt-
  variance.md`). Recommend MINIMIZE IN PLACE (strip its TOOL_HINTS,
  keep the il_n/legacy infra) and DELETE the redundant
  `direct_vlm_minimal` instead — i.e. for direct_vlm the canonical file
  is the existing one minimized, NOT a promote-from-minimal.

Cascade for the rename/delete (one coordinated op): solver `.py` files,
`configs/solver/*.yaml` (filename + `_target_`), any `solver=X` refs in
docs/scripts/coordination, run_id conventions, solver registry/discovery.

↳ **AMAX7 follow-up (do NOT run before this cascade):** after the
rename/delete lands, amax7 removes the whole-agent `@retry` from every
solver (rvlm_minimal already done, `895851f`) so per-call
`num_retries=5` is the only retry layer. **Please append a
`## NOTE FOR AMAX7` when the cascade is committed** so amax7 picks it
up — see amax7.md cell "Remove agent-level @retry from ALL solvers".

---

### images_for_last_n sweep for direct_vlm (replaces old n=2 trials)

User directive 2026-05-29: run `direct_vlm` cap=40 at
`images_for_last_n=1` and `=2` **instead of** the confirmatory n=2
trials (old R4 minimal-t2 / R5 direct_vlm-t2 are CANCELLED). Motivation:
R3 (il_n=3 default) crashes on long comics docs — the solver
`display()`s images into its own context and exceeds vllm's 64-image
limit. Trimming kept-image-context should fix the crash; the sweep also
measures the accuracy effect.

- **R4** `direct-vlm-iln1-val-iter40-t1`: `solver=direct_vlm
  solver.max_iterations=40 solver.images_for_last_n=1`
- **R5** `direct-vlm-iln2-val-iter40-t1`: `solver=direct_vlm
  solver.max_iterations=40 solver.images_for_last_n=2`

Comparison: R3 (il_n=3, crashes on comics) vs R4 (il_n=1) vs R5 (il_n=2),
all direct_vlm cap=40. Driven by the cron chain (afae64f5).

### 3. `[ ]` Gemma 4 E4B baseline + scaffold n=1 val (task #8 part 1)

Re-run on clean prompts (D-009). Original 2026-05-09 cells used
pre-scrub prompts. Direction is robust (+5.83pp lift in original n=3);
just locking the magnitude under clean prompts.

```bash
# Bring up vllm for Gemma 4 E4B (see existing setup notes in
# docs/experiments/gemma-4-e4b-baseline-scaffold.md)
TRIALS=1 SOLVER=rvlm bash scripts/run_gemma_chain.sh gemma-4-e4b-vllm-local 4-e4b
```

- Expected wall: ~2-3h (baseline + scaffold)
- Expected direction: lift sign preserved (~+5pp baseline → scaffold)

### 4. `[→amax7]` Qwen 3.5 9B model-axis (task #8 part 2) — CLAIMED BY AMAX7

**Do NOT run on amax1.** Per user direction 2026-05-31 18:38, amax7
took the 9B model-axis work. amax7 is running `rvlm_minimal` (the
paper method) directly in two VLM variants rather than the
baseline+scaffold chain below — see amax7.md "In progress":
- `rvlm-minimal-3_5-9b-val` (homogeneous: 9B LLM + 9B VLM)
- `rvlm-minimal-9b-llm-27b-vlm-val` (9B LLM + 27B VLM)

amax7 has the 9B vllm up at :8909 and the 27B at :8928, so it's the
natural host. If a clean baseline lift (`raw_vlm_multi` 9B) is still
wanted afterward, re-file it then.

Original chain (kept for reference, do not run here):
```bash
TRIALS=1 SOLVER=rvlm bash scripts/run_gemma_chain.sh qwen-3_5-9b-vllm-local 3_5-9b
```

- Expected wall: ~2-3h
- Expected direction: lift sign preserved (~+6pp from original n=3)

### 5. `[ ]` Gemma 4 31B baseline + scaffold n=1 val (task #8 part 3)

```bash
# Per docs/experiments/gemma-4-31b-baseline-scaffold.md: needs
# vllm --tensor-parallel-size 4 --enforce-eager to survive scaffold load
TRIALS=1 SOLVER=rvlm bash scripts/run_gemma_chain.sh gemma-4-31b-vllm-local 4-31b
```

- Expected wall: ~3-4h
- Expected direction: lift sign preserved (~+25pp from original n=3)

## Done

### ★ COMPACTION + max_messages findings (2026-05-30)

Two attempts to fix the direct_vlm >64-image comics crash WITHOUT the
il_n-truncation accuracy loss. Both via `direct_vlm` cap40, Qwen 27B.

**Attempt 1 — compaction prompt + strict per-step discipline** (commit
`af9ca2b`): exposed `RESET_HISTORY` + "1–2 images/step, note-then-move".
Result (il_n=3, c=8, n=3 started): **no crash through comics, but agent
NEVER called RESET_HISTORY (0)**, and the "1–2/step" rule made it
**~5× slower** (nibbling to the iteration cap on every question).
Killed before completion — too slow, compaction unused.

**Attempt 2 — `max_messages` window, removed `images_for_last_n`**
(commit `6bdf27b`): `VisualREPLHistory` now renders only the last
`max_messages=8` REPL steps (text+images), drops older; relaxed
per-step rule; strengthened "compact OFTEN". Run
`dvm-compact-v2-val-t1` (c=16): **q-weighted 33.3% (24/25 docs, 78q)**,
`BadReq64=1` (a single CAUGHT 64-image warning, NOT a process crash —
comics_2/4 completed), speed recovered (~57 q/h), **RESET_HISTORY still
0** (agent ignores it even with strong prompting). Run hung on the last
doc (business_report_1, iteration 35/40) — killed; verdict taken from
the 24 done docs.

**direct_vlm cap40 accuracy ladder (rough — subsets/prompts vary, n=1):**
| config | acc | crash? |
|---|---|---|
| il_n=3 (all-text + last-3-images) | 43.2% | crashes |
| **max_messages=8 (last-8-steps total)** | **33.3%** | no crash ✓ |
| il_n=2 | 28.9% | no |
| il_n=1 | 23.7% | no |

**KEY FINDINGS:**
1. **Prompt-motivated compaction does NOT work** — Qwen never calls
   `RESET_HISTORY` despite it being in the tool list + 3 guidelines
   saying "compact OFTEN". Two independent confirmations. If compaction
   is wanted, it must be **code-enforced** (auto-compact), not prompted.
2. **`max_messages` window is WORSE than `images_for_last_n` for
   accuracy** (33.3% vs 43.2%) because it drops old **TEXT** too, not
   just images. il_n kept ALL step text (the agent's accumulated notes)
   + only truncated images; the window discards reasoning history, so a
   non-compacting agent loses its notes → accuracy falls.
3. **Right design (recommendation):** keep ALL step TEXT (like il_n) +
   hard-cap total IMAGES to <64 (most-recent ~56, across steps). Full
   reasoning context (→ il_n=3-level accuracy) + hard crash safety (no
   >64 ever). This is the original "smarter total-image management"
   idea; the message-window overcorrected by throwing away text.

### ★ larger max_messages sweep — does a bigger window recover accuracy? (2026-05-30)

Testing the user's question "why not larger max_messages?" — if the
window drops TEXT and that's what cost accuracy (mm8=33.3% vs il_n=3
43.2%), a bigger window keeps more text and should climb back. Both
`direct_vlm` cap40, c=12, new-prompt build 6bdf27b, val 25docs/80q.
Note: with the new build the >64-image error is now **caught
per-question** (logged `Direct-VLM failed for Q ...` WARNING → that one
Q returns PRED=Unknown → run continues), NOT a process crash.

- **`dvm-mm40-val-t1` (max_messages=40): DIED — non-viable.** Process
  gone at docs=2/25, q=20, after ~63 min wall (vs mm24's 9 docs in the
  same time → ~4.5× slower). BadReq64=4 (all caught per-Q, not the
  crash). No fatal Python traceback at the tail (only logfire/otel
  telemetry-export noise); no OOM line accessible (308G RAM free at
  check, but swap 1/1G exhausted). Root cause is the window size itself:
  at mm=40 each step carries up to 40 prior steps of accumulated images
  → huge prompts → generation crawls AND the agent nibbles to the
  40-iter cap per question (seen hitting 30/40, 37/40). The death is
  secondary; the disqualifier is that mm=40 is pathologically slow.
  **Do not relaunch.** Confirms the window can't safely go this high.
- **`dvm-mm24-val-t1` (max_messages=24): FINISHED — 35.0% (28/80),
  clean 25/25 docs.** BadReq64=8, all caught per-Q (run finished cleanly,
  no crash). Wall ~3h21m (18:08→21:30) — ~1.5–2× slower than the il_n
  runs (~1.5–2h), the larger window's token cost. Comics partially
  scored despite caught overflows: 3/10 comics-Q correct
  (comics_1_q1, comics_2_q2, comics_3_q1), 4 lost to PRED=Unknown
  (comics_2_q3, comics_3_q2, comics_4_q1, comics_4_q2 — the caught
  64-image cases), 3 wrong on content.

**VERDICT — larger max_messages does NOT recover accuracy.**

| config | acc | crash/finish |
|---|---|---|
| il_n=3 (all-text + last-3-images) | **43.2%** | crashes on comics |
| max_messages=40 | — | **DIED** (too slow, 2/25 in 63min) |
| max_messages=24 | **35.0%** | finishes, ~3h21m |
| max_messages=8 | 33.3% | finishes |

Going 8→24 buys only **+1.7pp** (33.3→35.0), still **−8.2pp below
il_n=3's 43.2%**, and the bigger window costs ~1.5–2× wall time; mm=40
can't even finish. The sliding window is the wrong knob: it discards
the agent's accumulated TEXT notes (which a non-compacting agent can't
rebuild), and bigger windows just trade that loss for crippling
slowness. **Confirms recommendation (a): keep ALL step TEXT, hard-cap
only total IMAGES to <64.** That's the path to il_n=3-level accuracy +
crash safety without the window's text-loss or speed penalty.

### ★ il_n=3 under NEW prompts — the regression is the PROMPT (2026-05-31)

Ran the BEST config (il_n=3 = keep ALL text + last-3 images) for the
first time under the current relaxed look-then-note prompts, now
crash-safe (the >64-image error is caught per-question). Goal: does
the best config reach ~43% under the new build?

- **`dvm-iln3-newprompt-val-t1`** (max_messages=10000, images_for_last_n=3,
  cap40, NEW prompt): **Overall 27.6% (21/76)**; **non-comics subset
  20/70 = 28.6%**; comics 1/10. BadReq64=1 (caught, no crash). Wall
  ~4h04m. 24/25 docs (one doc dropped at finish).
- vs **old-prompt il_n=3 = 43.2%** (32/74, comics-excluded subset):
  **−14.6pp on the fair non-comics subset** — so the gap is NOT a
  comics/denominator artifact. The best config REGRESSED ~15pp under
  the new prompt.

**Why it's the prompt, proven by diff (not hand-waving).** Diffed the
saved solver `.py` of the old 43.2% run (`direct-vlm-val-iter40-t1`)
vs the new build. Identical: REPL-history rendering (old
`images_for_last_n=3` ≡ new `max_messages=10000`+`images_for_last_n=3`,
verified line-by-line), `max_image_pixels=8000000`, cap=40,
`use_category_tips`. **Only substantive change = the prompt**: the new
build added "CONTEXT IS A SLIDING WINDOW / old images drop off",
"COMPACT OFTEN (RESET_HISTORY)", "LOOK-THEN-NOTE / once noted the image
has done its job", and changed tables from "display each strip" →
"one strip per step". For a `max_messages=10000` config NOTHING drops
off, so the prompt is (a) factually false about the env and (b)
coaching the agent toward fewer-images/note-then-move — the OPPOSITE
of il_n=3's strength (accumulating visual context; il_n sweep showed
accuracy monotonic in retained images 1<2<3).

**CONFIRMING RUN DONE — verdict: it's n=1 NOISE, not a clean prompt
regression (2026-05-31).** `dvm-iln3-legacyprompt-val-t1` (byte-exact
OLD prompt, everything else matched, new build): **31.2% (25/80)**,
non-comics **34.3% (24/70)**, BadReq64=0 (never overflowed, clean
25/25), comics 1/10, wall ~3h31m.

3-way ladder (all n=1):

| il_n=3 | full | non-comics |
|---|---|---|
| old build, OLD prompt (`direct-vlm-val-iter40-t1`) | 43.2% (32/74) | 42.9% (30/70) |
| new build, NEW prompt | 27.6% (21/76) | 28.6% (20/70) |
| new build, LEGACY prompt (byte-exact old) | 31.2% (25/80) | 34.3% (24/70) |

Matched non-comics, legacy vs new-prompt: **34.3% vs 28.6% = +5.7pp**.

**What this actually shows (correcting two earlier over-reads):**
- The prompt is NOT the clean ~15pp lever I first claimed. Reverting it
  recovered only **+5.7pp** (matched non-comics), leaving **−8.6pp**
  still short of the old 43.2% run.
- But it's also NOT "exact tie / prompt irrelevant" (an artifact of a
  14-doc partial). On the full set legacy beats new-prompt by ~5-6pp.
- **At n=1 NONE of this is significant.** Per-run SE on ~70Q ≈ 5.5pp;
  difference SE ≈ 7.7pp. The +5.7pp prompt lift is <1 SE; the −8.6pp
  residual to old is ~1.1 SE. The new runs (27.6/31.2/34.3) all sit
  near il_n=2's old 28.9% — **the old 43.2% is most parsimoniously a
  high single draw.** We built a whole prompt theory on one old n=1
  point vs one new n=1 point without first checking reproducibility
  (violating the project's own "~3pp std, run 3+ trials" rule).
- Code surfaces are exonerated: identical rendering, `multimodal.py`
  image pipeline UNCHANGED (38-line diff is all max_messages logic),
  identical params, served model still `Qwen/Qwen3.5-27B`.

**Resolution — n=3 DONE (2026-05-31): il_n=3 is HIGH-VARIANCE; no
defensible regression.** Ran current-build legacy-prompt il_n=3 n=3.
Non-comics (clean 70Q):

| trial | non-comics | full |
|---|---|---|
| legacy t1 | 34.3% | 31.2% |
| legacy t2 | 31.4% | 30.3% |
| legacy t3 | **40.0%** | 38.2% |
| new-prompt t1 | 28.6% | 27.6% |
| **OLD-orig (the "43.2%")** | **42.9%** | 43.2% |

current-build legacy **n=3 = 35.2% ± 4.4pp (sample SD), range
31.4–40.0%**. The spread is WIDE (~9pp across 3 trials), NOT the tight
cluster t1+t2 alone suggested. Old-orig 42.9% sits **~1.75 SD above the
current mean**, and the best current trial (t3=40.0%) is within 2.9pp of
it → **43% is a plausible high draw of the same distribution. No
statistically significant regression** (1.75σ, n=1-vs-n=3).

**Meta-lesson (logged honestly):** this thread whipsawed 3× —
"prompt is the lever" → "tie, it's luck" → "tight n=3, maybe real
regression" → (with t3) "wide variance, no clear regression." Each flip
came from reading a verdict off 1–2 trials of a solver whose true SD is
~4–5pp. The project rule "~3pp std, run 3+ trials before claiming"
existed for exactly this; il_n=3's variance is even higher than that.
**Bottom line: direct_vlm il_n=3 ≈ 35% ± 4–5pp (non-comics); the old
43% was a high draw, not a regression.**

**FINAL — old-build rerun DONE (2026-05-31): NO regression, case
closed.** `oldbuild-iln3-val-t1` (worktree /tmp/dvo @ f737190, faithful
old code + old deps) = **33.8% full / 35.7% non-comics**, clean 25/25.
The old build did NOT reproduce 43.2% — it landed mid-distribution.

All 6 il_n=3 trials, non-comics 70Q (both builds):

| trial | build | non-comics |
|---|---|---|
| OLD-orig ("the 43.2") | f737190 | 42.9% |
| OLD-rerun | f737190 | 35.7% |
| legacy t1 | current | 34.3% |
| legacy t2 | current | 31.4% |
| legacy t3 | current | 40.0% |
| new-prompt t1 | current | 28.6% |

**Pooled: mean 35.5% ± 5.3pp.** The old build's OWN two trials span
42.9→35.7 (7pp) — so 42.9% was the high tail of the old build's own
variance, not a build effect. Old 2-trial mean 39.3% vs current 4-trial
mean 33.6% → 5.7pp ≈ 1.3σ, **not significant**. (Faint hint the old
build is marginally higher, but unestablishable at this n and not worth
chasing.)

**CASE CLOSED: `direct_vlm` il_n=3 ≈ 35% ± 5pp (non-comics). The 43.2%
was a high-variance single draw, NOT a regression.** The entire
"il_n=3 regressed" thread was an artifact of reading verdicts off
single high-variance trials — exactly what "run 3+ trials" prevents.
Actionable downstream: il_n=3 (or legacy prompt) is a fine crash-safe
direct_vlm default at ~35%; no code/dep bisect needed. Worktree /tmp/dvo
can be removed.

### NOTE: open design decision for user
Pick the image-bounding mechanism for direct_vlm: **(a)** total-image
hard cap keeping all text (recommended — likely ~43% + no crash), **(b)**
code-enforced auto-compaction, or **(c)** accept max_messages=8 @ 33.3%.
`images_for_last_n` is removed; current default is `max_messages=8`.
**Update:** `images_for_last_n` is RE-ADDED (a452d28); il_n=3 is now
crash-safe via the per-question catch — option (a) may be as simple as
"il_n=3 + legacy prompt" pending the confirming run.

### ★ CHAIN COMPLETE — il_n sweep cross-run summary (2026-05-30)

**(i) Iteration-budget effect (R1 cap20 vs R2 cap40):** ≈flat /
slightly down (27.5%→21.6%); 58-62 q hit the cap at BOTH budgets.
**Iteration budget is NOT the bottleneck** — the agent doesn't
converge at any budget.

**(ii) images_for_last_n sweep (direct_vlm cap40) — the main result:**
| il_n | crash on comics? | accuracy |
|---|---|---|
| 3 (R3) | **YES** (>64 imgs, process killed; comics_2/4 unscoreable) | 43.2% (32/74) |
| 2 (R5) | no (comics_2 just timed out) | 28.9% (22/76) |
| 1 (R4) | no | 23.7% (18/76) |

- **Crash fix confirmed:** il_n≤2 eliminates the >64-image crash
  (BadReq64=0, runs complete; comics_2 merely times out like the
  minimal solver).
- **Accuracy is MONOTONIC in il_n:** 1 < 2 < 3. R4 vs R5 is a clean
  same-76q compare → **il_n=2 beats il_n=1 by +5.2pp** (22 vs 18 / 76).
  il_n=3 (43.2%) is higher still (different/smaller 74q subset + n=1,
  but the gap is broad across categories, not a subset artifact).
- **The tension (key takeaway):** the setting with the best accuracy
  (il_n=3, more visual context) is exactly the one that crashes on long
  docs. Truncating kept-images (il_n=1/2) trades ~14-20pp of accuracy
  for stability. **A proper fix needs smarter image management** (cap
  TOTAL images / summarize old crops), not just last-N truncation —
  flag for the solver. n=1 each; directionally solid (monotone + clean
  R4/R5 pair).

**(iii) prompt/architecture (R2 minimal vs R3 full, same 74q, equal
cap):** full prompt **+21.6pp** (43.2 vs 21.6) — OPPOSITE of rvlm
(where minimal≈unified). Recursive arch robust to prompt
minimization; direct arch depends on prompt scaffolding. (rvlm
headline is cap20 — equal-cap caveat noted.)

**Model-axis cells #3-5 NOT run** (deferred, see NOTE) — chain ended
here; GPU idle awaiting user decision.

### R5. `[✓]` direct_vlm n=1 val @ cap=40, images_for_last_n=2 — 2026-05-30

run_id `direct-vlm-iln2-val-iter40-t1`. **Overall 28.9% (22/76)** —
`comics_2` timed out (24/25; 76q, same subset as R4). 47 cap40 hits,
**BadReq64=0** (no crash). Per-cat: infographics 80%, science_poster/
slide 30%, business_report/eng_drawing/maps/science_paper 20%,
comics 1/6, maps 1/10. Clean +5.2pp over R4 (il_n=1) on the identical
76q → confirms accuracy rises with kept-image count.

### R4. `[✓]` direct_vlm n=1 val @ cap=40, images_for_last_n=1 — 2026-05-30

run_id `direct-vlm-iln1-val-iter40-t1`. **Overall 23.7% (18/76)** —
`comics_2` task_timed-out (only doc missing; 76 not 80). **57 cap40
hits. `BadRequestError:64 images` = 0 — il_n=1 FIXED the comics
crash** (`comics_2` merely timed out like the minimal solver; no
process kill). Per-cat: infographics 60%, science_paper 40%,
science_poster 30%, business_report/slide 20%, eng_drawing 10%,
**maps 0/10, comics 0/6**.

**★ FINDING — the crash-fix has a big accuracy cost; il_n is a real
knob.** il_n=1 (23.7%) vs il_n=3 (R3, 43.2%) ≈ **−20pp**. Mechanism is
clear in the per-cat: il_n=1 **collapses on maps (0%) and comics (0%)**
— the multi-region / multi-panel categories where you must hold
several crops in context simultaneously to compare across them.
Keeping only the last iteration's image destroys that. So:
- il_n=3: best accuracy (43.2%) but CRASHES on the longest comics docs
- il_n=1: no crash, but accuracy collapses on multi-image reasoning (23.7%)
- il_n=2 (R5, running): the candidate sweet spot — does it keep the
  crash-fix while recovering maps/comics accuracy?
(Subsets differ slightly: R3 74q misses comics_2+comics_4; R4 76q
misses only comics_2. The ~20pp gap is far larger than that overlap
difference, so the direction is solid. n=1 each.)

### R3. `[✓ PARTIAL]` direct_vlm n=1 val @ cap=40, il_n=3 (task #19) — 2026-05-30

run_id `direct-vlm-val-iter40-t1`. **23/25 docs (74 q); `comics_2` &
`comics_4` UNSCOREABLE** — `direct_vlm` at il_n=3 crashed 4× on those
two long comics docs (>64-image / sandbox kill; 3 `BadRequestError:
64 images` logged). Resumed across attempts (16→21→23 docs). 47 cap40
hits. **q-weighted accuracy over the 74 completed q = 43.2%.**
Per-cat: infographics/science_poster/slide 60%, eng_drawing/science_paper
40%, business_report 30%, comics 50%(2/4), maps 10%.

**★ HEADLINE FINDING — full prompt is hugely load-bearing for the
DIRECT architecture (opposite of rvlm).** R3 and R2 cover the
**identical 74-q subset** (both miss only comics_2/comics_4), so paired:
| cap40, same 74 q | prompt | score |
|---|---|---|
| R2 `direct_vlm_minimal` | stripped | 21.6% (16/74) |
| R3 `direct_vlm` | full (TOOL_HINTS+tips) | **43.2% (32/74)** |
**Δ ≈ +21.6pp** from the full prompt — far beyond n=1 noise (~3pp).
This is the OPPOSITE of the rvlm family, where minimal ≈ unified
(prompt content NOT load-bearing, Δ=+1.09pp n.s.). Coherent story:
the **recursive (VLM-sub-call) architecture is robust to prompt
minimization; the direct (display-into-own-context) architecture
depends heavily on prompt scaffolding.** Caveats: n=1 each, partial
74/80 (internally valid — identical subset), single model (Qwen 27B).
Worth a paired n≥2 confirmation before it goes in the paper, but the
magnitude is striking. (Also note R3's 43.2% > R2's 21.6% > R1's
27.5%cap20 — direct+full-prompt is the strongest direct-VLM config so
far on this subset.)

### R2. `[✓]` direct_vlm_minimal n=1 val @ cap=40 (task #34) — 2026-05-29

run_id `direct-vlm-minimal-val-iter40-t1`, c=24, `max_iterations=40`.
**Overall 21.6% (16/74)** — per-cat: infographics 60%, science_paper
30%, science_poster/slide 20%, business_report/eng_drawing/maps 10%,
**comics 0% (0/4)**.

**⚠ INCOMPLETE DENOMINATOR (74, not 80).** `comics_2` and `comics_4`
each hit the 4h `task_timeout` (14400s) at 18:47Z and errored out
(6 questions never scored). So 21.6%/74 is NOT directly comparable to
R1's 27.5%/80. On the shared 74 questions the comparison is still
roughly flat-to-down.

**Headline finding (R1 vs R2 — iteration-budget effect): doubling the
cap did NOT help.** 58/74 questions still hit the 40/40 cap (vs 62/80
hitting 20/20 in R1), and the score did not rise. The agent does not
converge at *either* budget — it loops/wanders to whatever ceiling it
is given. **R1's low score was therefore NOT cap-truncation**; the
iteration budget was not the bottleneck. This weakens the original
"cap=20 was binding" hypothesis and shifts the read toward solver
behavior / task difficulty. NOTE for the chain: R4/R5 (more cap=40
trials) are now lower-value given this — flag for user on return.

Operational: logfire/otlp telemetry threw protobuf DecodeErrors
throughout (harmless to scoring). comics docs are pathological for the
direct-VLM family at cap=40 (4h timeout) — R3/R4/R5 will likely share
the <80 denominator problem.

### R1. `[✓]` direct_vlm_minimal n=1 val @ cap=20 (task #34) — 2026-05-29

run_id `direct-vlm-minimal-val-t1`, c=24, default `max_iterations=20`.
**Overall 27.5% (22/80).** Per-category: slide 60%, infographics 50%,
science_paper 30%, business_report/comics/science_poster 20%,
engineering_drawing/maps 10%. Wall ~96min.

**Key signal: 62/80 questions (78%) hit the 20/20 iteration cap.** The
low score is therefore confounded — cannot attribute to prompt
content until the cap is relaxed. Hence the cap=40 re-run (R2) and the
decision to run the whole direct-VLM family at cap=40. NO direct_vlm
baseline comparison yet (R3 will provide it at equal cap). Logfire
telemetry threw protobuf DecodeErrors throughout (broken span export)
but did NOT affect scoring/submission — harmless.

### Autonomous chain plan (R1→R5, all on the 27B @ 8927; user away 2026-05-29)

Strictly sequential on local vllm; driven by heartbeat cron. No vllm
swaps (safe while unattended).
- R1 `direct-vlm-minimal-val-t1` cap=20 — ✓ 27.5%
- R2 `direct-vlm-minimal-val-iter40-t1` cap=40 — ✓ 21.6%/74
- R3 `direct-vlm-val-iter40-t1` (direct_vlm, il_n=3 default) cap=40 — running (crashes on comics)
- R4 `direct-vlm-iln1-val-iter40-t1` (direct_vlm, **images_for_last_n=1**) cap=40
- R5 `direct-vlm-iln2-val-iter40-t1` (direct_vlm, **images_for_last_n=2**) cap=40

(R4/R5 changed 2026-05-29 from the original n=2 confirmatory trials to
the images_for_last_n sweep, per user "instead of more trials".)

Reads: R1-vs-R2 = iteration-budget effect (≈flat → budget not the
bottleneck); R3 vs R4 vs R5 = images_for_last_n sweep (crash-fix +
accuracy effect of trimming kept image-context); R2(minimal) vs
direct_vlm runs at equal cap = prompt/architecture; rvlm headline is
cap=20 (flagged in amax7.md).

### NOTE: model-axis cells #3-5 DEFERRED (need attention on return)

Not run unattended. Reasons in
`tmp/workspace/amax1-model-axis/vllm-bringup-notes.md`: (1) per-model
vllm bringup is unscripted and Gemma-4 needs an undocumented tool-call
parser (wrong parser → silent tool-call failure → fake-low scores);
(2) #5 Gemma-31B as-written needs TP=4 but amax1 has only 3 GPUs. The
vllm `docker run` template + per-model ports are in that file.

### A. `[✓]` ReAct baseline n=8 val — REPL-vs-no-REPL ablation

Finished 2026-05-29T01:53. c=24, `lm.timeout=1800` (overridden via the
new `LMConfig.timeout` field), run_ids `react-val-t1..t8`. **n=8 mean
= 30.47%, std 3.06pp, range 25.0–33.75%.** Paired vs rvlm n=8: **Δ =
−10.47pp, 95% CI [−13.42, −7.52]pp** (cleanly outside ±1.5pp noise).
Paired vs rvlm_unified n=8: **Δ = −10.47pp, 95% CI [−14.46, −6.48]pp**.
Lands in "≤ −5pp → REPL is load-bearing" per the pre-set decision
table. Largest per-category gaps (rvlm−react): `business_report`
(+23.8pp), `engineering_drawing` (+23.8pp), `comics`/`science_poster`
(+13–14pp) — exactly the zoom-then-read categories where the REPL's
`pages[i].crop()` matters. Operational notes: 3 `litellm.Timeout`s on
original chain (`science_poster_2`/`business_report_2`), recovered via
2 backfill passes; the second backfill exposed that the 600s default
timeout doesn't fit ReAct's long-trajectory completions, fixed via
config-driven `LMConfig.timeout` (commit `3acee78`). Full per-trial
tables, paired Δ stats, per-category breakdown, and the timeout
calibration story in [docs/experiments/react-baseline.md](../docs/experiments/react-baseline.md).

### 1. `[✓]` rvlm n=8 val — paired-comparison anchor for unified-tips (task #28)

Finished 2026-05-28T16:45 (c=24, run_ids `rvlm-val-t1`..`rvlm-val-t8`).
**n=8 mean = 40.94%, std 2.29pp, range 36.25%–43.75%.** Paired vs amax7
`rvlm_unified` n=8: **Δ = 0.00pp** (both arms 262/640 total correct),
95% CI [−3.91, +3.91]pp — lands cleanly in "promote unified to default"
per the pre-set decision table. Operational note: t6→t7 transition hit
a same-`run_id` contamination incident (chain's auto-t7 launched in
parallel with a standalone t7); recovered via clean restart on
`science_paper_1`. Runner's silent-timeout fallback was changed to
record-error-and-retry mid-chain (commit `8309710`). Full per-trial
tables, paired Δ stats, contamination story, and silent-failure audit
in [docs/experiments/unified-category-tips-ablation.md](../docs/experiments/unified-category-tips-ablation.md).

## Decision rules (set in advance)

- **All three cells: lift sign preserved** → model-axis prediction 1 is
  robust under clean prompts. Escalate to n=2 (then n=8 only after the
  paper headline locks).
- **Any cell: lift sign reverses** → halt this queue and `## NOTE FOR
  AMAX7`. Pre-scrub prompts were inflating the lift differentially
  somehow; need to investigate before drawing model-axis conclusions.
- **Magnitude shifts by 5+pp** → expected (per-trial std ~3-4pp on
  Qwen 27B; smaller models can be noisier). Note in the result line;
  still escalate to n=2.
