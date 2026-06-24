# Experiment Results — DocVQA 2026 (post-D-006)

Cross-axis summary under the D-006 framing (visual-context-budget
hypothesis). Scores are **mean ± std across trials** (no SC voting in the
headline per D-003 — SC-8 numbers shown only where they anchor an ICDAR
submission). Per-cell detail lives in `docs/experiments/{solver}-{model}.md`.

> **Status / queue / how-to-run:** see `docs/experiment-status.md` (single
> source of truth for what's done, in progress, and queued). Negative-result
> variants not in the table below: **`rvlm_subagent_full`** (full-agent
> sub-call, ≈ subagent at 10× cost — `rvlm_subagent_full-qwen-3_5-27b.md`);
> the model-size / harness × model axis incl. **Gemma** is in
> `harness-axis-summary.md` (synthesis) + the by-model files
> (`qwen-3_5-{4b,9b}.md`, `qwen3-8b.md`, `gemma-4-{e4b,31b}.md`).

> **⚠ All current numbers are from the post-2026-06-01 code** (minimized /
> parity-stripped prompts + per-call `num_retries=5` only; whole-agent
> `@retry` removed). **Pre-change numbers are no longer valid** — the
> prompt scrub and retry-logic change moved them; their writeups are in
> [`archive/experiments/`](../archive/experiments/) and
> [`archive/docs/results.md`](../archive/docs/results.md). Do not cite
> archived numbers as current. The 8-solver Qwen-27B re-run below is
> **complete at n=8** (val, 2026-06-03). The append-only/MDP twin of `rvlm`
> is the **`codeact_chat`** solver (corrected; see below + its per-cell
> doc); the **old dspy `codeact`** budget sweep is **deprecated/archived**
> (its "compaction ~free" finding holds — `archive/experiments/`).
> **`codeact_chat`** (2026-06-12) — the *corrected* codeact (true
> multi-turn chat MDP, no dspy in the loop) — is **done at n=8: 39.53% ±
> 2.83**, i.e. **+2.7pp over old `codeact` and tied with `rvlm`**; a
> thinking ablation (n=7, 37.68%) shows `enable_thinking` gives **no
> gain**. See [`codeact-chat-qwen-3_5-27b.md`](experiments/codeact-chat-qwen-3_5-27b.md).
> Cross-model `codeact_chat` model-axis (no-think): **4b/27b** (4B-LM /
> 27B-VLM) **22.34% ± 3.44 (n=8)** — **+6.7pp** vs old `codeact` (15.66)
> and **+6.1pp** vs `4b-homog` (16.25 ± 2.00, n=8), the perception-budget
> lift from swapping the 4B VLM → 27B VLM under a fixed 4B reasoner
> (supports D-006); **4b-homog** (4B/4B) **16.25% ± 2.00 (n=8)**;
> **9b-homog** (9B/9B) **22.97% ± 2.75 (n=8)** — +1.4pp vs old `codeact` and
> **+4.1pp vs `rvlm` 9b-homog (18.91)**, the corrected MDP loop leads the
> proposed method at 9B homog; **9b/27b** (9B-LM / 27B-VLM) **26.56% ± 4.21
> (n=8)** — +2.3pp vs old `codeact` (24.26), +3.6pp vs `9b-homog` (perception
> lift one rung up); **v3**
> (27B-LM / 9B-VLM) **32.81% ± 3.04 (n=8)** — +2.4pp vs old `codeact` (30.43),
> the reasoner-fixed middle rung (39.53 → 32.81 as VLM 27B→9B);
> **8b/27b** (Qwen3-8B-LM / 27B-VLM) **16.72% ± 3.20 (n=8)** — +7.2pp vs old
> `codeact` (9.50); off-axis (older Qwen3-8B reasoner degenerates into
> page-by-page scans, 32% Unknown), a weak-reasoner point that still clears old
> `codeact`; **gemma-31B** (gemma-4-31B homog) **30.31% ± 2.13 (n=4)** — in line
> with old `codeact` (29.25) and `rvlm` (33.04), after a **stop-token fix** to
> `codeact_chat` (gemma-4 doesn't self-stop → it had hallucinated the whole
> rollout in one turn for a bogus 5%; `_split_first_turn` enforces one action
> per turn). **vs `rvlm`:**
> `codeact_chat` **ties `rvlm` across the model axis, both families** — Qwen-27B
> 39.53 vs 39.38 (+0.15), gemma-31B 30.31 vs 33.04 (−2.7), 4b/27b 22.34 vs 21.09
> (+1.25), 4b-homog 16.25 vs 12.49 (+3.76, borderline); old `codeact` trailed
> `rvlm` at every config, so the corrected MDP loop *catches up* to the proposed
> method at no accuracy cost (it does not beat it). Queued:
> Phase-4 27b/4b ×3 harnesses, dataset axis, test SC-8. A per-cell
> **10-min `exec_timeout`** + clean subprocess-reset (commit `f7f497e`) was
> added to cap the 4B's degenerate per-page `batch_look` scans and recover
> from VLM-saturation doc-drops; t1/t6/t7/t8 of the 4b/27b n=8 ran/resumed
> on the fixed code.

## Official baselines (ICDAR 2026 — external, for context)

| Model | Val | Test |
|---|---|---|
| Gemini 3 Pro | 37.50% | **37.50%** |
| GPT-5.2 | — | 35.00% |
| Gemini 3 Flash | 33.75% | 33.75% |
| GPT-5 Mini | — | 22.50% |

## Method vs baselines — Qwen 3.5 27B, val (current code, **n=8 complete**)

> **⟳ Matrix RE-RUN in progress (2026-06-17).** The table below shows the
> original published `*-cmp-val` numbers, whose per-trial artifacts were
> **deleted on both hosts** (so they had no pass@8/SC@8). They are being
> **re-run** with fresh, retained artifacts + the full metric triple. Recovered
> so far (see [`pass-at-k.md`](pass-at-k.md) for the live table):
> **`rvlm` 41.88% ± 5.79** (pass@8 68.75, SC@8 47.50) — *+2.5pp above the old
> 39.38*; **`rvlm_ocr` 36.56% ± 2.89** (pass@8 67.50) — reproduces old 37.81;
> **`rvlm_nocrop` 35.78% ± 2.31** (pass@8 58.75) — reproduces old 36.88;
> **`react` 27.19% ± 3.19** (pass@8 53.75) — reproduces old 25.16;
> **`raw_vlm_multi` 20.94% ± 1.60** (pass@8 27.50) — reproduces old 20.47;
> **`official` 18.91% ± 1.94** (pass@8 33.75) — reproduces old 17.81;
> **`rlm_ocr` 14.69% ± 2.19** (pass@8 27.50) — reproduces old 13.91;
> **`rvlm_subagent` 36.72% ± 2.75** (pass@8 66.25, SC@8 41.25) — re-rolls −2.5pp
> vs old 39.22 (still in the proposed tier).
> **Matrix re-run complete: 8/9 cells recovered with the full triple.** The 9th,
> `rvlm_hybrid`, **fails at the model's context ceiling** (accepted negative
> result): its extra `display()` channel emits ~163k-token requests on heavy docs,
> **exceeding Qwen 3.5 27B's 131k max context** — per policy that counts as the
> solver failing, not a harness gap. Headline 35.47% ± 4.48 (below `rvlm`) is
> retained as an upper bound; true score under the ceiling is lower, pass@8/SC@8
> unavailable. Conclusion reinforced: `+display` doesn't help **and** blows the
> context budget. See [`pass-at-k.md`](pass-at-k.md).

8-solver comparison re-run (val 25 docs / 80 Qs, `enable_thinking=false`,
local vllm :8927, **n=8**). Mean ± std over 8 trials; Δ vs the `rvlm`
reference (difference of means).

| Group | Solver | Role | Val (n=8) | Δ vs `rvlm` |
|---|---|---|---|---|
| **proposed** | **`rvlm`** | REPL + recursive VLM `batch_look` (OCR-free) | **39.38% ± 1.49** | — |
| corrected twin | **`codeact_chat`** | true multi-turn chat MDP (no dspy in loop); RL-target transcript | **39.53% ± 2.83** | +0.15pp (tied) |
| ablation | `rvlm_ocr_ablation` | + OCR `page_texts` + BM25 `search` | 37.81% ± 3.12 | −1.56pp |
| ablation | `rvlm_hybrid_ablation` | + direct `display()` channel on top of sub-call | 35.47% ± 4.48 | −3.91pp |
| ablation | `rvlm_nocrop_ablation` | `batch_look` by **page index, no crop/zoom** (whole pages only) | 36.88% ± 3.20 | −2.51pp |
| ablation | `rvlm_subagent_ablation` | sub-call generalized to **`batch_subagent`** (any subtask, image optional) | 39.22% ± 3.34 | −0.16pp |
| ablation | `rvlm_rationale` | VLM sub-call returns **answer + observation/uncertainty `[note: ...]`** (every call) | 39.22% ± 2.91 | −0.16pp |
| baseline | `react_baseline` | perception (VLM tools), **no REPL** | 25.16% ± 4.60 | −14.22pp |
| baseline | `direct_vlm` | `display()` pages into own context, no sub-call | 22.34% ± 2.79 | −17.03pp |
| baseline | `raw_vlm_multi_baseline` | raw multi-image, no scaffold | 20.47% ± 1.63 | −18.91pp |
| anchor | `official_baseline` | competition `MASTER_PROMPT`, no scaffold | 17.81% ± 1.86 | −21.56pp |
| **control** | **`rlm_ocr`** | RLM + OCR text, **no vision** (perception modality swap) | **13.91% ± 1.56** | **−25.47pp** |

> **Provisional (not in the table above — n=6, paused):**
> **`rvlm_vsearch`** (OCR-free *visual* retrieval extension — `batch_look`
> + multimodal page-embedding `search` via ColModernVBERT, no OCR) =
> **36.67% ± 2.36 (n=6)**, ≈ `rvlm_ocr_ablation`, ~2.7pp below `rvlm`
> (within combined std → no real difference on moderate val docs, same as
> lexical search). t7/t8 pending. See
> `docs/experiments/rvlm-vsearch-qwen-3_5-27b.md`.

Detail: `docs/experiments/{solver}-qwen-3_5-27b.md` for each row.
`official_baseline` is an external anchor (competition kit prompt +
`max_pages=10` downscale; the strict kit-faithful config is 21.67% ±
1.91). `direct_vlm`'s n=8 required ~7 resumes — `comics_3`/`comics_4`
(dense multi-panel comics) crash its long-image in-context display and
only clear stochastically.

**Headlines (all hold at n=8):**
1. **Three clean tiers, every gap ≫ the std:** visual-recursive
   (`rvlm`/`rvlm_ocr`/`rvlm_hybrid`, 35–39%) ≫ no-recursion
   (`react`/`direct_vlm`/`raw_vlm_multi`, 20–25%) ≫ OCR-only floor
   (`rlm_ocr`, 14%).
2. **OCR-free is decisive (the headline control):** `rlm_ocr` — same
   LeanRLM scaffold as `rvlm` with visual perception swapped for OCR
   text — is the matrix floor, **−25.5pp** below `rvlm`, with
   engineering_drawing & maps 0/10 in **all 8 trials**. Recursive
   *visual* perception does work OCR text cannot replace.
3. **OCR adds nothing on top of vision** (`rvlm_ocr` −1.6pp); the
   **direct display channel is mildly harmful** (`rvlm_hybrid` −3.9pp,
   3× the variance — it destabilizes the agent).
3b. **Cropping is a category-specific lever, not a global one**
   (`rvlm_nocrop` −2.5pp overall): removing crop/zoom (whole-page reads
   only) costs −11.2pp on `engineering_drawing` and −17.5pp on
   `science_poster` — the detail-dense categories where zoom is
   load-bearing — but is ≈ 0 elsewhere. It does **not** raise iteration
   count (11.8 vs `rvlm` 13.0). Detail:
   `docs/experiments/rvlm_nocrop_ablation-qwen-3_5-27b.md`.
3c. **Generalizing the sub-call is harmless but unused** (`rvlm_subagent`
   −0.16pp, dead parity): replacing the perception-only `batch_look` with a
   general `batch_subagent` (delegate any subtask, image optional) doesn't
   move accuracy — because the agent uses it as a perception tool **~99%**
   of the time (only ~1% of delegations are non-visual). A single focused
   perception sub-call already captures the benefit → bounds the necessary
   sub-call interface. Detail:
   `docs/experiments/rvlm_subagent_ablation-qwen-3_5-27b.md`.
4. **Parity prompt is honest:** the competition `official` prompt sits
   *below* our minimized-prompt `raw_vlm_multi` (17.8 vs 20.5) — our
   prompt scrub is not sandbagging the baselines.

## Oracle ceiling & self-consistency (pass@k / SC@k) — diagnostic

Per [D-003](paper/decisions.md) the headline stays **mean ± std** (`avg@1`) and
SC is out of the method framing; these are **analysis-only** numbers. Full
per-cell table, method, and findings: **[`pass-at-k.md`](pass-at-k.md)**
(computed by [`scripts/pass_at_k.py`](../scripts/pass_at_k.py), same binary
DocVQA scorer; incomplete trials dropped so pass@k/SC@k are over the full 80 Qs).

- **pass@k** = oracle (any of the k trials correct); **SC@k** = majority-vote
  the k answers, then score.
- ⚠ **The published Qwen-27B `*-cmp-val` headline matrix (rvlm/react/baselines/
  ablations) has no retained per-trial artifacts** (deleted on both hosts) → its
  pass@k/SC@k are **pending a re-run**. Surviving `rvlm-minimal/-unified/
  -skeletal-val` are earlier *prompt-scrub variants*, not the published runs.

Headline-tier cells with retained artifacts:

| Cell | k | avg@1 (±std) | pass@k | SC@k |
|---|---|---|---|---|
| `codeact-chat` 27B homog (corrected twin) | 8 | 39.53 ± 2.83 | **63.75** | 45.00 |
| `rvlm-vsearch` (OCR-free visual-retrieval ext) | 6 | 36.67 ± 2.58 | 66.25 | 37.50 |
| `codeact-chat` 4B-LM / 27B-VLM | 8 | 22.34 ± 3.44 | 55.00 | 26.25 |
| `codeact-chat` 4B homog | 8 | 16.25 ± 2.00 | 47.50 | 20.00 |
| Gemma-4 31B `rvlm` | 7 | 33.04 ± 4.56 | 60.00 | 42.50 |

**Two findings worth carrying into the paper's analysis** (detail in
`pass-at-k.md`): (1) **large oracle headroom** on strong scaffolds — pass@8
nearly doubles avg@1 (`codeact-chat` 39.5→63.8) → ~25pp recoverable by a
verifier / best-of-n / RL reward model; (2) **pass@k cleanly marks
perception-budget-bound cells** — small reasoner + 27B VLM reaches the answer in
*some* trial on most questions (`rvlm` 4B/27B avg@1 21 but pass@8 56) but can't
land it consistently, whereas Gemma-E4B stays low even at pass@8 (a true
capacity floor).

## Active-perception mechanism (prediction 3)

The matrix above triangulates the mechanism — both halves of the scaffold
are load-bearing and the recursive sub-call is the active ingredient:

| Component dropped | Solver | vs `rvlm` (39.38%) | Reading |
|---|---|---|---|
| Recursive sub-call | `raw_vlm_multi_baseline` (20.47%) | **+18.9pp** | recursive agent↔VLM dominates one-shot multi-image |
| The REPL | `react_baseline` (25.16%) | **+14.2pp** | code REPL is load-bearing (crop/arith/compose) |
| Sub-call (kept pixels) | `direct_vlm` (22.34%) | **+17.0pp** | raw pixels in-context ≠ a focused VLM sub-call |
| All perception (OCR-only) | `rlm_ocr` (13.91%) | **+25.5pp** | swapping visual perception for OCR text collapses the score |

Dropping either half of the scaffold collapses the score: perception
served one-shot instead of via the recursive sub-call (`raw_vlm_multi`,
`direct_vlm`) and perception-without-REPL (`react`) both fall well below
`rvlm`. Adding OCR (`rvlm_ocr`) or a direct image channel (`rvlm_hybrid`)
on top of the OCR-free sub-call buys ≈ 0 → supports the OCR-free
recursive-perception framing.

### Efficiency — agent iterations per question (n=8, val)

Trajectory length (= code/observation steps the agent takes) per
question, aggregated over all 8 trials. `%@cap` = fraction of questions
that hit the iteration cap (a churn signal). Reproduce with
`python scripts/iter_stats.py '<run_id_glob>'`.

`budget` = configured `solver.max_iterations`; the effective per-question
cap is `budget` + a page-bonus (up to +10 on long docs), so observed
iterations can exceed the budget.

| Solver | budget | avg iters/Q | median | %@cap |
|---|---|---|---|---|
| `rvlm` | 25 | 13.0 | 11 | 1% |
| `rvlm_ocr_ablation` | 25 | 12.0 | 10 | 1% |
| `rvlm_hybrid_ablation` | 25 | 18.1 | 16 | 7% |
| `rvlm_nocrop_ablation` | 25 | 11.8 | 9 | 1% |
| `rvlm_subagent_ablation` | 25 | 9.7 | 8 | 1% |
| `react_baseline` | 25 | 5.1 | 4 | 0% |
| `direct_vlm` | 40 | 30.4 | 40 | 59% |
| `rlm_ocr` | 25 | 11.4 | 9 | 2% |
| `raw_vlm_multi_baseline` | — | — (single-shot) | — | — |
| `official_baseline` | — | 1.0 | 1 | 100% |

Efficiency tracks the accuracy story:
- **`rvlm` is efficient and accurate** — ~13 steps/Q, ~never hits the cap.
- **The display channel makes `rvlm_hybrid` churn** — 18.1 steps/Q (+40%
  vs `rvlm`) for a *lower* score: the extra image context destabilizes
  the agent rather than helping (consistent with its 3× variance).
- **`direct_vlm` is pathologically inefficient** — 30.4 steps/Q, **median
  = the cap (40)**, 59% of questions pin the cap. Pixels-in-own-context
  forces the agent to grind without converging — this is why it is the
  run-time bottleneck (each trial dies/resumes 5–9×), and it still scores
  in the no-recursion tier.
- **`react` is shallow** — 5.1 steps/Q: without a REPL it can't compose
  multi-step perception, so it terminates early *and* scores low.

## CodeAct (append-only/MDP twin of `rvlm`) — superseded by `codeact_chat`

The append-only/MDP twin is now the **`codeact_chat`** solver (corrected:
true multi-turn chat MDP, no dspy in the loop) — see
[`codeact-chat-qwen-3_5-27b.md`](experiments/codeact-chat-qwen-3_5-27b.md)
(27B-homog **39.53% ± 2.83**, ties `rvlm`; full model-axis above). The
**old dspy `codeact`** (single-turn `dspy.Predict` re-rendering history into
a `trajectory` string — POMDP-shaped, not a clean RL rollout) is
**deprecated**; its budget sweep `max_iterations ∈ {24,40,56}` is archived
in [`archive/experiments/codeact-qwen-3_5-27b.md`](../archive/experiments/codeact-qwen-3_5-27b.md).
Its one durable finding — **dropping compaction is ~free** (pooled 36.74% ±
4.29, n=23, ~13 iters/Q, cap never binds, within noise of `rvlm`) — is
confirmed and strengthened by `codeact_chat` (which closes the gap to a tie).
Old `codeact` still appears as the **CodeAct harness** in the harness×model
axis (`harness-axis-summary.md`, by-model files), but those rows are now
**STALE** (old dspy) — shown for provenance only and **not to be cited**.
Any finding that rests on them is **provisional** until the `codeact_chat`
re-run lands.

Hold the reasoner fixed, swap **only** the VLM tool backend. n=8 per arm,
val, current code. Detail: `docs/experiments/qwen-3_5-9b.md` and
`qwen-3_5-4b.md` (per-model, all harnesses); `qwen3-8b.md` for the
older-gen point; cross-cutting synthesis in `harness-axis-summary.md`.

| Reasoner (LLM) | v1 homog (VLM = LLM) | v2 mixed (VLM = 27B) | Δ (v2 − v1) | n |
|---|---|---|---|---|
| Qwen 3.5 9B | 16.67% ± 3.40 | 24.54% ± 5.30 | **+7.87pp** — Welch t=3.54, 95% CI [+3.4, +12.3], **sig.** | 8 |
| Qwen 3.5 4B | 12.49% ± 3.74 | 21.09% ± 3.16 | **+8.60pp** — Welch t=4.96, 95% CI [+5.20, +11.99], **sig.** | 8 |
| Qwen3 8B (older gen) | — (n/a, text-only) | 11.73% ± 2.96 | — (off the Qwen3.5 size curve) | 8 |

At both 9B and 4B, swapping only the VLM →27B with the reasoner fixed
lifts ~8pp (9B +7.87, 4B +8.60) → the scaffold is
**perception-budget-bound** for mid/small reasoners (supports D-006).
The lift's consistency across reasoner size is the signature of a
perception (not orchestration) bottleneck.

**Older-generation reasoner point (Qwen3-8B):** with perception fixed
at the 27B VLM, a Qwen3-8B reasoner scores only **11.73% ± 2.96** —
below 4B v2 (21.09%) and half of 9B v2 (24.54%) on the *same* VLM.
Modality is **not** a confound: in v2 the reasoner delegates all
perception to the VLM via `batch_look` and never sees pixels, so
text-only vs multimodal is irrelevant — every reasoner is a text
orchestrator. The only variable vs 9B/4B is **generation** (Qwen3 vs
the Qwen3.5 used for 9B/4B). The older 8B is simply a weaker
orchestrator: it thrashed ~18 RLM iterations/question and force-
submitted wrong. (Bug ruled out — `enable_thinking=false` correctly
applied, tool/parse errors negligible, `batch_look` returns content.)
Reading: even a strong fixed VLM can't rescue a weaker reasoner — a
clean reasoner-*quality* signal, kept off the Qwen3.5 9B↔4B size curve
because it's a different generation. A clean 8B size point would need
**Qwen3.5-8B** (same family). Detail in the experiment writeup.

Homogeneous cross-family **Gemma** model-axis sweep with **per-model
harness-LIFT** (val, all three harnesses + both no-scaffold baselines; **n=8**
unless noted — supersedes the earlier n=1/n=2 pilots). **Baseline =
max(rawvlm, official).**

| Gemma | rvlm | codeactᶜ | react | rawvlm | official | base | best lift |
|---|---|---|---|---|---|---|---|
| **E4B** | 7.34 ± 3.30 | 7.66 ± 1.94 | 6.09 ± 2.36 | 3.75 ± 0.00 | 6.25 ± 1.16 | 6.25 | **+1.4 (n.s.)** |
| **31B** | **32.50 ± 4.48** | 29.25 ± 5.77† | 18.44 ± 3.58 | 10.78 ± 0.93 | 11.09 ± 1.82 | 11.09 | **+21.4** |

† 31B codeact n=5 (stopped early per user); score depressed by slow-doc guards +
gemma4-31B codeact operational instability (8 shm-crashes + degenerate-gen /
max-iter runaways during the sweep) — `rvlm`/`react`/baselines ran clean. An
operational finding in its own right; see `gemma-4-31b.md`.

ᶜ **STALE — do not cite.** Old dspy `codeact` (deprecated). The corrected
**`codeact_chat`** twin is the sole source of truth for CodeAct numbers going
forward; a config without a `codeact_chat` value is **open** — the stale dspy
figure is shown for provenance only, not as a current result. Tracking and
replacements: `codeact-chat-qwen-3_5-27b.md`.

**Cross-family findings.** (1) At **31B every harness clears both no-scaffold
baselines by ≫ the std** — rvlm +21.4, codeact +18.2ᶜ, react +7.4 over base
11.09. rvlm 32.50 ≫ react 18.44 (+14.1pp): the recursive VLM sub-call is
**load-bearing**, mirroring Qwen 27B (rvlm 39.4 ≫ react 25.2) — "recursive-
perception ≫ tool-only ReAct" is **robust across model families**. (2) At
**E4B no harness clears `official_baseline`** (all within ~1 std) — a 4B model
is **too weak to exploit any scaffold**, a clean negative control. **The lift
is a capacity gate**, cleanly bracketed by one model family: sharp at 31B,
absent at 4B. Detail: `gemma-4-e4b.md` + `gemma-4-31b.md` (per-model) +
`harness-axis-summary.md` (cross-family synthesis, Finding 5). (Served
on `vllm/vllm-openai:gemma4`; 31B TP=2 one-trial-at-a-time, inherent shm crash
handled by restart+resume.)

## Document-length axis (prediction 2)

> Prior MP-DocVQA / MMLongBench-Doc cross-benchmark numbers were run on
> pre-change prompts/retry logic and are **invalid** (archived under
> `archive/experiments/mp-docvqa-qwen27b.md`,
> `mmlongbench-doc-qwen27b.md`). The mechanism (lift sign + magnitude
> scale with the benchmark's page-count distribution) is robust, but the
> magnitudes need a current-code re-run before citing. **Pending.**

## Solver taxonomy (engineering names)

| Name | Role |
|---|---|
| `rvlm` | proposed method — REPL + recursive VLM `batch_look` (OCR-free) |
| `rvlm_ocr_ablation` | + OCR `page_texts` + BM25 search (OCR extension) |
| `rvlm_hybrid_ablation` | + direct `display()` image channel on top of the sub-call |
| `direct_vlm` | single multimodal LLM with `display()`, no sub-call (alt angle) |
| `raw_vlm_multi_baseline` | raw multi-image, single VLM call, no REPL |
| `react_baseline` | `dspy.ReAct` + same VLM tools as `rvlm`, no Python REPL |
| `rlm_ocr` | REPL + OCR text + BM25, no vision (text-perception variant) |
| `official_baseline` | competition `MASTER_PROMPT`, multi-image, no scaffold |

Legacy pre-D-010 names (`flat_solo`, `leanest_solo`, `no_loop_multi`,
etc.) appear only in historical run IDs and `archive/`; D-010 doesn't
backfill them.

## Conventions for adding rows

1. When a cell reaches n=3 (or n=8), update its `docs/experiments/`
   file's Summary + Status, then refresh the matrix row here with the
   locked mean ± std.
2. Per D-008: flag the trial count on every number; don't headline an
   n=1 as if locked.
3. If a result triggers a paper-framing decision, add a `decisions.md`
   D-NNN entry.
4. Mark the cell `[✓]` in `coordination/<host>.md` with the run_id and
   one-line result.
