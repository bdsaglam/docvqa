# Prompt-scrub audit (val→test generalization gap)

## Hypothesis

Hand-tuned per-category tips in `src/docvqa/prompts.py::CATEGORY_TIPS`
had been written by inspecting val-set failures and contained two
classes of content that could inflate val without helping test:

1. **Val-leak phrases** — verbatim entity names ("The Man with 1000
   Faces", "Pantheon", part numbers `1901060-021` / `AN 910-2`) and
   verbatim val question phrasings ("first words up to the first
   comma", "last word on page X", "page before X").
2. **Tool-reference verbs** — `search()`, `page_texts`, "thorough
   search". These steer flat-solver path use BM25 + OCR text but were
   being injected into leanest's prompt too (leanest has only
   `batch_look`), where they read as dead references.

The 13pp val-test gap on Qwen 3.5 27B flat_solo (51.2% val → 39.0%
test SC-8) was the primary symptom motivating the audit.

## v1 — single-pass scrub (commit f4f0cfd, 2026-05-14)

Removed both classes of content in one pass from `CATEGORY_TIPS` and
`BASELINE_CATEGORY_TIPS`. `ANSWER_FORMATTING_RULES` left intact (it
comes from the official competition guideline).

Ran matched n=8 across {flat_solo, leanest_solo} × {val, test}:

```bash
# Per-trial command (i ∈ 1..8, split ∈ val,test, solver ∈ flat_solo,leanest_solo)
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local \
  vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=$solver \
  solver.rlm_type=lean \
  data.split=$split data.num_samples=null \
  max_concurrency=24 \
  run_id=$solver-$split-scrub-t$i
```

Chain scripts: `scripts/run_prompt_scrub_chain.sh` (base n=4) +
`scripts/run_prompt_scrub_test_t5to8.sh` (extension to n=8).

### v1 results — locked, ICDAR-scored where applicable

**Val (locally scored, ANLS):**

| Solver | Scrubbed n=8 mean ± std | Pre-scrub n=8 mean | Δ per-trial | Scrubbed SC-8 (local) | Pre-scrub SC-8 (local) | Δ SC-8 |
|---|---|---|---|---|---|---|
| flat_solo | 40.62% ± 2.67pp | 44.7% ± 2.6pp | **−4.08pp** | **45.0%** | 51.2% | **−6.20pp** |
| leanest_solo | 42.81% ± 4.42pp | ~43.8% peak | **−1.0pp** | **48.8%** | (no SC-8 baseline) | n/a |

**Test (ICDAR-scored SC-8 only):**

| Solver | Scrubbed SC-8 | Pre-scrub SC-8 | Δ |
|---|---|---|---|
| flat_solo | **37.0%** | 39.0% | **−2.0pp** |
| leanest_solo | **39.0%** | 36.0% | **+3.0pp** |

**Val→test gap (SC-8 to SC-8, matched):**

| Solver | Pre-scrub gap | Scrubbed gap | Improvement |
|---|---|---|---|
| flat_solo | 51.2 − 39.0 = **12.2pp** | 45.0 − 37.0 = **8.0pp** | **−4.2pp** gap narrowed |
| leanest_solo | n/a | 48.8 − 39.0 = 9.8pp | — |

Submissions: `submissions/{flat,leanest}-solo-{val,test}-scrub-{t1..t8,sc4,sc8}.json`.

### v1 conclusions

1. **The audit hit its primary objective on flat_solo:** val→test gap
   narrowed by 4.2pp. Performance redistributed away from val-only
   overfitting toward test generalization.
2. **Leanest was a pure win:** +3pp on test, val SC-8 lifted to the
   best of any cell (48.8%). The val-leak removal helped leanest
   strictly because leanest had been collateral damage to the
   tool-reference verbiage anyway.
3. **Flat_solo paid a 2pp test cost** alongside the gap narrowing.
   The tool-reference removal (`search()`/`page_texts`) was over-broad
   for the solver that *actually* has those tools. The val-leak
   scrub was net-positive on flat too, but the tool-verb scrub
   subtracted from it.
4. **Solver ordering on val SC-8 reversed:** leanest 48.8% > flat
   45.0%. Pre-scrub, flat dominated. Tells us how much of flat's val
   lead came from val-leak items vs from genuine search-tool routing.
5. **SC voting headroom shrank on flat:** SC-8 minus per-trial mean
   was +6.5pp pre-scrub, only +4.4pp post-scrub on val; SC-8 = SC-4 =
   37% on test. Scrubbed prompts produce more correlated answers;
   less diversity → less voting gain. Consistent with the
   interpretation that val-leak items had been giving the agent
   *diverse* correct answers on aliased questions.

## v2 — split scrub: keep val-leak removal, restore flat tool steerage

**Hypothesis**: re-introducing the `search()` / `page_texts` tool
routing verbiage only for solvers that *have* those tools recovers
flat_solo's 2pp test loss without re-introducing val bias. Leanest
stays unchanged (the val-leak scrub alone was beneficial for it).

**Implementation** (this commit):

- `src/docvqa/prompts.py`:
  - `CATEGORY_TIPS` unchanged from v1 (val-leak-scrubbed,
    tool-verb-free). Continues to be the prompt source for **leanest**
    via `get_category_tips`.
  - New `FLAT_SOLO_TOOL_HINTS` dict — per-category overlay restoring
    tool-routing verbiage for `science_paper`, `slide`, `infographics`
    (the categories where flat-solver BM25 + OCR routing dominates).
  - New `get_flat_solo_category_tips(category)` — returns
    `CATEGORY_TIPS[cat] + FLAT_SOLO_TOOL_HINTS[cat]`.
- `flat_solo_solver.py`, `flat_batch_solver.py`, `lean_solo_solver.py`
  switched from `get_category_tips` → `get_flat_solo_category_tips`
  (these all expose `search()`).
- `leanest_solo_solver.py` and DA variants left as-is.

**Predictions to validate / refute:**

| Cell | v1 result | v2 prediction |
|---|---|---|
| flat_solo val SC-8 | 45.0% | hold ~45% or climb toward 46-48% (extra tool verbiage might also help val) |
| flat_solo test SC-8 | 37.0% | climb to 38-40% (recover the lost 2pp) |
| leanest_solo val/test | 48.8% / 39.0% | unchanged (leanest's prompt path is the same as v1) |

Refutation criteria — v2 fails if **flat_solo test SC-8 stays at
37% or drops**, meaning the val-leak removal alone is what cost flat
the 2pp (not the tool-verb removal). In that case, the audit's hard
lesson is "leanest gained, flat broke even" and the v2 path doesn't
help.

### v2 chain command (for the runner on the other server)

```bash
# Run from repo root after `git pull` lands v2 commits.
# Single tmux session, sequential 16-trial chain on the other server.
tmux new-session -d -s prompt-scrub-v2 -n chain
mkdir -p logs/prompt-scrub-v2

# Phase A: flat_solo val ×8 (locally scorable, fast feedback)
# Phase B: flat_solo test ×8 (ICDAR submission target)

for split in val test; do
  for i in 1 2 3 4 5 6 7 8; do
    uv run python evals.py \
      lm=qwen-3_5-27b-vllm-local \
      vlm=qwen-3_5-27b-vllm-local \
      lm.enable_thinking=false \
      solver=flat_solo solver.rlm_type=lean \
      data.split=$split data.num_samples=null \
      max_concurrency=24 \
      run_id=flat-solo-$split-scrubv2-t$i 2>&1 \
      | tee logs/prompt-scrub-v2/flat-solo-$split-scrubv2-t$i.log
  done
done

# Then SC-8 vote:
uv run python scripts/vote_submissions.py \
  --runs $(for i in 1 2 3 4 5 6 7 8; do echo output/runs/flat-solo-val-scrubv2-t$i; done) \
  --output submissions/flat-solo-val-scrubv2-sc8.json

uv run python scripts/vote_submissions.py \
  --runs $(for i in 1 2 3 4 5 6 7 8; do echo output/runs/flat-solo-test-scrubv2-t$i; done) \
  --output submissions/flat-solo-test-scrubv2-sc8.json

# Submit submissions/flat-solo-test-scrubv2-sc8.json to ICDAR
# (per-trial JSONs in submissions/flat-solo-test-scrubv2-t{1..8}.json if desired).
```

**Expected wall time** (per local-box reference run):
- Val phase: 8 × ~50min = ~7h
- Test phase: 8 × ~3-5h = ~24-40h (long-tail docs `maps_5`,
  `business_report_4`, `business_report_10` dominate per-trial cost)
- **Total: ~30-48h.** Notification cadence: phase boundaries.

### What about leanest? (decision: skip)

Leanest's prompt path is unchanged by v2 (still uses
`get_category_tips` → `CATEGORY_TIPS`). The v1 numbers (val SC-8 48.8%,
test SC-8 39.0%) are the final leanest scrubbed numbers; no re-run
needed. If a future v3 hypothesis specifically targets leanest, that's
a separate experiment.

## Cross-experiment context — full flat_solo tips spectrum on test SC-8

The no-tips ablation (commit `e590c0d`,
`docs/experiments/flat-solo-category-tips-off.md`) ran flat_solo with
`use_category_tips=false`, n=8, same Qwen 3.5 27B setup. Combined
with v1, we now have a 3-point spectrum on test SC-8:

| Variant | flat_solo test SC-8 | Δ vs full | leanest test SC-8 | Δ vs pre-scrub |
|---|---|---|---|---|
| **full tips** (pre-scrub) | **38.75%** | — | 36.00% (m=40) | — |
| **v1 scrubbed** (val-leak + tool-verb removed) | 37.00% | −1.75 | **39.00%** | **+3.00** |
| **no tips** (all category tips off) | 35.00% | −3.75 | — | — |

Reading: **tips do contribute to flat_solo test**, just less than they
contribute to val. The 4pp test delta (full − no-tips = 38.75 − 35.00)
is in the same direction as the 6pp val delta — directionally
consistent generalization, not val overfitting. This refutes the
strongest version of the "tips memorize val" hypothesis. The audit's
v1 cost on flat (−1.75 to −2pp) sits in between full and no-tips,
roughly proportional to how much was removed.

**For leanest the story is opposite:** the scrub net-helped (+3pp
test). The val-leak entity removals + the dead tool-verb removals
were both hurting leanest in distinct ways.

This is exactly the pattern v2 tries to exploit — separate the two
concerns and route the helpful tool-verb context only to solvers that
have those tools.

Pre-scrub leanest m=40 SC-8 test 36.0% comes from
`docs/experiments/leanest-test-matched-baseline.md` (commit
1c2325e, 2026-05-14).

## Status

**v1: done.** All numbers locked, both ICDAR-scored test SC-8 figures
confirmed by user-side submissions on 2026-05-19. Submission JSONs
+ chain scripts committed in `a0c4436`.

**v2: code landed, chain not yet launched on the v2 server.** Ready
for the runner to `git pull` and execute the chain block above.

## Open questions for follow-up

- Does flat_solo's `look()` only / `batch_look()` only (drop `search`,
  drop `page_texts`) variant beat both v1 and v2? — an even cleaner
  ablation of the OCR contribution on flat_solo.
- Does the SC-voting-headroom shrinkage transfer to test? (We only
  observed it on val SC-8 because test has no local GT for per-trial
  variance analysis.)
- Should `BASELINE_CATEGORY_TIPS` (used by `no_loop_*`) get its own
  v2-style overlay, or is the single-shot VLM context too narrow for
  tool-routing verbiage to matter? Current scrubbed baseline tips are
  fine for the no_loop baselines.
