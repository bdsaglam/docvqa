# Experiment Results — DocVQA 2026

Generated: 2026-04-25

## Summary

| Config | LLM | VLM | Val Score | Test Score |
|--------|-----|-----|:---------:|:----------:|
| Flat Solo SC-8 | Qwen 3.6 27B | Qwen 3.6 27B | **51.2%** | **43.75%** |
| Flat Solo SC-8 | Qwen 3.5 27B | Qwen 3.5 27B | 51.2% | 39.0% |
| Flat Solo | Qwen 3.5 27B | Qwen 3.5 27B | 48.8% | 35.6% |
| Flat Solo mean±std (8 trials) | Qwen 3.5 27B | Qwen 3.5 27B | 44.7±2.6% | — |
| Flat Solo mean±std (8 trials) | Qwen 3.6 27B | Qwen 3.6 27B | 44.1±3.0% | — |
| Flat Solo | Qwen 3.6 35B-A3B | Qwen 3.5 27B | 36.2% | — |
| Flat Solo | Gemini 3.1 Pro| Gemini 3 Flash | — | **59.4%** |
| Flat Batch (Pro+Flash) | Pro 3 | Flash | 55.0% | — |
| Flat Batch (v4fix) | Pro 3.1 | Qwen 3.5 27B | 50.0% | — |

## Official Baselines

| Model | Overall |
|-------|:-------:|
| Gemini 3 Pro | **37.50%** |
| GPT-5.2 | 35.00% |
| Gemini 3 Flash | 33.75% |
| GPT-5 Mini | 22.50% |

## Key Findings

- **Solo >> Batch**: ~10pp gap — one question at a time is much better
- **Flat Solo lean+nothink is best Qwen/Qwen**: 51.2% val (SC-8), 48.8% val peak single, 39% test (SC-8)
- **SC-8 voting**: +6.6pp over individual mean on val, +2.5pp over best single run
- **Val 8-trial stats**: mean 44.7%, std 2.6%, range 40.0-48.8%
- **rlm_type matters for batch**: lean >> code (+4-6pp) for batch solvers
- **rlm_type for solo**: lean+nothink best, code+think comparable (40.0%)
- **Thinking hurts lean solo**: 38.8% (think) vs 41.6% mean (nothink)
- **High variance**: std ~2.6% across 8 trials — voting stabilizes results
- **Budget**: m25 slightly better than m20 for flat solo
- **Qwen 3.6 35B LLM underperforms 27B**: 36.2% (1 trial, flat solo lean+nothink) vs 27B's 41.6% mean / 46.2% best — larger LLM does not help here, maps regressed to 0%

## Detailed Results — All Qwen/Qwen Runs (Val Set)

### Solo Solvers

| Solver | rlm_type | think | Trials | Mean | Best | Worst |
|--------|----------|-------|:-------:|:----:|:----:|:-----:|
| **Flat Solo (m25)** | **lean** | false | 3 | **41.6%** | **46.2%** | 37.5% |
| Flat Solo (m20) | lean | false | 3 | 40.4% | 43.8% | 36.2% |
| Lean Solo | lean | false | 3 | 40.0% | 42.5% | 37.5% |
| Flat Solo (m25) | code | true | 1 | 40.0% | 40.0% | 40.0% |
| Flat Solo (m25) | lean | true | 1 | 38.8% | 38.8% | 38.8% |

### Batch Solvers

| Solver | rlm_type | think | Trials | Mean | Best | Worst |
|--------|----------|-------|:-------:|:----:|:----:|:-----:|
| RLM Lean Batch | lean | true | 3 | 35.4% | 37.5% | 31.2% |
| RLM Lean Batch | lean | false | 3 | 35.0% | 36.2% | 33.8% |
| RLM Code Batch | code | false | 3 | 31.2% | 32.5% | 30.0% |
| RLM Code Batch | code | true | 3 | 29.6% | 33.8% | 26.2% |

### All 21 Runs (ranked)

| # | Run | Solver | LLM | VLM | Score |
|:--:|-----|--------|-----|-----|:-----:|
| 1 | solo-m25-t1-val | Flat Solo | Qwen 27B | Qwen 27B | **37/80 = 46.2%** |
| 2 | solo-m20-t1-val | Flat Solo | Qwen 27B | Qwen 27B | 35/80 = 43.8% |
| 3 | lean-solo-val-t2 | Lean Solo | Qwen 27B | Qwen 27B | 34/80 = 42.5% |
| 4 | solo-m20-t2-val | Flat Solo | Qwen 27B | Qwen 27B | 33/80 = 41.2% |
| 5 | solo-m25-t2-val | Flat Solo | Qwen 27B | Qwen 27B | 33/80 = 41.2% |
| 6 | lean-solo-val-local | Lean Solo | Qwen 27B | Qwen 27B | 32/80 = 40.0% |
| 7 | solo-m25-t3-val | Flat Solo | Qwen 27B | Qwen 27B | 30/80 = 37.5% |
| 8 | lean-solo-val-t3 | Lean Solo | Qwen 27B | Qwen 27B | 30/80 = 37.5% |
| 9 | rlm-lean-thinktrue-t2-val | Flat Batch | Qwen 27B | Qwen 27B | 30/80 = 37.5% |
| 10 | rlm-lean-thinktrue-t3-val | Flat Batch | Qwen 27B | Qwen 27B | 30/80 = 37.5% |
| 11 | solo-m20-t3-val | Flat Solo | Qwen 27B | Qwen 27B | 29/80 = 36.2% |
| 12 | rlm-lean-thinkfalse-t1-val | Flat Batch | Qwen 27B | Qwen 27B | 29/80 = 36.2% |
| 13 | rlm-lean-thinkfalse-t3-val | Flat Batch | Qwen 27B | Qwen 27B | 28/80 = 35.0% |
| 14 | rlm-lean-thinkfalse-t2-val | Flat Batch | Qwen 27B | Qwen 27B | 27/80 = 33.8% |
| 15 | rlm-code-thinktrue-t1-val | Flat Batch | Qwen 27B | Qwen 27B | 27/80 = 33.8% |
| 16 | rlm-code-thinkfalse-t1-val | Flat Batch | Qwen 27B | Qwen 27B | 26/80 = 32.5% |
| 17 | rlm-lean-thinktrue-t1-val | Flat Batch | Qwen 27B | Qwen 27B | 25/80 = 31.2% |
| 18 | rlm-code-thinkfalse-t3-val | Flat Batch | Qwen 27B | Qwen 27B | 25/80 = 31.2% |
| 19 | rlm-code-thinkfalse-t2-val | Flat Batch | Qwen 27B | Qwen 27B | 24/80 = 30.0% |
| 20 | rlm-code-thinktrue-t2-val | Flat Batch | Qwen 27B | Qwen 27B | 23/80 = 28.7% |
| 21 | rlm-code-thinktrue-t3-val | Flat Batch | Qwen 27B | Qwen 27B | 21/80 = 26.2% |

## Per-Category Breakdown

| Category | solo-m25-t1 | solo-m20-t1 | lean-solo-t2 | solo-m20-t2 | solo-m25-t2 | lean-solo-local | solo-m25-t3 | lean-solo-t3 | lean-batch-tt2 | lean-batch-tt3 | solo-m20-t3 | lean-batch-tf1 | lean-batch-tf3 | lean-batch-tf2 | code-batch-tt1 | code-batch-tf1 | lean-batch-tt1 | code-batch-tf3 | code-batch-tf2 | code-batch-tt2 | code-batch-tt3 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| business_report | 6/10 | 4/10 | 4/10 | 6/10 | 6/10 | 5/10 | 4/10 | 4/10 | 5/10 | 3/10 | 4/10 | 3/10 | 3/10 | 3/10 | 2/10 | 5/10 | 3/10 | 3/10 | 3/10 | 4/10 | 4/10 |
| comics | 5/10 | 4/10 | 4/10 | 4/10 | 1/10 | 5/10 | 4/10 | 5/10 | 0/10 | 0/10 | 2/10 | 2/10 | 1/10 | 2/10 | 0/10 | 3/10 | 0/10 | 3/10 | 0/10 | 0/10 | 0/10 |
| engineering_drawing | 6/10 | 6/10 | 7/10 | 5/10 | 5/10 | 5/10 | 6/10 | 5/10 | 5/10 | 7/10 | 3/10 | 4/10 | 7/10 | 5/10 | 7/10 | 4/10 | 6/10 | 5/10 | 4/10 | 4/10 | 4/10 |
| infographics | 8/10 | 8/10 | 6/10 | 7/10 | 6/10 | 6/10 | 5/10 | 5/10 | 7/10 | 7/10 | 5/10 | 8/10 | 7/10 | 6/10 | 6/10 | 2/10 | 6/10 | 7/10 | 6/10 | 6/10 | 3/10 |
| maps | 1/10 | 0/10 | 0/10 | 1/10 | 1/10 | 2/10 | 0/10 | 0/10 | 0/10 | 0/10 | 1/10 | 0/10 | 0/10 | 1/10 | 1/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| science_paper | 2/10 | 3/10 | 2/10 | 3/10 | 4/10 | 2/10 | 3/10 | 2/10 | 3/10 | 3/10 | 4/10 | 3/10 | 1/10 | 1/10 | 3/10 | 3/10 | 2/10 | 1/10 | 3/10 | 2/10 | 3/10 |
| science_poster | 4/10 | 5/10 | 5/10 | 2/10 | 4/10 | 3/10 | 4/10 | 5/10 | 5/10 | 6/10 | 5/10 | 5/10 | 5/10 | 5/10 | 5/10 | 5/10 | 3/10 | 2/10 | 5/10 | 3/10 | 4/10 |
| slide | 5/10 | 5/10 | 6/10 | 5/10 | 6/10 | 4/10 | 4/10 | 4/10 | 5/10 | 4/10 | 5/10 | 4/10 | 4/10 | 4/10 | 3/10 | 4/10 | 5/10 | 4/10 | 3/10 | 4/10 | 3/10 |
| **Overall** | **37/80 = 46.2%** | **35/80 = 43.8%** | **34/80 = 42.5%** | **33/80 = 41.2%** | **33/80 = 41.2%** | **32/80 = 40.0%** | **30/80 = 37.5%** | **30/80 = 37.5%** | **30/80 = 37.5%** | **30/80 = 37.5%** | **29/80 = 36.2%** | **29/80 = 36.2%** | **28/80 = 35.0%** | **27/80 = 33.8%** | **27/80 = 33.8%** | **26/80 = 32.5%** | **25/80 = 31.2%** | **25/80 = 31.2%** | **24/80 = 30.0%** | **23/80 = 28.7%** | **21/80 = 26.2%** |

### Best per-category (across all runs)

| Category | Best Score | Runs |
|----------|:----------:|------|
| infographics | **80%** | solo-m25-t1-val, solo-m20-t1-val, rlm-lean-thinkfalse-t1-val |
| engineering_drawing | **70%** | lean-solo-val-t2, rlm-lean-thinktrue-t3-val, rlm-lean-thinkfalse-t3-val, rlm-code-thinktrue-t1-val |
| business_report | **60%** | solo-m25-t1-val, solo-m20-t2-val, solo-m25-t2-val |
| science_poster | **60%** | rlm-lean-thinktrue-t3-val |
| slide | **60%** | lean-solo-val-t2, solo-m25-t2-val |
| comics | **50%** | solo-m25-t1-val, lean-solo-val-local, lean-solo-val-t3 |
| science_paper | **40%** | solo-m25-t2-val, solo-m20-t3-val |
| maps | **20%** | lean-solo-val-local |

## Qwen 3.6 35B LLM Variant (2026-04-22)

Single trial replacing Qwen 27B LLM with Qwen 3.6 35B-A3B, keeping Qwen 27B as VLM. Co-hosted on 3×A100 80GB via separate vLLM servers; GPU memory stable at ~71–73 GB throughout (no OOM). Total runtime ~2h at `max_concurrency=3`.

| Config | Score | vs 27B mean (3 trials) | vs 27B best single |
|--------|:-----:|:----:|:----:|
| Flat Solo, lean, nothink, m30 | 29/80 = 36.2% | 41.6% (−5.4pp) | 46.2% (−10.0pp) |

### Per-Category

| Category | 35B LLM (t1) | 27B best single | Δ |
|----------|:-------:|:---:|:---:|
| engineering_drawing | 7/10 = 70% | 7/10 = 70% | 0 |
| comics | 5/10 = 50% | 5/10 = 50% | 0 |
| business_report | 4/10 = 40% | 6/10 = 60% | −2 |
| infographics | 4/10 = 40% | 8/10 = 80% | −4 |
| science_paper | 3/10 = 30% | 4/10 = 40% | −1 |
| science_poster | 3/10 = 30% | 6/10 = 60% | −3 |
| slide | 3/10 = 30% | 6/10 = 60% | −3 |
| maps | 0/10 = 0% | 2/10 = 20% | −2 |

Run ID: `flat-solo-qwen35b-val-t1`. Biggest regressions on infographics (−40pp), science_poster (−30pp), slide (−30pp); ed and comics held even.

## Qwen 3.6 27B (LLM + VLM) — 8 Trials (2026-04-23 to 2026-04-25)

8 independent flat-solo runs with Qwen 3.6 27B serving both LLM and VLM roles
(single vLLM instance at `localhost:8928`, lean RLM, no thinking, m30, c=4).

### Per-Trial

| Trial | Score | Correct | Wall |
|-------|:-----:|:-------:|------|
| t1 | 40.0% | 32/80 | 2h 24m |
| t2 | 42.5% | 34/80 | 2h 32m |
| t3 | 47.5% | 38/80 | 2h 05m |
| t4 | 47.5% | 38/80 | 2h 14m |
| t5 | 47.5% | 38/80 | 1h 55m |
| t6 | 43.75% | 35/80 | 2h 16m |
| t7 | 42.5% | 34/80 | 2h 46m |
| t8 | 41.25% | 33/80 | 3h 04m |

**Mean: 44.06% | Std: 3.04pp | Range: 40.0–47.5 | Total: 282/640**

### SC-8 Voting (majority vote with `_clean_text` clustering)

- **Val: 51.2% (41/80)** — matches the 3.5 27B SC-8 val exactly.
- **Test: 43.75% (70/160)** — **+4.75pp over 3.5 27B** (39.0%) on the held-out test set.

| Category | SC-8 | Best single | 3.5 27B SC-8 |
|----------|:----:|:-----------:|:------------:|
| engineering_drawing | **70%** | 70% (t2/t3/t4) | 70% |
| business_report | 60% | 50% (t3) | 70% |
| infographics | 60% | 70% (t2) | 70% |
| science_poster | 60% | 50% (t3) | 60% |
| comics | 50% | 60% (t3) | 60% |
| slide | 50% | 50% (t1/t3) | 60% |
| science_paper | 40% | 50% (t2) | 30% |
| maps | 20% | 20% (t1) | 30% |

Voting lifted accuracy by **+7.1pp** over the per-trial mean and **+3.7pp** over the best
single trial — comparable to the 3.5 27B SC-8 lift (+6.5pp / +2.4pp).

### Notes

- 3.6 27B per-trial mean (44.1%) is +3.7pp over 3.5 27B (40.4% on lean+nothink m20).
- After SC-8 voting, 3.6 and 3.5 land on the same overall score (51.2%) but 3.6 trades
  off some categories (business_report, comics, slide, maps regress 10pp each) for
  gains on science_paper (+10pp) — net neutral.
- maps remains the floor (20%) for both. Spatial path-tracing isn't fixed by the model
  upgrade.
- Submissions: `submissions/flat-solo-3_6-27b-val-sc8.json`, `submissions/flat-solo-3_6-27b-test-sc8.json`.
- Voting script: `scripts/vote_submissions.py`.
- Test trial wall: ~66h total (8 × ~8h, ranging 7h02 to 8h53).

## No-Loop Baseline (Qwen 3.5 27B) — 3 Trials (2026-05-08)

Direct VLM Q&A — single forward pass per question, no REPL, no agent loop. All
pages stacked vertically into one composite image (capped at 16384px tall) and
sent in one VLM call. Implements ablation row "no_loop" (raw model point) for
the experiment plan's claim 3 (component contributions). VLM at vllm 8928,
`solver=no_loop`, `lm.enable_thinking=false`, `max_concurrency=16`, val=80q.

### Per-Trial

| Trial | Score | Correct | Wall |
|-------|:-----:|:-------:|------|
| t1 | 20.00% | 16/80 | ~12 min |
| t2 | 15.00% | 12/80 | ~12 min |
| t3 | 16.25% | 13/80 | ~12 min |

**Mean: 17.08% | Sample std: 2.60pp | Range: 15.00–20.00%**

### Per-Category (mean over 3 trials)

| Category | t1 | t2 | t3 | Mean | Scaffold mean (8-trial 3.5) |
|----------|:--:|:--:|:--:|:----:|:----:|
| business_report | 0/10 | 0/10 | 0/10 | 0.0/10 | — |
| comics | 2/10 | 2/10 | 2/10 | 2.0/10 | — |
| engineering_drawing | 2/10 | 2/10 | 2/10 | 2.0/10 | — |
| infographics | 5/10 | 4/10 | 3/10 | 4.0/10 | — |
| maps | 2/10 | 0/10 | 0/10 | 0.7/10 | — |
| science_paper | 0/10 | 0/10 | 0/10 | 0.0/10 | — |
| science_poster | 3/10 | 3/10 | 4/10 | 3.3/10 | — |
| slide | 2/10 | 1/10 | 2/10 | 1.7/10 | — |
| **Overall** | **20.0%** | **15.0%** | **16.25%** | **17.08%** | **44.7%** |

### Notes

- **Scaffold lift (matched within-model, Qwen 3.5 27B):**
  scaffold 44.7% (Flat Solo lean+nothink m=20, 8-trial mean) − no_loop 17.08%
  = **+27.6pp**. Strong signal that the agent loop / OCR / VLM-tool channels
  contribute (claim-3 risk-rank #2 cleared on n=1; n=3 confirms).
- **business_report and science_paper at 0% across all 3 trials.** Long docs
  (105–181 pages, 19–44 pages) get crushed by the 16384px composite-image cap;
  per-page resolution drops to ~80–150px tall after rescaling — far below
  legibility for text answers. This is a structural limit of the no-loop
  baseline, not a tunable.
- **comics and engineering_drawing at exact 2/10 every trial.** Suggests the
  raw VLM gets a deterministic subset of "easy" questions right and loses the
  rest the same way each time — low entropy in the wrong-answer distribution.
- **infographics is the best category (4.0/10 mean).** Single-page,
  text-heavy layouts survive the composite path; this is where raw VLM is
  closest to the scaffolded agent.
- **maps high variance on a small base.** t1 hit 2/10, t2/t3 0/10 — std on 10
  questions is large; not a real signal.
- Variance (2.60pp std) is comparable to the scaffold's 8-trial std (~2.6pp),
  so the lift is not noise.
- Run dirs: `output/runs/no-loop-val-{t1,t2,t3}/`.

## Leanest Solo (OCR-off ablation, Qwen 3.5 27B) — 3 Trials (2026-05-08)

`solver=leanest_solo`. Agent has the VLM `look()` tool and the REPL but **no
OCR `page_texts` and no BM25 `search()`**. The agent must rely on visual
perception only — implements the "OCR on/off" ablation row from the
experiment plan (claim 3, component contributions). All else equal:
Qwen 3.5 27B for both LLM and VLM via vllm 8928, `lm.enable_thinking=false`,
`max_concurrency=16`, `max_iterations=25`, val=80q.

### Per-Trial

| Trial | Score | Correct | Wall |
|-------|:-----:|:-------:|------|
| t1 | 40.00% | 32/80 | ~1h 05m |
| t2 | 40.00% | 32/80 | ~1h 16m |
| t3 | 40.00% | 32/80 | ~1h 00m |

**Mean: 40.00% | Sample std: 0.00pp | Range: 40.00–40.00%**

The exact-tie across 3 trials at 32/80 is a coincidence on overall — per
category the trials differ noticeably (e.g. infographics 80/50/80, business_report
20/50/50, maps 0/10/10). The variance is real, just not at the aggregate level
on n=3.

### Per-Category (mean over 3 trials)

| Category | t1 | t2 | t3 | Mean |
|----------|:--:|:--:|:--:|:----:|
| business_report | 2/10 | 5/10 | 5/10 | 4.0/10 |
| comics | 4/10 | 4/10 | 3/10 | 3.7/10 |
| engineering_drawing | 6/10 | 6/10 | 4/10 | 5.3/10 |
| infographics | 8/10 | 5/10 | 8/10 | 7.0/10 |
| maps | 0/10 | 1/10 | 1/10 | 0.7/10 |
| science_paper | 4/10 | 3/10 | 4/10 | 3.7/10 |
| science_poster | 3/10 | 3/10 | 2/10 | 2.7/10 |
| slide | 5/10 | 5/10 | 5/10 | 5.0/10 |
| **Overall** | **40.0%** | **40.0%** | **40.0%** | **40.00%** |

### Component contributions (so far, Qwen 3.5 27B val)

| Stage | Mean | Δ vs prior |
|-------|:----:|:----------:|
| no_loop (no agent, no OCR, no VLM-tool) | 17.08% (n=3) | — |
| leanest_solo (+ agent loop + VLM `look()` tool) | 40.00% (n=3) | **+22.92pp** |
| flat_solo (+ OCR `page_texts` + BM25 `search()` + category tips + VLM cropping) | 44.7% (n=8, existing) | **+4.70pp** |

**Headline takeaway for ablation table:**
- The recursive VLM sub-call + agent loop channel does the bulk of the lift
  (+22.9pp).
- The OCR symbolic-context channel + the remaining flat_solo features (tips,
  cropping) together add another +4.7pp on top — meaningful but secondary.
- Caveat: the leanest→flat_solo delta is not a clean "OCR on/off" — it also
  includes `use_category_tips` and `vlm_cropping`. To isolate OCR alone we
  would need a flat_solo variant with `page_texts/search` removed but tips and
  cropping kept. (The plan explicitly accepts leanest_solo as the OCR-off
  proxy; flagging the conflation for the paper writeup.)

### Notes

- Wall ≈ 1h per trial at c=16, ~3-4× the no_loop trial wall. Agent does
  multiple VLM `look()` calls per question (max_iterations=25).
- maps remains a floor (0.7/10 mean) — VLM can read map labels but not
  trace paths, consistent with the flat_solo finding.
- Run dirs: `output/runs/leanest-solo-val-{t1,t2,t3}/`.

## Turn-Budget Endpoint — Flat Solo m=5 (Qwen 3.5 27B) — 3 Trials (2026-05-08)

`solver=flat_solo solver.max_iterations=5`, all other defaults (lean RLM,
nothink, OCR + tips + cropping). Defensive lower-end of the turn-budget
curve. Qwen 3.5 27B for both LLM and VLM via vllm 8928, c=16, val=80q.

### Per-Trial

| Trial | Score | Correct | Wall |
|-------|:-----:|:-------:|------|
| t1 | 30.00% | 24/80 | ~28 min |
| t2 | 30.00% | 24/80 | ~25 min |
| t3 | 30.00% | 24/80 | ~25 min |

**Mean: 30.00% | Sample std: 0.00pp | Range: 30.00–30.00%**

Same exact-tie phenomenon as leanest_solo. Per-category t1/t2/t3 differs
(business_report 60/50/50, comics 20/50/40, infographics 60/50/50,
science_paper 20/10/20, slide 50/30/40), but overall lands at 30.00% all three.

### Updated turn-budget curve (Qwen 3.5 27B val, lean+nothink, full tools)

| max_iterations | mean | std | n | range |
|---|---|---|---|---|
| **5** | **30.00%** | **0.00pp** | **3** | **30.0** |
| 10 | 41.41% | 3.16pp | 8 | 36.2–45.0 |
| 20 | 40.94% | 3.32pp | 8 | 36.2–46.2 |
| **30** | **44.69%** | **2.81pp** | **8** | **40.0–48.8** |
| 40 | 40.78% | 3.89pp | 8 | 35.0–47.5 |

m=5 is ~11pp below m=10 — the agent doesn't have enough turns to finish its
work. Combined with the existing curve, the turn-budget figure now spans
m∈{5, 10, 20, 30, 40} with the peak clearly at m=30. Run dirs:
`output/runs/flat-solo-m5-val-{t1,t2,t3}/`.
