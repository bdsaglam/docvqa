# Experiment Results — DocVQA 2026 (Val Set)

Generated: 2026-04-09

## Summary by Solver (Qwen/Qwen, val set)

### Solo Solvers (one question at a time)
| Solver | rlm_type | think | Trials | Mean | Best | Worst |
|--------|----------|-------|--------|------|------|-------|
| **Flat Solo (m25)** | **lean** | false | 3 | **41.6%** | **46.2%** | 37.5% |
| Flat Solo (m20) | lean | false | 3 | 40.4% | 43.8% | 36.2% |
| Lean Solo | lean | false | 3 | 40.0% | 42.5% | 37.5% |
| Flat Solo (m25) | code | true | 1 | 40.0% | 40.0% | 40.0% |
| Flat Solo (m25) | lean | true | 1 | 38.8% | 38.8% | 38.8% |

### Batch Solvers (all questions together)
| Solver | rlm_type | think | Trials | Mean | Best | Worst |
|--------|----------|-------|--------|------|------|-------|
| RLM Lean Batch | lean | true | 3 | 35.4% | 37.5% | 31.2% |
| RLM Lean Batch | lean | false | 3 | 35.0% | 36.2% | 33.8% |
| RLM Code Batch | code | false | 3 | 31.2% | 32.5% | 30.0% |
| RLM Code Batch | code | true | 3 | 29.6% | 33.8% | 26.2% |

### Key Findings
- **Solo >> Batch**: ~10pp gap — one question at a time is much better
- **Flat Solo lean+nothink is best**: 46.2% peak, beats Pro baseline by 8.7pp
- **rlm_type matters for batch**: lean >> code (+4-6pp) for batch solvers
- **rlm_type for solo**: lean+nothink best (46.2%), code+think comparable (40.0%) — pending more code+nothink trials
- **Thinking hurts lean solo**: 38.8% (think) vs 41.6% mean (nothink)
- **Thinking helps code solo**: 40.0% (think) vs TBD (nothink) — need comparison runs
- **High variance**: std ~3-4% across trials, always run 3+ trials
- **Budget**: m25 slightly better than m20 for flat solo

## Official Baselines

| Model | Overall |
|-------|---------|
| Gemini 3 Pro | **37.50%** |
| GPT-5.2 | 35.00% |
| Gemini 3 Flash | 33.75% |
| GPT-5 Mini | 22.50% |

## Results

| # | Run | Solver | LLM | VLM | Score |
|---|-----|--------|-----|-----|-------|
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

| Category | solo-m25-t1-val | solo-m20-t1-val | lean-solo-val-t2 | solo-m20-t2-val | solo-m25-t2-val | lean-solo-val-local | solo-m25-t3-val | lean-solo-val-t3 | rlm-lean-thinktrue-t2-val | rlm-lean-thinktrue-t3-val | solo-m20-t3-val | rlm-lean-thinkfalse-t1-val | rlm-lean-thinkfalse-t3-val | rlm-lean-thinkfalse-t2-val | rlm-code-thinktrue-t1-val | rlm-code-thinkfalse-t1-val | rlm-lean-thinktrue-t1-val | rlm-code-thinkfalse-t3-val | rlm-code-thinkfalse-t2-val | rlm-code-thinktrue-t2-val | rlm-code-thinktrue-t3-val |
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

### Best per-category

| Category | Best Score | Runs |
|----------|:----------:|------|
| business_report | 60% | solo-m25-t1-val, solo-m20-t2-val, solo-m25-t2-val |
| comics | 50% | solo-m25-t1-val, lean-solo-val-local, lean-solo-val-t3 |
| engineering_drawing | 70% | lean-solo-val-t2, rlm-lean-thinktrue-t3-val, rlm-lean-thinkfalse-t3-val, rlm-code-thinktrue-t1-val |
| infographics | 80% | solo-m25-t1-val, solo-m20-t1-val, rlm-lean-thinkfalse-t1-val |
| maps | 20% | lean-solo-val-local |
| science_paper | 40% | solo-m25-t2-val, solo-m20-t3-val |
| science_poster | 60% | rlm-lean-thinktrue-t3-val |
| slide | 60% | lean-solo-val-t2, solo-m25-t2-val |
