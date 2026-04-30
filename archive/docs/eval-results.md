# Eval Results Tracker

All runs on DocVQA 2026 val set (25 docs, 80 questions) unless noted.

## Summary Table

| # | Run ID | Solver | LLM | VLM | Budget | Conc | Score | Date |
|---|--------|--------|-----|-----|--------|:----:|:-----:|------|
| 1 | flat-full-pro-flash-v2 | flat_batch | Pro | Flash | b6-pq4 | 1 | **55.0%** | Mar 23 |
| 2 | full-val-parvlm-pro31-qwen | parallel_vlm | Pro 3.1 | Qwen 27B | — | 1 | 53.8% | Mar 28 |
| 3 | full-val-v4fix-pro31-qwen | flat_batch | Pro 3.1 | Qwen 27B | b6-pq4 | 1 | 50.0% | Mar 28 |
| 4 | full-val-pro | rlm (sequential) | Pro 3.1 | Qwen 27B | — | 1 | 45.0% | Mar 22 |
| 5 | full-val-v4-highiter-t2 | flat_batch | Qwen 27B | Qwen 27B | b6-pq4 | 1 | 42.5% | Mar 27 |
| 6 | **flat-remote-b6pq4-c4** | **flat_batch** | **Qwen 27B** | **Qwen 27B** | **b6-pq4** | **4** | **41.2%** | Apr 1 |
| 7 | full-val-parvlm-v3-qwen | parallel_vlm | Qwen 27B | Qwen 27B | — | 1 | 41.2% | Mar 25 |
| 8 | full-val-v4fix-qwen (mean) | flat_batch | Qwen 27B | Qwen 27B | b6-pq3 | 1 | 38.8% | Mar 27 |
| 9 | flat-remote-b8pq4-c4 | flat_batch | Qwen 27B (remote) | Qwen 27B (remote) | b8-pq4 | 4 | 37.5% | Apr 1 |
| 10 | flat-full-val-flash-qwen | flat_batch | Flash | Qwen 27B | b2-pq3 | 1 | 37.5% | Mar 23 |
| 11 | flat-full-qwen-v4fix-c4 | flat_batch | Qwen 27B | Qwen 27B | b6-pq2 | 4 | 35.0% | Apr 1 |
| 12 | parvlm-full-qwen-v1-c4 | parallel_vlm | Qwen 27B | Qwen 27B | — | 4 | 27.5% | Apr 1 |
| 13 | orch-remote-sub12-c8 | orchestrator sub12 | Qwen 27B (remote) | Qwen 27B (remote) | sub12 | 8 | 21.2% | Apr 1 |
| 14 | **parvlm-remote-c8** | **parallel_vlm** | **Qwen 27B (remote)** | **Qwen 27B (remote)** | **—** | **8** | **43.8%** | Apr 2 |
| 15 | sweep-lean-b6pq4-c2-t1 | flat_batch (lean) | Qwen 27B | Qwen 27B | b6-pq4 | 2 | 37.5% | Apr 2 |
| 16 | sweep-std-b6pq4-c2-t1 | flat_batch (std) | Qwen 27B | Qwen 27B | b6-pq4 | 2 | 33.8% | Apr 2 |
| 17 | lean-v2-remote-c8 | lean | Qwen 27B (remote) | Qwen 27B (remote) | — | 8 | 32.5% | Apr 2 |
| 18 | **parvlm-pb-remote-c4** | **parallel_vlm** | **Qwen 27B (remote)** | **Qwen 27B (remote)** | **b6-pq4+pf1.0** | **4** | **38.8%** | Apr 2 |
| 19 | flat-pb-remote-v3-c4 | flat_batch (lean) | Qwen 27B (remote) | Qwen 27B (remote) | b6-pq4+pf1.0 | 4 | 35.0% | Apr 2 |
| 20 | **flat-pf15-local-c3** | **flat_batch (lean)** | **Qwen 27B** | **Qwen 27B** | **b6-pq4+pf1.5** | **3** | **42.5%** | Apr 2 |
| 21 | parvlm-b4pq3-pf15-remote-c4 | parallel_vlm | Qwen 27B (remote) | Qwen 27B (remote) | b4-pq3+pf1.5 | 4 | 38.8% | Apr 2 |
| 22 | parvlm-b6pq2-pf15-remote-c4 | parallel_vlm | Qwen 27B (remote) | Qwen 27B (remote) | b6-pq2+pf1.5 | 4 | 38.8% | Apr 2 |
| 23 | flat-b6pq2-pf15-remote-c4 | flat_batch (lean) | Qwen 27B (remote) | Qwen 27B (remote) | b6-pq2+pf1.5 | 4 | 35.0% | Apr 2 |
| 24 | flat-b4pq3-pf15-remote-c4 | flat_batch (lean) | Qwen 27B (remote) | Qwen 27B (remote) | b4-pq3+pf1.5 | 4 | 31.2% | Apr 2 |
| 25 | **qwen-general-precise-c8** | **flat_batch (lean)** | **Qwen 27B (remote) t=0.7 tp=0.8 tk=20 pp=1.5** | **Qwen 27B (remote) t=0.6 tp=0.95 tk=20** | **b6-pq4+pf1.5** | **8** | **42.5%** | Apr 3 |
| 26 | qwen-reason-precise-c8 | flat_batch (lean) | Qwen 27B (remote) t=1.0 tp=1.0 tk=40 pp=2.0 | Qwen 27B (remote) t=0.6 tp=0.95 tk=20 | b6-pq4+pf1.5 | 8 | 38.8% | Apr 3 |
| 27 | **t06-precise-local-c3** | **flat_batch (lean)** | **Qwen 27B t=0.6 tp=0.95 tk=20** | **Qwen 27B t=0.3 tk=20** | **b6-pq4+pf1.5** | **3** | **45.0%** | **Apr 3** |
| 28 | qwen-general-precise-c8-t2 | flat_batch (lean) | Qwen 27B (remote) t=0.7 tp=0.8 tk=20 pp=1.5 | Qwen 27B (remote) t=0.6 tp=0.95 tk=20 | b6-pq4+pf1.5 | 8 | 40.0% | Apr 3 |
| 29 | qwen-general-lean-remote-c8 | flat_batch (lean) | Qwen 27B (remote) t=0.7 tp=0.8 tk=20 pp=1.5 | Qwen 27B (remote) t=0.6 tp=0.95 tk=20 | b4-pq3+pf1.5 | 8 | 37.5% | Apr 4 |
| 30 | explore-precise-local-c3 | flat_batch (lean) | Qwen 27B t=1.0 tp=0.95 tk=20 pp=1.5 | Qwen 27B t=0.3 tk=20 | b6-pq4+pf1.5 | 3 | 36.2% | Apr 3 |
| 28 | routing-local-c4 | routing (flat+pvlm) | Qwen 27B | Qwen 27B | b6-pq4+pf1.5 | 4 | 36.2% | Apr 2 |
| 26 | routing-remote-c8 | routing (flat+pvlm) | Qwen 27B (remote) | Qwen 27B (remote) | b6-pq4+pf1.5 | 8 | 33.8% | Apr 2 |
| 27 | orch-full-qwen-v1-c4 | orchestrator | Qwen 27B | Qwen 27B | — | 4 | 22.5% | Apr 2 |
| 26 | test-lean-b6pq4-c4 | flat_batch (lean) | Qwen 27B | Qwen 27B | b6-pq4 | 4 | 0.0% | Apr 2 |

## In Progress

| Run ID | Solver | LLM | VLM | Notes |
|--------|--------|-----|-----|-------|
| pagebudget-qw-t1/t2/t3 | rvlm (page budget) | Qwen 27B | Qwen 27B | Stuck/timed out, no results |
| orch-remote-qwen | orchestrator | Qwen 27B (remote) | Qwen 27B (remote) | Incomplete, no results |
| parvlm-remote-v2-c8 | parallel_vlm | Qwen 27B (remote) | Qwen 27B (remote) | Cancelled, no results |

## Key Findings

### Budget (Qwen/Qwen flat_batch)
- **b6-pq4 = 41.2%** (best Qwen/Qwen)
- b8-pq4 = 37.5% (too much budget, agent wastes iterations)
- b6-pq2 = 35.0% (too little per-question budget)
- Sweet spot is b6-pq4

### Solver Comparison (Qwen/Qwen, same server)
- **parallel_vlm (43.8%) >= flat_batch (41.2%) >> orchestrator (22.5%)** at best configs
- parallel_vlm with remote c8 is new Qwen/Qwen best (43.8%)
- Orchestrator overhead (sub-agent spawning) not worth it with Qwen

### Concurrency
- c4 is safe for single Qwen server (3x A100)
- c8 on remote works well for parallel_vlm (43.8%) — no quality regression
- c8 causes regression for lean solver (32.5% vs 37.5% at c2)

### LLM Quality Matters Most
- Pro LLM + Flash VLM = 55.0% (best overall)
- Pro LLM + Qwen VLM = 50-54%
- Qwen LLM + Qwen VLM = 35-41%
- The gap is in reasoning quality, not vision

### Orchestrator Analysis
- 4/25 docs timed out at 3600s (comics_2, comics_3, eng_drawing_1, infographics_2)
- Flat beats orchestrator on 9/25 docs, orchestrator wins on only 1
- Flat is 3x faster (median 534s vs 1778s)
- Testing Pro as main agent + Qwen sub-agents to isolate cause

### Lean vs Standard flat_batch (Apr 2, c2, b6-pq4)
- Lean: 37.5% — Standard: 33.8%
- Lean wins on business_report (+20pp), science_paper (+20pp), science_poster (+20pp), maps (+10pp)
- Standard wins on infographics (+20pp), slide (+20pp)
- test-lean-b6pq4-c4: **0%** (160 questions, 2x data) — likely a bug (broken solver or double-counted questions)

### Prompt Changes (Apr 1)
- LeanRLM/RVLM action instructions prompt was changed
- Previous wording was validated (38-42%), new wording may cause regression
- Standard RLM prompt was NOT changed (only LeanRLM + RVLM)
- Need to revert or A/B test to confirm

## Per-Category Best (Qwen/Qwen)

| Category | Best Score | Run |
|----------|:----------:|-----|
| engineering_drawing | 100% | flat-remote-b6pq4-c4 |
| business_report | 100% | flat-remote-b6pq4-c4 |
| infographics | 75% | flat-remote-b6pq4-c4 |
| slide | 67% | flat-remote-b6pq4-c4 |
| science_poster | 60% | flat-remote-b6pq4-c4 |
| comics | 50% | flat-remote-b6pq4-c4 |
| science_paper | 43% | flat-remote-b6pq4-c4 |
| maps | 40% | flat-remote-b6pq4-c4 |

## Servers

| Server | Port | GPUs | Notes |
|--------|------|------|-------|
| Local Qwen | 8927 | 3x A100 80GB | Safe at c4 |
| Remote Qwen | 8928 | 4x GPU | Safe at c8 |

## Per-Doc Comparison: Orchestrator vs Flat (remote, Apr 1)

| Doc | Orch sub12 | Flat b6pq4 | Winner |
|-----|:----------:|:----------:|--------|
| business_report_1 | 0% (2985s) | 50% (239s) | FLAT |
| business_report_2 | 0% (1641s) | 0% (154s) | TIE |
| business_report_3 | 50% (1105s) | 100% (370s) | FLAT |
| business_report_4 | 40% (1778s) | 20% (536s) | ORCH |
| comics_1 | 0% (1143s) | 0% (515s) | TIE |
| comics_2 | 0% (3600s timeout) | 50% (2120s) | FLAT |
| comics_3 | 0% (3600s timeout) | 33% (834s) | FLAT |
| comics_4 | 50% (1928s) | 50% (1508s) | TIE |
| engineering_drawing_1 | 0% (3600s timeout) | 0% (985s) | TIE |
| engineering_drawing_2 | 100% (1253s) | 100% (153s) | TIE |
| engineering_drawing_3 | 33% (2621s) | 33% (417s) | TIE |
| engineering_drawing_4 | 50% (1176s) | 50% (534s) | TIE |
| infographics_1 | 50% (2071s) | 50% (131s) | TIE |
| infographics_2 | 0% (3600s timeout) | 75% (350s) | FLAT |
| maps_1 | 0% (3031s) | 0% (876s) | TIE |
| maps_2 | 0% (2017s) | 40% (1197s) | FLAT |
| maps_3 | 0% (2212s) | 0% (948s) | TIE |
| science_paper_1 | 14% (999s) | 43% (997s) | FLAT |
| science_paper_2 | 0% (775s) | 0% (226s) | TIE |
| science_paper_3 | 0% (722s) | 0% (450s) | TIE |
| science_poster_1 | 20% (1051s) | 20% (1394s) | TIE |
| science_poster_2 | 40% (1369s) | 60% (641s) | FLAT |
| slide_1 | 50% (1211s) | 50% (194s) | TIE |
| slide_2 | 33% (1309s) | 67% (524s) | FLAT |
| slide_3 | 40% (2134s) | 40% (1933s) | TIE |

Orch wins: 1, Flat wins: 9, Ties: 15. Flat is 3x faster.
