# Solver Comparison — Qwen/Qwen (all categories)

Config: `lm=qwen27b vlm=qwen27b`
Solvers: flat_batch, orchestrator, parallel_rvlm, rvlm (meta dropped after poster)

## Results (t1 scores)

| Category | flat_batch | orchestrator | parallel_rvlm | rvlm | Best |
|----------|-----------|-------------|--------------|------|------|
| science_poster | **4/10 (40%)** | 0/10 (0%) | 1/10 (10%) | 3/10 (30%) | flat_batch |
| engineering_drawing | **4/10 (40%)** | ~3/10 (30%)* | 4/10 (40%) | 3/10 (30%) | flat_batch/prvlm |
| infographics | **7/10 (70%)** | 4/10 (40%) | 1/10 (10%) | 5/10 (50%) | flat_batch |
| slide | 3/10 (30%) | 3/10 (30%) | **4/10 (40%)** | 3/10 (30%) | parallel_rvlm |
| science_paper | 0/10 (0%) | **2/10 (20%)** | 0/10 (0%) | **2/10 (20%)** | orch/rvlm |
| maps | **2/10 (20%)** | 1/10 (10%) | 1/10 (10%) | 0/10 (0%) | flat_batch |
| business_report | **5/10 (50%)** | 1/10 (10%) | 4/10 (40%) | 0/10 (0%) | flat_batch |
| comics | **4/10 (40%)** | _running_ | _running_ | 2/10 (20%) | flat_batch |
| **TOTAL** | **29/80** | **13/70** | **19/80** | **18/80** | |

*ed-orchestrator timed out on ed_3 — 3/7 correct on completed Qs

## Full eval (flat_batch lean, all 80 Qs)

**21/80 (26.2%)**

| Category | Score |
|----------|-------|
| slide | 5/10 (50%) |
| infographics | 4/10 (40%) |
| business_report | 3/10 (30%) |
| engineering_drawing | 3/10 (30%) |
| comics | 2/10 (20%) |
| science_poster | 2/10 (20%) |
| maps | 1/10 (10%) |
| science_paper | 1/10 (10%) |

## Per-Category Details

### science_poster (2 docs, 10 Qs)

| Solver | t1 | t2 | Mean |
|--------|:---:|:---:|:---:|
| **flat_batch** | 4/10 (40%) | 6/10 (60%) | **50%** |
| rvlm | 3/10 (30%) | 5/10 (50%) | 40% |
| parallel_rvlm | 1/10 (10%) | 7/10 (70%) | 40% |
| orchestrator | 0/10 (0%) | 3/10 (30%) | 15% |

### engineering_drawing (4 docs, 10 Qs)

| Solver | t1 |
|--------|:---:|
| **flat_batch** | **4/10 (40%)** |
| **parallel_rvlm** | **4/10 (40%)** |
| rvlm | 3/10 (30%) |
| orchestrator | _running_ |

### infographics (2 docs, 10 Qs)

| Solver | t1 |
|--------|:---:|
| **flat_batch** | **7/10 (70%)** |
| rvlm | 5/10 (50%) |
| orchestrator | 4/10 (40%) |
| parallel_rvlm | 1/10 (10%) |

### slide (3 docs, 10 Qs)

| Solver | t1 |
|--------|:---:|
| **parallel_rvlm** | **4/10 (40%)** |
| flat_batch | 3/10 (30%) |
| rvlm | 3/10 (30%) |
| orchestrator | _running_ |

### science_paper (3 docs, 10 Qs)

| Solver | t1 |
|--------|:---:|
| **orchestrator** | **2/10 (20%)** |
| **rvlm** | **2/10 (20%)** |
| flat_batch | 0/10 (0%) |
| parallel_rvlm | _running_ |

Note: flat_batch scored 0/10 on science_paper — worst category for it.

### RLM type comparison (flat_batch, 5 random docs, 18 Qs)

| RLM type | t1 | t2 | Mean |
|----------|----|----|------|
| **lean** | 8/18 (44.4%) | 5/18 (27.8%) | **36.1%** |
| standard | 3/18 (16.7%) | 2/18 (11.1%) | 13.9% |

Lean RLM is decisively better with Qwen — now default for all solvers.
