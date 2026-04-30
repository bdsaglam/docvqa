# Hyperparameter Sweep: flat_batch solver

## Fixed params (all experiments)
- Solver: flat_batch
- LLM: Qwen3.5-27B local (port 8927)
- VLM: Qwen3.5-27B local (port 8927)
- LLM: t=0.6, top_p=0.95, top_k=20
- VLM: t=0.3, top_k=20, enable_thinking=false
- Data: full val set (25 docs, 80 questions)
- Concurrency: 8 per run (3 parallel)
- Timeout: 7200s

## Phase 1: Budget sweep (code RLM, thinking ON)

| Exp | base | pq | pf | max | run_id | Score |
|-----|:----:|:--:|:--:|:---:|--------|:-----:|
| 1a  | 4    | 3  | 1.5| 30  | budget-low-code-val | 38.8% |
| **1b** | **6** | **4** | **1.5** | **40** | **budget-mid-code-val** | **41.2%** |
| 1c  | 8    | 5  | 1.5| 50  | budget-high-code-val | 36.2% |

**Winner: b6-pq4, max=40** — mid budget is optimal, higher hurts (over-exploration).

## Phase 2: RLM type + thinking (b6-pq4, max=40)

| Exp | rlm_type | thinking | run_id | Score |
|-----|----------|:--------:|--------|:-----:|
| 2a  | code     | ON       | budget-mid-code-val | 41.2% |
| 2b  | lean     | ON       | phase2-lean-think-val | 41.2% |
| 2c  | code     | OFF      | phase2-code-nothink-val | 40.0% |
| **2d** | **lean** | **OFF** | **phase2-lean-nothink-val** | **45.0%** |

**Winner: lean + no thinking** — reasoning field helps, thinking tokens hurt.

## Phase 3: Temperature (lean, no thinking, b6-pq4)

| Exp | LLM temp | VLM temp | run_id | Score |
|-----|:--------:|:--------:|--------|:-----:|
| 3a  | 0.3      | 0.3      | phase3-t03-t03-val | 38.8% |
| **3b** | **0.6** | **0.3** | **(Phase 2 winner)** | **45.0%** |
| 3c  | 0.6      | 0.6      | phase3-t06-t06-val | 36.2% |
| 3d  | 1.0      | 0.3      | phase3-t10-t03-val | 35.0% |

**Winner: LLM t=0.6, VLM t=0.3** — confirmed. VLM t=0.6 hurts (-9pp), higher/lower LLM both hurt.

## Phase 4: Page factor (best from Phase 3)

| Exp | pf  | run_id | Score |
|-----|:---:|--------|:-----:|
| 4a  | 0.0 | phase4-pf0-val | 38.8% |
| 4b  | 1.0 | phase4-pf1-val | 35.0% |
| **4c** | **1.5** | **(from Phase 3 winner)** | **45.0%** |
| 4d  | 2.0 | phase4-pf2-val | 41.2% |

**Winner: pf=1.5** — confirmed. No page bonus (0.0) loses 6pp on multi-page docs.

## Final Best Config
- **Solver**: flat_batch
- **RLM type**: lean (reasoning field ON)
- **Thinking**: OFF
- **Budget**: b6-pq4, pf=1.5, max=40
- **LLM**: t=0.6, top_p=0.95, top_k=20, enable_thinking=false
- **VLM**: t=0.3, top_k=20, enable_thinking=false
- **Score**: 45.0%

## Phase 5: Variance measurement (3 trials of best config)

| Trial | Score |
|:-----:|:-----:|
| t1    | 31.2% |
| t2    | 37.5% |
| t3    | 41.2% |
| **Mean** | **36.7% +/- 5.1%** |

Per-category variance:
- Stable: maps (10% +/- 0), science_paper (20% +/- 0)
- High variance: comics (46.7% +/- 15.3%), engineering_drawing (53.3% +/- 11.5%)

## Solo Solver Sweep (lean, no thinking, LLM 0.6, VLM 0.3, pf=1.5)

| Exp | max_iter | run_id | Score |
|-----|:--------:|--------|:-----:|
| S1  | 15       | solo-low-val | 37.5% |
| **S2** | **20** | **solo-mid-val** | **45.0%** |
| S3  | 30       | solo-high-val | 37.5% |

**Winner: max_iterations=20** — same mid-budget sweet spot as flat_batch.

## Solo Variance (3 trials, max_iter=20, lean, no thinking)

| Trial | Score |
|:-----:|:-----:|
| t1    | 40.0% |
| t2    | 35.0% |
| t3    | 46.2% |
| **Mean** | **40.4% +/- 5.6%** |

Stable categories: slide (50% +/- 0), maps (10% +/- 0), ED (56.7% +/- 5.8%)
High variance: science_poster (40% +/- 26.5%), science_paper (23.3% +/- 15.3%)

## Solver Comparison (mean across 3 trials)

| Solver | Mean | Std |
|--------|:----:|:---:|
| **flat_solo** | **40.4%** | 5.6% |
| flat_batch | 36.7% | 5.1% |

**Solo is +3.7pp better on average.** Per-question isolation helps.

## Phase 6: RLM type sweep with 3 trials (remote Qwen, flat_batch)

| Config | t1 | t2 | t3 | Mean | Std |
|--------|:---:|:---:|:---:|:----:|:---:|
| **lean + think** | 31.2% | 37.5% | 37.5% | **35.4%** | 3.6% |
| lean + nothink | 36.2% | 33.8% | 35.0% | 35.0% | 1.2% |
| code + think | 33.8% | 28.7% | 26.2% | 29.6% | 3.8% |
| code + nothink | 32.5% | 30.0% | 31.2% | 31.2% | 1.2% |

Lean > code consistently. Thinking adds variance but not accuracy.
Remote Qwen scores ~2-3pp lower than local across configs.

## lean_solo (batch_look only, max_iter=25, 3 trials, local)

| Trial | Score |
|:-----:|:-----:|
| t1 | 40.0% |
| t2 | 42.5% |
| t3 | 37.5% |
| **Mean** | **40.0% +/- 2.5%** |

Comparable to flat_solo but **lower variance** (2.5% vs 4.4%).

## Results tracking
- **Best single trial**: 46.2% (flat_solo m25 t1)
- **Best mean flat_solo**: 41.6% +/- 4.4% (m25)
- **Best mean lean_solo**: 40.0% +/- 2.5% (batch_look only, m25)
- **Best mean flat_batch**: 36.7% +/- 5.1%
- High variance categories (comics, ED, poster) are the main source of instability
