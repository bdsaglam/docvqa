# Experiment Results — DocVQA 2026

Full validation set (25 docs, 80 questions) results across all solver architectures and model configurations.

## Official Baselines

| Model | Overall |
|-------|---------|
| Gemini 3 Pro | **37.5%** |
| GPT-5.2 | 35.0% |
| Gemini 3 Flash | 33.75% |
| GPT-5 Mini | 22.5% |

## Model Shorthand

| Name | Full |
|------|------|
| Pro | vertex_ai/gemini-3-pro-preview |
| Pro 3.1 | vertex_ai/gemini-3.1-pro-preview |
| Flash | vertex_ai/gemini-3-flash-preview |
| Qwen 27B | hosted_vllm/Qwen/Qwen3.5-27B (localhost:8927) |

## Summary: Best Results per Solver

| Rank | Run | Solver | LLM | VLM | Score |
|------|-----|--------|-----|-----|-------|
| 1 | flat-full-pro-flash-v2 | Flat Batch | Pro | Flash | **44/80 = 55.0%** |
| 2 | full-val-parvlm-pro31-qwen | Parallel VLM | Pro 3.1 | Qwen 27B | **43/80 = 53.8%** |
| 3 | full-val-v4fix-pro31-qwen | Flat Batch | Pro 3.1 | Qwen 27B | **40/80 = 50.0%** |
| 4 | full-val-pro | Sequential RLM | Pro 3.1 | Qwen 27B | **36/80 = 45.0%** |
| 5 | full-val-v4-highiter-t2 | Flat Batch | Qwen 27B | Qwen 27B | **34/80 = 42.5%** |
| 5 | meta-v3-t3 | Meta | Qwen 27B | Qwen 27B | **34/80 = 42.5%** |
| 7 | full-val-parvlm-v3-qwen | Parallel VLM | Qwen 27B | Qwen 27B | **33/80 = 41.2%** |
| 8 | full-val-v4fix-qwen | Flat Batch | Qwen 27B | Qwen 27B | **31/80 = 38.8%** |
| 9 | flat-full-val-flash-qwen | Flat Batch | Flash | Qwen 27B | **30/80 = 37.5%** |
| 10 | full-val-v2 | Sequential RLM | Flash | Qwen 27B | **29/80 = 36.2%** |
| 11 | full-val-v3 | Sequential RLM | Flash | Qwen 27B | **24/80 = 30.0%** |

**Key finding**: Pro LLM is the dominant factor — +10-15pp over Qwen 27B regardless of solver or VLM choice. The only run using Flash VLM (Pro+Flash flat) is #1, but all other top-10 runs use local Qwen 27B for VLM.

---

## Per-Category Breakdown — Top Runs

| Category | Pro+Flash (flat) | Pro3.1+Qwen (par) | Pro3.1+Qwen (flat) | Pro3.1+Qwen (seq) | Qwen+Qwen (flat, t2) | Qwen+Qwen (par, v3) |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| business_report | 50% | **60%** | **60%** | 30% | 50% | 30% |
| comics | **50%** | 40% | 40% | 30% | 40% | **50%** |
| engineering_drawing | 70% | **90%** | 70% | 70% | 40% | 70% |
| infographics | 60% | **80%** | **70%** | 50% | **100%** | 60% |
| maps | **40%** | 20% | 20% | 20% | 10% | 10% |
| science_paper | **50%** | 30% | 30% | 30% | 10% | 30% |
| science_poster | **80%** | 50% | 50% | **70%** | 40% | 50% |
| slide | 40% | **60%** | **60%** | **60%** | 50% | 30% |
| **Overall** | **55.0%** | **53.8%** | **50.0%** | **45.0%** | **42.5%** | **41.2%** |

### Best per-category across all runs

| Category | Best | Score | Run |
|----------|------|:---:|-----|
| engineering_drawing | Pro3.1+Qwen (par) | 90% | full-val-parvlm-pro31-qwen |
| science_poster | Pro+Flash (flat) | 80% | flat-full-pro-flash-v2 |
| infographics | Qwen+Qwen (flat, t2) | 100% | full-val-v4-highiter-t2 |
| business_report | Pro3.1+Qwen (par/flat) | 60% | full-val-parvlm-pro31-qwen, full-val-v4fix-pro31-qwen |
| comics | Pro+Flash (flat), Qwen+Qwen (par) | 50% | flat-full-pro-flash-v2, others |
| science_paper | Pro+Flash (flat) | 50% | flat-full-pro-flash-v2 |
| slide | Pro3.1+Qwen (par/flat/seq) | 60% | multiple |
| maps | Pro+Flash (flat) | 40% | flat-full-pro-flash-v2 |

---

## Solver Architectures

### 1. Sequential RLM (`rlm_solver.py`)

Original sequential solver — one question at a time, sub-agent for VLM reasoning.

| Run | LLM | VLM | Score |
|-----|-----|-----|:---:|
| full-val-pro | Pro 3.1 | Qwen 27B | 36/80 = 45.0% |
| full-val-v2 | Flash | Qwen 27B | 29/80 = 36.2% |
| full-val-v3 | Flash | Qwen 27B | 24/80 = 30.0% |

### 2. Flat Batch (`flat_batch_solver.py`)

Direct VLM calls, no sub-agent hierarchy. PIL images loaded in sandbox, agent crops freely. Fastest architecture.

| Run | LLM | VLM | Budget | Score |
|-----|-----|-----|:---:|:---:|
| flat-full-pro-flash-v2 | Pro | Flash | b2-pq3 | **44/80 = 55.0%** |
| full-val-v4fix-pro31-qwen | Pro 3.1 | Qwen 27B | b6-pq4 | 40/80 = 50.0% |
| full-val-v4-highiter-t2 | Qwen 27B | Qwen 27B | b6-pq4 | 34/80 = 42.5% |
| full-val-v4fix-qwen | Qwen 27B | Qwen 27B | b6-pq4 | 31/80 = 38.8% |
| flat-full-val-flash-qwen | Flash | Qwen 27B | b2-pq3 | 30/80 = 37.5% |

#### Budget sweep (Qwen 27B + Qwen 27B, b=X per_question=Y)

3 trials each. Mean and range:

| Config | Mean | Range |
|--------|:---:|:---:|
| **b4-pq3** | **34.1%** | 31.6-38.8% |
| b6-pq2 | 32.5% | 29.4-36.2% |
| b6-pq5 | 28.1% | 26.2-30.0% |

### 3. Parallel VLM (`parallel_vlm_solver.py`)

Forces parallel VLM reads: redundant, multi-scale, grid scan. `batch_look` only, no single `look()`.

| Run | LLM | VLM | Score |
|-----|-----|-----|:---:|
| full-val-parvlm-pro31-qwen | Pro 3.1 | Qwen 27B | **43/80 = 53.8%** |
| full-val-parvlm-v3-qwen | Qwen 27B | Qwen 27B | 33/80 = 41.2% |
| full-val-parvlm-v4-qwen | Qwen 27B | Qwen 27B | 32/80 = 40.0% |
| full-val-parvlm-v5-leanrlm | Qwen 27B | Qwen 27B | 32/80 = 40.0% |
| full-val-parvlm-v6-noindentt | Qwen 27B | Qwen 27B | 32/80 = 40.0% |
| full-val-parvlm-qwen | Qwen 27B | Qwen 27B | 26/80 = 32.5% |

Consistently ~40% with Qwen/Qwen. +13pp with Pro LLM. LeanRLM and no-indent patches had no significant effect.

### 4. Meta Solver (`meta_solver.py`)

Flat-batch main agent + optional `subagent()` tool for delegation. Sub-agent = separate RLM with own context window. All trials use Qwen 27B for both LLM and VLM.

| Run | Version | Score |
|-----|---------|:---:|
| meta-v3-t3 | v3 (flat-like + optional sub) | **34/80 = 42.5%** |
| meta-v3-t2 | v3 | 32/80 = 40.0% |
| meta-v3-t1 | v3 | 24/80 = 30.0% |
| meta-v4-t2 | v4 (hybrid, nudge delegation) | 27/80 = 33.8% |
| meta-v4-t3 | v4 | 24/80 = 30.0% |
| meta-v6-t3 | v6 (parallel VLM sub-agents) | 26/80 = 32.5% |
| meta-v6-t1 | v6 | 25/80 = 31.2% |
| meta-v6-t2 | v6 | 24/80 = 30.0% |

**v3 mean**: 30.0%, **v4 mean**: 28.9%, **v6 mean**: 27.1%

**Conclusion**: Sub-agent overhead (new subprocess + RLM + PIL loading ~2-5 min) outweighs benefits. Qwen not smart enough to effectively orchestrate. v3 sub-agent never used — effectively flat.

### 5. Structured VLM (`structured_vlm_solver.py`)

Structured output VLM approach. Early exploration, not competitive.

| Run | Score |
|-----|:---:|
| structured-vlm-targeted-t1 (8q) | 37.5% |
| structured-vlm-quick-t2 (5q) | 80.0% |
| structured-vlm-quick-t1 (5q) | 20.0% |

### 6. RVLM (`rvlm_solver.py`)

Direct image display (no tool-based VLM). Early exploration.

| Run | LLM | Score |
|-----|-----|:---:|
| rvlm-flash-3docs (9q) | Flash | 55.6% |
| rvlm-flash-v2 (9q) | Flash | 44.4% |
| rvlm-qwen-v2 (9q) | Qwen 27B | 22.2% |
| rvlm-qwen-3docs (9q) | Qwen 27B | 11.1% |

### 7. Skilled Batch (`skilled_batch_solver.py`)

Skill-based approach with specialized tooling.

| Run | LLM | VLM | Score |
|-----|-----|-----|:---:|
| skilled-full-pro-flash | Pro | Flash | 33/80 = 41.2% |
| skilled-full-pro-flash-v2 | Pro | Flash | 20/80 = 25.0% |

---

## Prompt Version Experiments

All on Qwen 27B LLM + Qwen 27B VLM. Dev set = 17 docs (59 questions), full val = 25 docs (80 questions).

| Version | Dev set mean (3 trials) | Full val |
|---------|:---:|:---:|
| Baseline (b4-pq3, old prompts) | 31.6% | — |
| **v4 (b4-pq3)** | **37.3%** (+5.7pp) | — |
| **v4+HiIter (b6-pq4)** | **39.5%** (+7.9pp) | 38.8% (t1=35.0, t2=42.5) |

v4 additions: VLM conflict resolution, superlative enumeration, stronger Unknown rules, per-category tips (OCR confusion, story maps, citation regex).

---

## Key Findings

1. **Pro LLM is the biggest factor** — +10-15pp over Qwen 27B regardless of solver architecture
2. **Local Qwen 27B VLM** is the default VLM for almost all runs — reliable, no rate limits
3. **Flash VLM on Vertex** only used in one run (flat-full-pro-flash-v2) but achieved the top score — fast, good quality
4. **Flat Batch is the best architecture** — simple, fast, 55% with Pro+Flash
5. **Parallel VLM is second** — consistently ~40% with Qwen/Qwen, ~54% with Pro+Qwen
6. **Meta/sub-agent overhead is net negative** with Qwen — orchestration iterations are NOT extraction iterations
7. **Maps are the hardest category** — spatial reasoning needs Pro LLM, 40% max
8. **Variance is high with Qwen LLM** — 3-4% std, always run 3+ trials
