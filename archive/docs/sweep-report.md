# Experiment Results — DocVQA 2026

## Official Baselines

| Model | Overall |
|-------|---------|
| Gemini 3 Pro | **37.50%** |
| GPT-5.2 | 35.00% |
| Gemini 3 Flash | 33.75% |
| GPT-5 Mini | 22.50% |

## Model Shorthand

| Short | Full |
|-------|------|
| Pro 3.1 | `vertex_ai/gemini-3.1-pro-preview` |
| Qwen 27B | `hosted_vllm/Qwen/Qwen3.5-27B` |

## Results

| # | Run | Solver | LLM | VLM | Score |
|---|-----|--------|-----|-----|-------|
| 1 | full-val-parvlm-pro31-qwen | Parallel VLM | Pro 3.1 | Qwen 27B | **43/80 = 53.8%** |
| 2 | ocr-tool-full-val-qwen-t2 | Flat Batch | Qwen 27B | Qwen 27B | 37/80 = 46.2% |
| 3 | t06-precise-local-c3 | Flat Batch | Qwen 27B | Qwen 27B | 36/80 = 45.0% |
| 4 | parvlm-remote-c8 | Parallel VLM | Qwen 27B | Qwen 27B | 35/80 = 43.8% |
| 5 | flat-pf15-local-c3 | Flat Batch | Qwen 27B | Qwen 27B | 34/80 = 42.5% |
| 6 | flat-t06-full-remote-c4 | Flat Batch | Qwen 27B | Qwen 27B | 34/80 = 42.5% |
| 7 | flat-t06both-remote-c4 | Flat Batch | Qwen 27B | Qwen 27B | 34/80 = 42.5% |
| 8 | qwen-general-precise-c8 | Flat Batch | Qwen 27B | Qwen 27B | 34/80 = 42.5% |
| 9 | flat-solo-val-local-c4 | Flat Solo | Qwen 27B | Qwen 27B | 33/80 = 41.2% |
| 10 | qwen-general-precise-c8-t2 | Flat Batch | Qwen 27B | Qwen 27B | 32/80 = 40.0% |
| 11 | t06-precise-lean-local-c3 | Flat Batch | Qwen 27B | Qwen 27B | 32/80 = 40.0% |
| 12 | t06-precise-local-c3-t2 | Flat Batch | Qwen 27B | Qwen 27B | 32/80 = 40.0% |
| 13 | t06-think-local-c16 | Flat Batch | Qwen 27B | Qwen 27B | 32/80 = 40.0% |
| 14 | gen-think-remote-c16 | Flat Batch | Qwen 27B | Qwen 27B | 31/80 = 38.8% |
| 15 | parvlm-b4pq3-pf15-remote-c4 | Parallel VLM | Qwen 27B | Qwen 27B | 31/80 = 38.8% |
| 16 | parvlm-b6pq2-pf15-remote-c4 | Parallel VLM | Qwen 27B | Qwen 27B | 31/80 = 38.8% |
| 17 | parvlm-pb-remote-c4 | Parallel VLM | Qwen 27B | Qwen 27B | 31/80 = 38.8% |
| 18 | qwen-reason-precise-c8 | Flat Batch | Qwen 27B | Qwen 27B | 31/80 = 38.8% |
| 19 | sweep-lean-b6pq4-t2 | Flat Batch | Qwen 27B | Qwen 27B | 31/80 = 38.8% |
| 20 | flat-t06both-v2-remote-c4 | Flat Batch | Qwen 27B | Qwen 27B | 30/80 = 37.5% |
| 21 | ocr-tool-full-val-qwen | Flat Batch | Qwen 27B | Qwen 27B | 30/80 = 37.5% |
| 22 | qwen-general-lean-remote-c8 | Flat Batch | Qwen 27B | Qwen 27B | 30/80 = 37.5% |
| 23 | sweep-lean-b6pq4-c2-t1 | Flat Batch | Qwen 27B | Qwen 27B | 30/80 = 37.5% |
| 24 | explore-precise-local-c3 | Flat Batch | Qwen 27B | Qwen 27B | 29/80 = 36.2% |
| 25 | lean-no-search-c8 | Lean | Qwen 27B | Qwen 27B | 29/80 = 36.2% |
| 26 | ocr-tool-v2-full-val-qwen | Flat Batch OCR | Qwen 27B | Qwen 27B | 29/80 = 36.2% |
| 27 | routing-local-c4 | Routing | Qwen 27B | ? | 29/80 = 36.2% |
| 28 | sweep-std-b6pq4-t3 | Flat Batch | Qwen 27B | Qwen 27B | 29/80 = 36.2% |
| 29 | flat-b6pq2-pf15-remote-c4 | Flat Batch | Qwen 27B | Qwen 27B | 28/80 = 35.0% |
| 30 | flat-best-local-c3-t2 | Flat Batch | Qwen 27B | Qwen 27B | 28/80 = 35.0% |
| 31 | flat-pb-remote-v3-c4 | Flat Batch | Qwen 27B | Qwen 27B | 28/80 = 35.0% |
| 32 | sweep-lean-b6pq4-t1 | Flat Batch | Qwen 27B | Qwen 27B | 28/80 = 35.0% |
| 33 | parvlm-remote-pq4-c4 | Parallel VLM | Qwen 27B | Qwen 27B | 27/80 = 33.8% |
| 34 | routing-remote-c8 | Routing | Qwen 27B | ? | 27/80 = 33.8% |
| 35 | sweep-std-b6pq4-c2-t1 | Flat Batch | Qwen 27B | Qwen 27B | 27/80 = 33.8% |
| 36 | sweep-std-b6pq4-t1 | Flat Batch | Qwen 27B | Qwen 27B | 27/80 = 33.8% |
| 37 | lean-v2-remote-c8 | Lean | Qwen 27B | Qwen 27B | 26/80 = 32.5% |
| 38 | sweep-std-b6pq4-t2 | Flat Batch | Qwen 27B | Qwen 27B | 26/80 = 32.5% |
| 39 | flat-b4pq3-pf15-remote-c4 | Flat Batch | Qwen 27B | Qwen 27B | 25/80 = 31.2% |
| 40 | sweep-lean-b6pq4-t3 | Flat Batch | Qwen 27B | Qwen 27B | 25/80 = 31.2% |
| 41 | flat-best-test-remote-c4 | Flat Batch | Qwen 27B | Qwen 27B | 0/160 = 0.0% |
| 42 | qwen-general-test-c8 | Flat Batch | Qwen 27B | Qwen 27B | 0/160 = 0.0% |
| 43 | t06-precise-test-c4 | Flat Batch | Qwen 27B | Qwen 27B | 0/160 = 0.0% |
| 44 | test-lean-b6pq4-c4 | Flat Batch | Qwen 27B | Qwen 27B | 0/160 = 0.0% |
| 45 | test-set-parvlm-qwen | Parallel VLM | Qwen 27B | Qwen 27B | 0/160 = 0.0% |

## Per-Category Breakdown

| Category | full-val-parvlm-pro31-qwen | ocr-tool-full-val-qwen-t2 | t06-precise-local-c3 | parvlm-remote-c8 | flat-pf15-local-c3 | flat-t06-full-remote-c4 | flat-t06both-remote-c4 | qwen-general-precise-c8 | flat-solo-val-local-c4 | qwen-general-precise-c8-t2 | t06-precise-lean-local-c3 | t06-precise-local-c3-t2 | t06-think-local-c16 | gen-think-remote-c16 | parvlm-b4pq3-pf15-remote-c4 | parvlm-b6pq2-pf15-remote-c4 | parvlm-pb-remote-c4 | qwen-reason-precise-c8 | sweep-lean-b6pq4-t2 | flat-t06both-v2-remote-c4 | ocr-tool-full-val-qwen | qwen-general-lean-remote-c8 | sweep-lean-b6pq4-c2-t1 | explore-precise-local-c3 | lean-no-search-c8 | ocr-tool-v2-full-val-qwen | routing-local-c4 | sweep-std-b6pq4-t3 | flat-b6pq2-pf15-remote-c4 | flat-best-local-c3-t2 | flat-pb-remote-v3-c4 | sweep-lean-b6pq4-t1 | parvlm-remote-pq4-c4 | routing-remote-c8 | sweep-std-b6pq4-c2-t1 | sweep-std-b6pq4-t1 | lean-v2-remote-c8 | sweep-std-b6pq4-t2 | flat-b4pq3-pf15-remote-c4 | sweep-lean-b6pq4-t3 | flat-best-test-remote-c4 | qwen-general-test-c8 | t06-precise-test-c4 | test-lean-b6pq4-c4 | test-set-parvlm-qwen |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| business_report | 6/10 | 5/10 | 6/10 | 5/10 | 3/10 | 6/10 | 4/10 | 4/10 | 5/10 | 4/10 | 5/10 | 4/10 | 3/10 | 4/10 | 2/10 | 4/10 | 5/10 | 4/10 | 4/10 | 5/10 | 5/10 | 6/10 | 4/10 | 2/10 | 4/10 | 3/10 | 4/10 | 2/10 | 3/10 | 3/10 | 4/10 | 3/10 | 1/10 | 4/10 | 2/10 | 4/10 | 3/10 | 5/10 | 4/10 | 3/10 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| comics | 4/10 | 6/10 | 6/10 | 4/10 | 2/10 | 5/10 | 3/10 | 5/10 | 3/10 | 3/10 | 4/10 | 2/10 | 5/10 | 4/10 | 5/10 | 2/10 | 4/10 | 1/10 | 4/10 | 6/10 | 4/10 | 5/10 | 3/10 | 3/10 | 2/10 | 4/10 | 3/10 | 3/10 | 4/10 | 1/10 | 3/10 | 4/10 | 5/10 | 3/10 | 3/10 | 3/10 | 5/10 | 1/10 | 3/10 | 4/10 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| engineering_drawing | 9/10 | 6/10 | 5/10 | 6/10 | 6/10 | 7/10 | 5/10 | 5/10 | 4/10 | 5/10 | 6/10 | 6/10 | 6/10 | 6/10 | 6/10 | 6/10 | 6/10 | 6/10 | 5/10 | 4/10 | 5/10 | 5/10 | 5/10 | 4/10 | 4/10 | 4/10 | 5/10 | 5/10 | 5/10 | 7/10 | 3/10 | 6/10 | 7/10 | 4/10 | 5/10 | 5/10 | 2/10 | 4/10 | 4/10 | 5/10 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| infographics | 8/10 | 7/10 | 7/10 | 7/10 | 8/10 | 5/10 | 6/10 | 5/10 | 6/10 | 7/10 | 6/10 | 8/10 | 4/10 | 4/10 | 6/10 | 6/10 | 6/10 | 6/10 | 6/10 | 6/10 | 6/10 | 6/10 | 4/10 | 7/10 | 8/10 | 7/10 | 6/10 | 5/10 | 7/10 | 5/10 | 6/10 | 5/10 | 4/10 | 6/10 | 6/10 | 5/10 | 6/10 | 5/10 | 4/10 | 2/10 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| maps | 2/10 | 2/10 | 0/10 | 2/10 | 2/10 | 0/10 | 1/10 | 1/10 | 2/10 | 1/10 | 2/10 | 0/10 | 2/10 | 2/10 | 0/10 | 1/10 | 0/10 | 1/10 | 1/10 | 2/10 | 1/10 | 0/10 | 2/10 | 0/10 | 0/10 | 1/10 | 0/10 | 2/10 | 1/10 | 0/10 | 1/10 | 0/10 | 0/10 | 1/10 | 1/10 | 1/10 | 1/10 | 0/10 | 0/10 | 1/10 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| science_paper | 3/10 | 2/10 | 3/10 | 2/10 | 3/10 | 4/10 | 3/10 | 5/10 | 3/10 | 4/10 | 0/10 | 2/10 | 3/10 | 4/10 | 3/10 | 2/10 | 3/10 | 4/10 | 1/10 | 2/10 | 2/10 | 1/10 | 2/10 | 4/10 | 4/10 | 0/10 | 2/10 | 1/10 | 2/10 | 3/10 | 6/10 | 4/10 | 2/10 | 2/10 | 0/10 | 2/10 | 2/10 | 1/10 | 2/10 | 2/10 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| science_poster | 5/10 | 5/10 | 7/10 | 4/10 | 5/10 | 4/10 | 6/10 | 5/10 | 6/10 | 4/10 | 5/10 | 7/10 | 4/10 | 4/10 | 5/10 | 6/10 | 4/10 | 3/10 | 4/10 | 2/10 | 5/10 | 4/10 | 5/10 | 4/10 | 4/10 | 5/10 | 4/10 | 6/10 | 2/10 | 4/10 | 3/10 | 3/10 | 2/10 | 3/10 | 3/10 | 3/10 | 4/10 | 5/10 | 4/10 | 3/10 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| slide | 6/10 | 4/10 | 2/10 | 5/10 | 5/10 | 3/10 | 6/10 | 4/10 | 4/10 | 4/10 | 4/10 | 3/10 | 5/10 | 3/10 | 4/10 | 4/10 | 3/10 | 6/10 | 6/10 | 3/10 | 2/10 | 3/10 | 5/10 | 5/10 | 3/10 | 5/10 | 5/10 | 5/10 | 4/10 | 5/10 | 2/10 | 3/10 | 6/10 | 4/10 | 7/10 | 4/10 | 3/10 | 5/10 | 4/10 | 5/10 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| **Overall** | **43/80 = 53.8%** | **37/80 = 46.2%** | **36/80 = 45.0%** | **35/80 = 43.8%** | **34/80 = 42.5%** | **34/80 = 42.5%** | **34/80 = 42.5%** | **34/80 = 42.5%** | **33/80 = 41.2%** | **32/80 = 40.0%** | **32/80 = 40.0%** | **32/80 = 40.0%** | **32/80 = 40.0%** | **31/80 = 38.8%** | **31/80 = 38.8%** | **31/80 = 38.8%** | **31/80 = 38.8%** | **31/80 = 38.8%** | **31/80 = 38.8%** | **30/80 = 37.5%** | **30/80 = 37.5%** | **30/80 = 37.5%** | **30/80 = 37.5%** | **29/80 = 36.2%** | **29/80 = 36.2%** | **29/80 = 36.2%** | **29/80 = 36.2%** | **29/80 = 36.2%** | **28/80 = 35.0%** | **28/80 = 35.0%** | **28/80 = 35.0%** | **28/80 = 35.0%** | **27/80 = 33.8%** | **27/80 = 33.8%** | **27/80 = 33.8%** | **27/80 = 33.8%** | **26/80 = 32.5%** | **26/80 = 32.5%** | **25/80 = 31.2%** | **25/80 = 31.2%** | **0/160 = 0.0%** | **0/160 = 0.0%** | **0/160 = 0.0%** | **0/160 = 0.0%** | **0/160 = 0.0%** |

### Best per-category

| Category | Best Score | Runs |
|----------|:----------:|------|
| business_report | 60% | full-val-parvlm-pro31-qwen, t06-precise-local-c3, flat-t06-full-remote-c4, qwen-general-lean-remote-c8 |
| comics | 60% | ocr-tool-full-val-qwen-t2, t06-precise-local-c3, flat-t06both-v2-remote-c4 |
| engineering_drawing | 90% | full-val-parvlm-pro31-qwen |
| infographics | 80% | full-val-parvlm-pro31-qwen, flat-pf15-local-c3, t06-precise-local-c3-t2, lean-no-search-c8 |
| maps | 20% | full-val-parvlm-pro31-qwen, ocr-tool-full-val-qwen-t2, parvlm-remote-c8, flat-pf15-local-c3, flat-solo-val-local-c4, t06-precise-lean-local-c3, t06-think-local-c16, gen-think-remote-c16, flat-t06both-v2-remote-c4, sweep-lean-b6pq4-c2-t1, sweep-std-b6pq4-t3 |
| science_paper | 60% | flat-pb-remote-v3-c4 |
| science_poster | 70% | t06-precise-local-c3, t06-precise-local-c3-t2 |
| slide | 70% | sweep-std-b6pq4-c2-t1 |
