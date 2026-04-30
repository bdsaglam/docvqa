# Experiment History

Historical solver experiments and prompt iterations that aren't captured in
`docs/solvers/` or `docs/results.md`. These approaches were tried, evaluated,
and either superseded or abandoned — kept here for context on what's been ruled
out and why.

## Meta Solver (2026-03-29 → 2026-03-31)

`MetaSolver` was a flat-batch-style main agent with an optional `subagent()`
tool that delegated work to another RLM with the same sandbox (pages, look,
search, batch_look). The sub-agent carried its own context window so it didn't
pollute the main agent's REPL.

Config: `solver=meta`, `configs/solver/meta.yaml` (Qwen/Qwen).

| Variant | Mean val | Notes |
|---|---|---|
| v3 (flat-like + optional subagent) | 37.5% | subagent never used — effectively flat |
| v4 (hybrid, nudges delegation) | t2=38%, t3=28% | high variance; comics +30pp on best trial |
| v6 (parallel VLM sub-agents + category tips + budget mgmt) | **31.2%** | worse than flat |

**Conclusion:** Qwen isn't smart enough to orchestrate sub-agents
effectively — overhead exceeds benefit. Iterations spent planning and
delegating are iterations not spent extracting. Subagent calls cost ~2–5 min
each (new subprocess + RLM + PIL loading). Engineering drawing consistently
gained +10–20pp from delegation, but infographics, science_paper, and poster
regressed by similar amounts. **Parallel VLM solver (no orchestration) ended up
the best Qwen/Qwen approach.**

Two subprocess bugs surfaced and were fixed during this work:

- Sandbox code errors swallowed silently → fixed in `subprocess_interpreter.py`.
- Agent could overwrite `pages = []` if sandbox loading wasn't assertive →
  fixed with assertions in the sandbox prelude.

## Parallel VLM Solver

`solver=parallel_vlm` exposes only `batch_look` (no single `look()`), forcing
the agent to issue parallel reads — redundant queries, multi-scale crops, grid
scans, overlapping sweeps. Internal helpers (`_look_impl`, `_batch_look_impl`,
`_search`) are hidden from the LLM prompt; LeanRLM's `action_instructions` is
configurable and ChatAdapter is patched to remove indentation.

| Run | Val |
|---|---|
| v3 | 41.2% |
| v4–v6 | 40.0% |

Consistently around 40% with Qwen/Qwen — about +2pp over flat batch (38.8%),
with the gains concentrated on engineering drawings and comics.

## Parallel RVLM Solver

`solver=parallel_rvlm`. Main agent is LeanRLM and never sees images directly;
`look`/`batch_look` call the VLM (Qwen) via `litellm.completion()` rather than
through `dspy.Predict`. There is no `display()` — the main agent only ever
receives text results. Run with `vlm=qwen lm=qwen max_concurrency=3` (safe for
Qwen on 3×A100).

## Prompt v4 Evolution

Commit `c5c943f` introduced the v4 prompt set, layering on top of the lean v3
prompt. Key additions:

- VLM conflict resolution: crop tighter to break ties, don't trust the last
  read.
- Superlative enumeration: list **all** candidates before selecting.
- Stronger Unknown rules: don't substitute similar entities, respect N/A in
  charts.
- Per-category tips: OCR confusion classes (I/O ↔ 1/0), story map for comics,
  citation regex for papers.
- `python` prefix stripping in `rlm.py` to avoid wasted iterations.

### Iteration sweep (Qwen/Qwen, 4 trials each)

| Config | Mean | Std |
|---|---|---|
| **b4-pq3** | **34.1%** | 3.6% |
| b6-pq2 | 32.5% | 3.8% |
| b6-pq5 | 28.1% | 2.6% |

### V4 dev results (3 trials, 17-doc dev set)

| Setup | Mean | Range |
|---|---|---|
| Baseline (b4-pq3, old prompts) | 31.6% | 28.8–35.6% |
| **V4 (b4-pq3)** | **37.3%** | 35.6–39.0% (+5.7pp) |
| **V4 + HiIter (b6-pq4)** | **39.5%** | 33.9–42.4% (+7.9pp, dev best) |

Higher iteration budget helped hard categories (poster +14pp, slide +21pp,
paper +17pp over V4) but hurt easy ones (infographics −8pp, comics −13pp).
Full-val with V4+b6-pq4 (Qwen/Qwen): t1=35.0%, t2=42.5%, mean=38.8% — beat the
Gemini 3 Pro baseline (37.5%) with a local model. Best categories: infographics
85%, engineering_drawing 50%, poster/slide 40%; worst: maps 10%, science_paper
15%.

## Maps Reasoning

Several approaches tried for spatial path-tracing on map docs:

| Approach | Score (10q) |
|---|---|
| Coordinate graph extraction | 0/10 |
| Visual crawl | 2/10 |
| Original tips | 1–3/10 |

Path tracing is fundamentally hard for Qwen VLM — even single-question focus
fails. Best results were on road-type lookup (legend reading), grid population,
and 2-turn navigation. Complex spatial reasoning needs Pro LLM. Maps remained
the floor (~20%) even after the 3.6 model upgrade — model size doesn't fix it.
