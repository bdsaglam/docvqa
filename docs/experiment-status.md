# Experiment status — DocVQA-2026 (single source of truth)

Last updated 2026-06-10. Quick view of what's **done**, **in progress**, and
**queued/deferred**, plus how to run. Detailed numbers live in
`docs/results.md` (cross-solver) and `docs/experiments/<name>.md` (per-cell).

## ✅ Done (n=8 val, Qwen 3.5 27B unless noted) — all committed

| Experiment | Result | Δ vs `rvlm` 39.38 | Doc |
|---|---|---|---|
| 8-solver matrix (rvlm + ablations + baselines) | rvlm 39.38; tiers hold | — | results.md |
| codeact 3-budget sweep {24,40,56} | pooled 36.74 ± 4.29 | −2.6pp | codeact-qwen-3_5-27b.md |
| nocrop ablation (no crop/zoom) | 36.88 ± 3.20 | −2.5pp (crop is category-specific) | rvlm_nocrop_ablation-…md |
| subagent ablation (general delegation) | 39.22 ± 3.34 | −0.16pp (parity; affordance unused ~1%) | rvlm_subagent_ablation-…md |
| **subagent_full** (sub-call = full agent) | **negative** | +2.4 many-page / −3.5 single-page, within noise; ≈ subagent at 10× cost | rvlm_subagent_full-…md |
| **rvlm_rationale** (VLM answer + `[note:]`) | 39.22 ± 2.91 | −0.16pp (parity; note redundant w/ verify loop) | rvlm_rationale-…md |
| **v3 reasoning-vs-perception** (27B-LM / 9B-VLM vs v2) | RLM +10.3, CodeAct +6.2, ReAct −3.05 | RLM/CodeAct reasoning-bound; ReAct perception-bound | harness-axis-summary.md |
| Gemma-4 E4B harness-lift (n=8) | rvlm 7.34/codeact 7.66/react 6.09 vs base 6.25 | **no lift at 4B** — all 3 within noise of `official_baseline` (clean negative control) | gemma-4-e4b.md |
| Gemma-4 31B harness-lift (n=8; codeact n=5) | rvlm 32.50 (+21.4), codeact 29.25 (+18.2), react 18.44 (+7.4) vs base 11.09 | **every harness ≫ both no-scaffold baselines**; lift is a capacity gate (sharp @31B, absent @4B) | gemma-4-31b.md, harness-axis-summary.md |

**Cross-cutting finding:** enriching the perception sub-call — generality
(subagent), full agency (subagent_full), or a rationale channel
(rvlm_rationale) — does **not** move accuracy on DocVQA val; the minimal
`rvlm` sub-call is sufficient. The REPL crop/zoom loop is what converts
reasoning into perception (v3).

## 🔄 In progress

- _(none — Gemma n=8 sweep + baselines complete; see below.)_

## ⏸ Queued / deferred (not active — need a go-ahead)

- **Dataset/document-length axis (current code) — QUEUED, needs go-ahead.**
  rvlm/codeact/`raw_vlm_multi_baseline`/`official_baseline` (±`rvlm_ocr_ablation`)
  on **MP-DocVQA + MMLongBench-Doc**, Qwen 3.5 27B, with the mandatory
  cross-benchmark rules (dataset-aware profile, `use_profile_scoring=true`,
  **raised page budget** so baselines see evidence pages). The earlier numbers
  are pre-2026-06-01 / invalid. Plan: `tmp/workspace/solver-cmp/DATASET_AXIS_QUEUE.md`.
- **Gemma n>2 escalation** — _done_ (n=8 harnesses + n=8 baselines, both models;
  31B codeact n=5 stopped early per user — Done rows above + harness-axis-summary.md).
- **rvlm_rationale / subagent / subagent_full on long-doc** (MMLongBench-Doc) —
  the perception-sub-call enrichments are null on short DocVQA; the open
  question is whether they pay off where routing is harder (long multi-page).
- **Model-axis: Gemma other sizes / base-vs-it** — only E4B + 31B configured.
- **Test-set submission** (rvlm SC-vote) — see coordination/amax7.md.
- **Smaller-model families beyond Gemma** (per the 27B-only directive) — parked.

## 🖥 Infra (amax1)

- 27B is docker `qwen35-27b` @8927; **currently DP=1 on GPU 0** (kept up;
  Gemma 31B pilots done, GPUs 1–2 now free — tear down `gemma-31b` or
  re-expand 27B to DP/TP across all 3 when convenient). GPU recipes + the
  device-quoting gotcha:
  `tmp/workspace/solver-cmp/GPU_SWITCH_PLAN.md`, `GEMMA_PHASE_PLAN.md`.
- **Always keep a 27B up.** Serve one model per full GPU set sequentially;
  DP for small models, TP for large.
- **Gemma serving:** canonical image is **`vllm/vllm-openai:gemma4`** +
  **`--reasoning-parser gemma4`** (see `docs/scratchpad.md`), NOT generic
  `:latest`. Image entrypoint is already `["vllm","serve"]` → container
  command must start with `--port` (a leading `serve` → instant exit(2)).

## How to run (canonical commands)

```bash
# rvlm (proposed method), 27B homog, n=1
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false solver=rvlm data.split=val data.num_samples=null \
  max_concurrency=16 run_id=rvlm-val-t1
# swap solver= for the variant: codeact | rvlm_ocr_ablation | rvlm_hybrid_ablation |
#   rvlm_nocrop_ablation | rvlm_subagent_ablation | rvlm_subagent_full | rvlm_rationale |
#   react_baseline | raw_vlm_multi_baseline | direct_vlm | official_baseline | rlm_ocr
# cross-model: lm=/vlm= one of qwen-3_5-{4b,9b,27b}- / gemma-4-{e4b,31b}-vllm-local
# report: python scripts/report.py --all   |   per-run iters: python scripts/iter_stats.py '<glob>'
```

Concurrency convention: c=16–24 on a healthy 27B; lower (c=4–8) for
heavy/nested solvers (subagent_full, codeact on long docs) and small/slow
servers.
