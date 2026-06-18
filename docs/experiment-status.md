# Experiment status — DocVQA-2026 (single source of truth)

Last updated 2026-06-14. Quick view of what's **done**, **in progress**, and
**queued/deferred**, plus how to run. Detailed numbers live in
`docs/results.md` (cross-solver) and `docs/experiments/<name>.md` (per-cell).

## ✅ Done (n=8 val, Qwen 3.5 27B unless noted) — all committed

| Experiment | Result | Δ vs `rvlm` 39.38 | Doc |
|---|---|---|---|
| 8-solver matrix (rvlm + ablations + baselines) | rvlm 39.38; tiers hold | — | results.md |
| **codeact_chat** (corrected codeact: true chat MDP, no dspy) | **39.53 ± 2.83**; +2.7pp vs old codeact | +0.15pp (tied) | codeact-chat-qwen-3_5-27b.md |
| codeact_chat thinking ablation (n=7) | 37.68 ± 4.42; −1.85pp vs no-think | thinking = **no gain** (worse/slower/hang-prone) | codeact-chat-qwen-3_5-27b.md |
| **codeact_chat 4b/27b** (4B-LM / 27B-VLM, n=8) | **22.34 ± 3.44** | +6.7pp vs old codeact; +6.5pp vs 4b-homog (perception-budget lift) | codeact-chat-qwen-3_5-27b.md |
| **codeact_chat 4b-homog** (4B/4B, n=8) | **16.25 ± 2.00** | +4.1pp vs old codeact; +3.76pp vs rvlm 4b-homog | codeact-chat-qwen-3_5-27b.md |
| codeact_chat v3 (27B-LM / 9B-VLM, n=3) | 32.9 | +2.5pp vs old codeact | codeact-chat-qwen-3_5-27b.md |
| nocrop ablation (no crop/zoom) | 36.88 ± 3.20 | −2.5pp (crop is category-specific) | rvlm_nocrop_ablation-…md |
| subagent ablation (general delegation) | 39.22 ± 3.34 | −0.16pp (parity; affordance unused ~1%) | rvlm_subagent_ablation-…md |
| **subagent_full** (sub-call = full agent) | **negative** | +2.4 many-page / −3.5 single-page, within noise; ≈ subagent at 10× cost | rvlm_subagent_full-…md |
| **rvlm_rationale** (VLM answer + `[note:]`) | 39.22 ± 2.91 | −0.16pp (parity; note redundant w/ verify loop) | rvlm_rationale-…md |
| **v3 reasoning-vs-perception** (27B-LM / 9B-VLM vs v2) | RLM +10.3, CodeAct +6.2ᶜ, ReAct −3.05 | RLM/CodeAct reasoning-bound; ReAct perception-bound | harness-axis-summary.md |
| Gemma-4 E4B harness-lift (n=8) | rvlm 7.34/codeact 7.66ᶜ/react 6.09 vs base 6.25 | **no lift at 4B** — all 3 within noise of `official_baseline` (clean negative control) | gemma-4-e4b.md |
| Gemma-4 31B harness-lift (n=8; codeact n=5) | rvlm 32.50 (+21.4), codeact 29.25ᶜ (+18.2), react 18.44 (+7.4) vs base 11.09 | **every harness ≫ both no-scaffold baselines**; lift is a capacity gate (sharp @31B, absent @4B) | gemma-4-31b.md, harness-axis-summary.md |

ᶜ **STALE — do not cite.** Old dspy `codeact` (deprecated). The corrected
**`codeact_chat`** twin is the sole source of truth for CodeAct numbers going
forward; a config without a `codeact_chat` value is **open** — the stale dspy
figure is shown for provenance only, not as a current result. Tracking and
replacements: `codeact-chat-qwen-3_5-27b.md`.

**Cross-cutting finding:** enriching the perception sub-call — generality
(subagent), full agency (subagent_full), or a rationale channel
(rvlm_rationale) — does **not** move accuracy on DocVQA val; the minimal
`rvlm` sub-call is sufficient. The REPL crop/zoom loop is what converts
reasoning into perception (v3).

## ✅ Headline 9-solver matrix re-run — COMPLETE (val, n=8)

The published `*-cmp-val` matrix lost its per-trial artifacts → no pass@8/SC@8.
Re-ran all 9 cells with retained artifacts: **8/9 recovered with the full
triple** (rvlm 41.88 · rvlm_ocr 36.56 · rvlm_nocrop 35.78 · rvlm_subagent 36.72 ·
react 27.19 · raw_vlm_multi 20.94 · official 18.91 · rlm_ocr 14.69 — see
`pass-at-k.md`). The 9th, **`rvlm_hybrid`**, is **closed as a documented
limitation**: headline 35.47% ± 4.48 retained, pass@8/SC@8 **unrecoverable** —
its `+display()` channel emits ~163k-token requests on heavy docs that overflow
both the 32k local (→"Unknown") and 131k remote (→scaffold spin-loop, no
exec-timeout). Needs a solver fix, not more context.

## 🔄 In progress

- **Test-set submission (T4): `rvlm`** — 4 trials @ c=4 on local 27B (toward
  SC-8; 48 test docs, no gold → vote → submission JSON).

## 🎯 Paper-completion queue (user directive 2026-06-18: "all experiments must be done for the paper")

Full runnable detail + commands: **`tmp/workspace/paper-completion-2026-06-18/QUEUE.md`**.
Heartbeat-driven (cron, no chain scripts). Standing recording rule: every
completed cell → mean±std + pass@8 + SC@8 into `pass-at-k.md` + `results.md`
+ its experiment doc; commit+push.

- **T0 — finish the matrix re-run** (above): `rvlm_subagent` + `rvlm_hybrid` → n=8.
- **T1 — `codeact_chat` grid completion** (retire stale dspy `codeact` at every
  still-cited config; val/80-Q, no-think, n=8): **9b/27b**, **8b/27b**,
  **v3 27B/9B** finish (t4-t8), **gemma-31B** (n=4, gemma4 image). Done already:
  27B-homog 39.53 · 9b-homog 22.97 · 4b-homog 16.25 · 4b/27b 22.34 · gemma-E4B 6.56.
- **T2 — Phase-4 27B/4B rung × 3 harnesses** (reasoner-fixed perception-ladder
  bottom rung; split 27B+4B, n=8): **`rvlm`**, **`codeact_chat`**, **`react_baseline`**.
- **T3 — dataset / document-length axis** (current code, Qwen 27B homog;
  **mandatory** cross-benchmark rules — dataset-aware profile,
  `use_profile_scoring=true`, **raised page budget**). Datasets: **MP-DocVQA**
  + **MMLongBench-Doc**. Solvers: `rvlm`, `codeact_chat`, `raw_vlm_multi_baseline`,
  `official_baseline`, `rvlm_ocr_ablation` (OCR-extension long-doc payoff test).
  n=1 → escalate. Earlier numbers pre-2026-06-01 / invalid. Plan:
  `tmp/workspace/solver-cmp/DATASET_AXIS_QUEUE.md`.
- **T4 — test-set submission** (competition; 48 test docs, no gold → SC-vote →
  submission JSON). Main solvers only: **`rvlm`**, **`codeact_chat`**,
  **`react_baseline`** (SC-8 each).

### Parked (not in the paper-completion set)
- **rvlm_rationale / subagent / subagent_full on long-doc** — perception-sub-call
  enrichments are null on short DocVQA; long-doc payoff is an open extension, not
  a headline cell.
- **Model-axis: Gemma other sizes / base-vs-it; smaller-model families beyond
  Gemma** — parked per the 27B-only directive.

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
