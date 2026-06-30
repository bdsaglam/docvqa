# Experiment status — DocVQA-2026 (single source of truth)

Quick view of what's **done**, **in progress**, and **parked**, plus how to
run. Detailed numbers live in `docs/results.md` (cross-solver), `docs/pass-at-k.md`
(pass@k / SC@k diagnostic), and `docs/experiments/<name>.md` (per-cell). All
numbers are **current code** (post-2026-06-01: minimized/parity-stripped prompts
+ per-call `num_retries=5`); pre-change numbers are archived and invalid.

Headlines are **mean ± std across trials** (no SC voting per D-003). The
proposed method is **`rvlm`** (REPL + recursive VLM `batch_look`, OCR-free); its
append-only/MDP twin is **`codeact_chat`** (the FT target). The old dspy
`codeact` is **deprecated/archived** — `codeact_chat` is the sole source of truth
for CodeAct numbers; where a config has a `codeact_chat` value it supersedes the
dspy figure entirely.

## ✅ Done

### Headline 27B matrix — val 25-doc/80-Q, n=8

Nine-solver comparison, Qwen 3.5 27B homog (`enable_thinking=false`), with
retained per-trial artifacts (pass@8/SC@8 in `pass-at-k.md`). **8/9 cells carry
the full metric triple; `rvlm_hybrid` is an accepted failure** (its `display()`
channel emits ~163k-token requests on heavy docs, exceeding Qwen 3.5 27B's 131k
context — per policy that counts as the solver failing, not a harness gap; the
35.47% ± 4.48 figure is an upper bound, true score lower).

| Solver | Role | Val n=8 |
|---|---|---|
| **`rvlm`** | proposed — REPL + recursive `batch_look`, OCR-free | **41.88% ± 5.79** |
| `codeact_chat` | append-only/MDP twin (FT target) | 39.53% ± 2.83 (within ~2pp of `rvlm`) |
| `rvlm_subagent_ablation` | generalized `batch_subagent` | 36.72% ± 2.75 (within combined std) |
| `rvlm_ocr_ablation` | + OCR + BM25 (OCR extension) | 36.56% ± 2.89 (OCR adds ~0) |
| `rvlm_nocrop_ablation` | no crop/zoom | 35.78% ± 2.31 |
| `react_baseline` | perception tools, no REPL | 27.19% ± 3.19 (REPL load-bearing) |
| `direct_vlm` | pages into own context, no sub-call | 22.34% ± 2.79† (sub-call load-bearing) |
| `raw_vlm_multi_baseline` | raw multi-image, no scaffold | 20.94% ± 1.60 (scaffold floor) |
| `official_baseline` | competition `MASTER_PROMPT`, no scaffold | 18.91% ± 1.94 (external anchor) |
| `rlm_ocr` | RLM + OCR, no vision | 14.69% ± 2.19 (OCR-free control / floor) |

> Numbers are from the **canonical re-run** (retained per-trial artifacts → the
> pass@k/SC@k triple). The original `*-cmp-val` batch lost its per-trial files;
> the re-run reproduces every tier (rvlm re-rolls +2.5pp to 41.88, std inflated
> by a single t1=30 outlier). †`direct_vlm` and `rvlm_rationale` (39.22, parity)
> were **not** in the re-run; their numbers are original-batch (see `results.md`).

**Three clean tiers + cross-cutting reads.** Visual-recursive (`rvlm` ~42% /
`codeact_chat` ~40%) ≫ no-recursion (`react`/`direct_vlm`/`raw_vlm_multi` 21–27%) ≫ OCR-only floor
(`rlm_ocr` 14%). Both halves of the scaffold are load-bearing (drop REPL → `react`;
drop sub-call → `raw_vlm_multi`/`direct_vlm`). Enriching the sub-call — generality
(`subagent`), full agency (`subagent_full`), a rationale channel (`rvlm_rationale`)
— moves accuracy ~0 on DocVQA val; the minimal `batch_look` is sufficient. OCR on
top of vision (`rvlm_ocr`) or a direct image channel (`rvlm_hybrid`) buys ~0.

### `codeact_chat` model-axis grid — val/80-Q, no-think, n=8 (gemma-31B n=4)

Retires the stale dspy `codeact` at every still-cited config. `codeact_chat`
**ties `rvlm` across the model axis in both families** — the corrected MDP loop
catches up to the proposed method at no accuracy cost (it does not beat it).

| Config (LM / VLM) | `codeact_chat` n=8 | vs `rvlm` same config |
|---|---|---|
| 27B / 27B (homog) | 39.53% ± 2.83 | +0.15 (tie) |
| 27B / 9B (v3) | 32.81% ± 3.04 | reasoner-fixed middle rung |
| 9B / 27B | 26.56% ± 4.21 | perception lift one rung up |
| 9B / 9B (homog) | 22.97% ± 2.75 | +4.1 vs `rvlm` 9b-homog (18.91) |
| 8B / 27B (Qwen3-8B, older gen) | 16.72% ± 3.20 | off-axis weak reasoner (32% Unk, page-scans) |
| 4B / 27B | 22.34% ± 3.44 | +1.25 |
| 4B / 4B (homog) | 16.25% ± 2.00 | +3.76 (borderline) |
| gemma-4-31B homog (n=4) | 30.31% ± 2.13 | −2.7 vs `rvlm` 33.04 |
| gemma-4-E4B homog | 7.81% ± 1.86 | floor (fix barely moves a 4B) |

> **gemma stop-token fix:** gemma-4 does not emit a turn-final stop, so
> `codeact_chat` had role-played the whole rollout in one completion (bogus 5%);
> `_split_first_turn` enforces one action per turn → 30.31% at 31B.

### Phase-4 — perception-ladder bottom rung (27B/4B), n=4

Strong reasoner × weakest VLM, per harness. The harness hierarchy holds at the
bottom rung: **`rvlm` 32.81% ± 3.13 > `codeact_chat` 27.19% ± 2.77 > `react`
20.00% ± 3.54** (they tie at 27B-homog; the gap opens as the VLM weakens).

### Model-size / VLM-quality axis (prediction 1) — n=8

Hold the reasoner fixed, swap **only** the VLM →27B: at both 9B (+7.87pp) and 4B
(+8.60pp) the `rvlm` headline lifts ~8pp — the signature of a **perception-budget**
(not reasoning) bottleneck (supports D-006). Gemma confirms it's a capacity gate:
sharp harness-lift at 31B (`rvlm` +21.4pp over baseline), absent at E4B (all
harnesses within noise of `official_baseline`). Older-gen Qwen3-8B reasoner is a
clean reasoner-*quality* point (11.73%, off the 9B↔4B size curve). Detail:
`harness-axis-summary.md` + by-model files.

### Test-set submissions (competition; 48 docs, no gold → SC-vote)

- **`rvlm`** — SC-8 → `submissions/rvlm-test-sc8.json`.
- **`codeact_chat`** — SC-8 → `submissions/codeact-chat-test-sc8.json`.
- `react_baseline` test — **skipped** (user): the two recursive submissions
  suffice; a no-REPL baseline isn't worth the ~2-day heavy-doc grind.

## 🔄 In progress

### Dataset / document-length axis (prediction 2) — escalating to n=3

Current code, Qwen 27B homog, **main solvers only** (`rvlm`, `codeact_chat`,
`official_baseline`, `raw_vlm_multi_baseline`). Mandatory cross-benchmark rules
applied: dataset-aware profile, `use_profile_scoring=true`, raised page budget.
Stratified-random subsets (`scripts/stratified_sample.py`, seed 0). n=1 is in and
reads as predicted; **escalating n=1 → n=3** (16 runs: 4 solvers × 2 datasets ×
trials 2–3), two-lane (MMLongBench judge-scored / MP-DocVQA ANLS), c=4.

n=1 result — the recursive-perception advantage **scales with document length**:

| Solver | MP-DocVQA (≤20pg, ANLS) | MMLongBench-Doc (~47pg, judge) |
|---|---|---|
| `codeact_chat` | 64.4% | 65.8% (0% Unk) |
| `rvlm` | 60.8% | 66.5% (0% Unk) |
| `official_baseline` | 58.8% (8% Unk) | 49.7% (36% Unk) |
| `raw_vlm_multi` | 58.2% (22% Unk) | 24.2% (87% Unk) |
| recursive − baseline gap | **~2–6pp** | **~16–42pp** |

Recursive methods are flat (~61–66%, ~0% Unknown); baselines degrade with length
as the fixed page budget misses later-page evidence (the Unknown-rate ladder is
the mechanism). Detail: `dataset-axis-{mp-docvqa,mmlongbench}.md`, `results.md`
("Document-length axis"). OCR-extension long-doc payoff (`rvlm_ocr`) is untestable
here (no OCR data for these benchmarks) — skipped per main-solvers scope.

## ⏸ Parked (not in the paper-completion set)

- **Perception-sub-call enrichments on long-doc** (`rvlm_rationale` / `subagent` /
  `subagent_full`) — null on short DocVQA; long-doc payoff is an open extension.
- **Model-axis beyond the above** (other Gemma sizes, base-vs-it, non-Qwen/Gemma
  families) — parked per the 27B-headline directive.

## 🖥 Infra (amax1)

- 27B is docker `qwen35-27b` @8927, **DP=3 across all 3 GPUs**, started with
  `--limit-mm-per-prompt {"image":32}` so multi-image baselines run alongside the
  recursive solvers (which send 1 image/`batch_look`). **Always keep a 27B up**;
  serve one model per full GPU set, DP for small models, TP for large.
- **MMLongBench judge** runs on the live 27B via
  `QWEN_JUDGE_BASE_URL=http://localhost:8927/v1` (judge model = Qwen3.5-27B).
- **Gemma serving:** image `vllm/vllm-openai:gemma4` + `--reasoning-parser gemma4`
  (TP=2, `--shm-size=32g --ipc=host`, c=2); container command must start with
  `--port` (a leading `serve` → instant exit(2)). Recipes in `docs/scratchpad.md`.

## How to run (canonical commands)

```bash
# rvlm (proposed method), 27B homog, n=1
uv run python evals.py lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false solver=rvlm data.split=val data.num_samples=null \
  max_concurrency=16 run_id=rvlm-val-t1
# swap solver= for a variant: codeact_chat | rvlm_ocr_ablation | rvlm_hybrid_ablation |
#   rvlm_nocrop_ablation | rvlm_subagent_ablation | rvlm_subagent_full | rvlm_rationale |
#   react_baseline | raw_vlm_multi_baseline | direct_vlm | official_baseline | rlm_ocr
# cross-model: lm=/vlm= one of qwen-3_5-{4b,9b,27b}- / qwen-3-8b- / gemma-4-{e4b,31b}-vllm-local
# cross-benchmark: + data.dataset=<id> data.use_profile_scoring=true (raise solver.max_pages)
# report: python scripts/report.py --all   |   per-run iters: python scripts/iter_stats.py '<glob>'
```

Concurrency: c=16–24 on a healthy 27B; lower (c=4–8) for heavy/nested solvers
(`subagent_full`, `codeact_chat` on long docs), small/slow servers, and long-doc
benchmarks.
