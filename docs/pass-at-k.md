# pass@k & SC@k — oracle ceiling and self-consistency (DocVQA-2026 val)

**Diagnostic, not headline.** Per [D-003](paper/decisions.md) the paper's
headline reporting is **mean ± std across independent trials** (`avg@1` below)
and self-consistency is out of the method framing. This page reports two extra
per-cell numbers for analysis only:

- **pass@k** — *oracle ceiling*: a question counts correct if **any** of the
  cell's `k` trials got it right. Upper bound an ideal trial-selector / verifier
  could reach. The gap `pass@k − avg@1` is the headroom a re-ranker or RL
  reward model could in principle recover.
- **SC@k** — *self-consistency*: majority-vote the `k` trial answers, then score
  the vote. What post-hoc voting (the old competition tactic) would buy.

## Method

Computed by [`scripts/pass_at_k.py`](../scripts/pass_at_k.py):

- A **cell** is one config run as trials `<run_id>-t1..-tN`.
- Scoring is the **same binary DocVQA-2026 correctness** the runner uses
  (`docvqa.metrics.evaluate_prediction`; strict numeric/date match + relaxed
  ANLS ≥ 0.9). `avg@1` reproduces the documented mean ± std (sanity: codeact_chat
  27B 39.53, 4b/27b 22.34, gemma cells exact).
- SC vote reuses the established
  [`scripts/vote_submissions.py:vote`](../scripts/vote_submissions.py)
  (group by normalized form; ties → longest canonical, then first-seen).
- **Incomplete trials are dropped**: only trials at the cell's max gold-question
  coverage are kept, so pass@k / SC@k are over the full 80-question set. `k`
  below is the kept-trial count (≤ the cell's `n`); `n_q` is the scored
  question count.

Reproduce:

```bash
uv run python -m scripts.pass_at_k --all --min-trials 3 --markdown   # all multi-trial cells
uv run python -m scripts.pass_at_k codeact-chat-val rvlm-vsearch-val  # specific cells
```

## Headline 8-solver matrix — RE-RUN in progress (2026-06-17)

The published Qwen-27B headline matrix (`rvlm-cmp-val`, `react-cmp-val`,
`raw-vlm-multi-cmp-val`, `official-cmp-val`, `rlm-ocr-cmp-val`, +`rvlm_*`
ablations) had its per-trial `submission.json`s **deleted on both hosts** (disk
cleanup), so its pass@k/SC@k were unrecoverable. It is now being **re-run** on a
local 27B DP=3 (rvlm+ablations) + a remote 27B (light baselines) — fresh
artifacts, fresh `avg@1` re-rolled within trial noise. Recovered so far:

| cmp cell | k | avg@1 (±std) | pass@k | SC@k | vs old avg@1 |
|---|---|---|---|---|---|
| **`rvlm-cmp-val` (proposed M, headline)** | 8 | **41.88 ± 5.79** | **68.75** | **47.50** | 39.38 → **+2.5** (re-rolls higher) |
| `rlm-ocr-cmp-val` (OCR-floor control) | 8 | **14.69 ± 2.19** | 27.50 | 15.00 | 13.91 → +0.8 (✓ reproduces) |

`rvlm-cmp` per-trial: 30.0 / 43.8 / 45.0 / 50.0 / 43.8 / 42.5 / 41.2 / 38.8 (n=8) —
the fresh re-roll lands **+2.5pp above the deleted-run 39.38**, with higher std
(5.79 vs the old 1.49; t1's 30.0 is the low outlier, the other 7 cluster 38.8–50.0).
In progress: react / raw_vlm_multi / official baselines + the rvlm ablations
(rvlm_ocr/nocrop/subagent/hybrid). The surviving
`rvlm-minimal/-unified/-skeletal-val` cells are **undocumented earlier
prompt-scrub variants** (not the published `*-cmp-val` runs) and are labeled as
variants below.

## Results (cells with retained artifacts)

`avg@1` = 8-trial-style mean ± std (the headline metric, recomputed here).

### Proposed-method tier — `rvlm` / `codeact_chat` (Qwen 3.5 27B homog)

| Cell | config | k | avg@1 (±std) | pass@k | SC@k |
|---|---|---|---|---|---|
| `codeact-chat-val` | codeact_chat 27B homog (corrected twin) | 8 | 39.53 ± 2.83 | **63.75** | 45.00 |
| `codeact-chat-think-val` | codeact_chat 27B + thinking | 7 | 37.68 ± 4.42 | 58.75 | 38.75 |
| `rvlm-vsearch-val` | rvlm + visual-retrieval ext (OCR-free) | 6 | 36.67 ± 2.58 | 66.25 | 37.50 |
| `rvlm-minimal-val` | rvlm 27B — *prompt variant*¹ | 8 | 42.03 ± 2.21 | 67.50 | 47.50 |
| `rvlm-unified-val` | rvlm 27B — *prompt variant*¹ | 8 | 40.94 ± 4.05 | 71.25 | 45.00 |
| `rvlm-skeletal-val` | rvlm 27B — *prompt variant*¹ | 6 | 39.17 ± 2.19 | 60.00 | 42.50 |
| `rvlm-hybrid-val` | rvlm + display channel — *variant*¹ | 8 | 36.72 ± 4.12 | 66.25 | 41.25 |

¹ Earlier prompt-scrub variants, **not** the published `*-cmp-val` headline runs.

### codeact_chat model axis (no-think)

| Cell | config | k | avg@1 (±std) | pass@k | SC@k |
|---|---|---|---|---|---|
| `codeact-chat-27b-llm-9b-vlm-val` | v3 27B-LM / 9B-VLM | 3 | 32.92 ± 4.39 | 51.25 | 32.50 |
| `codeact-chat-4b-llm-27b-vlm-val` | 4B-LM / 27B-VLM | 8 | 22.34 ± 3.44 | 55.00 | 26.25 |
| `codeact-chat-4b-val` | 4B homog | 8 | 16.25 ± 2.00 | 47.50 | 20.00 |

### Cross-model harness axis — RLM (`rvlm-minimal-*`) / CodeAct (old dspy) / ReAct

| Cell | config | k | avg@1 (±std) | pass@k | SC@k |
|---|---|---|---|---|---|
| `rvlm-minimal-27b-llm-9b-vlm-val` | RLM v3 27B/9B | 2 | 36.88 ± 0.88 | 48.75 | 33.75 |
| `codeact-3_5-27b-val` | CodeAct 27B homog (old dspy) | 5 | 37.50 ± 4.24 | 56.25 | 46.25 |
| `codeact-27b-llm-9b-vlm-val` | CodeAct v3 27B/9B (old dspy) | 4 | 32.19 ± 4.00 | 51.25 | 32.50 |
| `rvlm-minimal-9b-llm-27b-vlm-val` | RLM v2 9B-LM / 27B-VLM | 8 | 25.31 ± 4.16 | 58.75 | 31.25 |
| `codeact-9b-llm-27b-vlm-val` | CodeAct 9B-LM / 27B-VLM | 8 | 24.22 ± 3.78 | 57.50 | 33.75 |
| `react-9b-llm-27b-vlm-val` | ReAct 9B-LM / 27B-VLM | 8 | 22.66 ± 4.93 | 56.25 | 27.50 |
| `codeact-3_5-9b-val` | CodeAct 9B homog (old dspy) | 8 | 21.56 ± 4.11 | 57.50 | 26.25 |
| `rvlm-minimal-4b-llm-27b-vlm-val` | RLM v2 4B-LM / 27B-VLM | 8 | 21.09 ± 3.16 | 56.25 | 26.25 |
| `react-8b-llm-27b-vlm-val` | ReAct 8B-LM / 27B-VLM | 8 | 19.69 ± 1.29 | 37.50 | 21.25 |
| `rvlm-minimal-3_5-9b-val` | RLM 9B homog (v1) | 8 | 18.91 ± 3.81 | 47.50 | 28.75 |
| `react-27b-llm-9b-vlm-val` | ReAct v3 27B/9B | 8 | 18.28 ± 3.40 | 50.00 | 21.25 |
| `react-4b-llm-27b-vlm-val` | ReAct 4B-LM / 27B-VLM | 8 | 18.12 ± 4.06 | 52.50 | 23.75 |
| `codeact-4b-llm-27b-vlm-val` | CodeAct 4B-LM / 27B-VLM | 8 | 16.88 ± 3.13 | 50.00 | 22.50 |
| `react-3_5-9b-val` | ReAct 9B homog | 8 | 16.25 ± 3.06 | 46.25 | 21.25 |
| `codeact-3_5-4b-val` | CodeAct 4B homog (old dspy) | 8 | 14.22 ± 4.28 | 45.00 | 17.50 |
| `rvlm-minimal-3_5-4b-val` | RLM 4B homog (v1) | 8 | 14.22 ± 3.83 | 46.25 | 20.00 |
| `rvlm-minimal-8b-llm-27b-vlm-val` | RLM 8B-LM / 27B-VLM | 8 | 13.91 ± 2.71 | 40.00 | 16.25 |
| `react-3_5-4b-val` | ReAct 4B homog | 8 | 13.44 ± 1.98 | 38.75 | 13.75 |
| `codeact-8b-llm-27b-vlm-val` | CodeAct 8B-LM / 27B-VLM | 8 | 12.34 ± 1.82 | 35.00 | 11.25 |

### Gemma harness-lift (val)

| Cell | config | k | avg@1 (±std) | pass@k | SC@k |
|---|---|---|---|---|---|
| `gemma-31b-rvlm-val` | Gemma-4 31B rvlm | 7 | 33.04 ± 4.56 | 60.00 | 42.50 |
| `gemma-31b-codeact-val` | Gemma-4 31B codeact (old dspy) | 5 | 29.25 ± 5.77 | 52.50 | 38.75 |
| `gemma-31b-react-val` | Gemma-4 31B react | 4 | 18.44 ± 3.73 | 31.25 | 17.50 |
| `gemma-31b-official-val` | Gemma-4 31B official baseline | 8 | 11.09 ± 1.82 | 17.50 | 11.25 |
| `gemma-31b-rawvlm-val` | Gemma-4 31B raw-vlm baseline | 8 | 10.78 ± 0.93 | 12.50 | 11.25 |
| `gemma-e4b-codeact-val` | Gemma-4 E4B codeact (old dspy) | 8 | 7.66 ± 1.94 | 17.50 | 10.00 |
| `gemma-e4b-rvlm-val` | Gemma-4 E4B rvlm | 8 | 7.34 ± 3.30 | 27.50 | 8.75 |
| `gemma-e4b-official-val` | Gemma-4 E4B official baseline | 8 | 6.25 ± 1.16 | 7.50 | 7.50 |
| `gemma-e4b-react-val` | Gemma-4 E4B react | 8 | 6.09 ± 2.36 | 16.25 | 7.50 |
| `gemma-e4b-rawvlm-val` | Gemma-4 E4B raw-vlm baseline | 8 | 3.75 ± 0.00 | 3.75 | 3.75 |

### Legacy cells (pre-D-010 names; n=3)

| Cell | config | k | avg@1 (±std) | pass@k | SC@k |
|---|---|---|---|---|---|
| `leanest-solo-val` | leanest_solo (legacy) | 3 | 40.00 ± 0.00 | 52.50 | 45.00 |
| `flat-solo-m5-val` | flat_solo (legacy) | 3 | 30.00 ± 0.00 | 43.75 | 33.75 |
| `no-loop-multi-tips-val` | no_loop_multi + tips (legacy) | 3 | 23.75 ± 2.17 | 26.25 | 25.00 |
| `no-loop-multi-val` | no_loop_multi (legacy) | 3 | 22.50 ± 1.25 | 25.00 | 23.75 |
| `no-loop-tips-val` | no_loop + tips (legacy) | 3 | 21.25 ± 1.25 | 25.00 | 21.25 |
| `no-loop-val` | no_loop single (legacy) | 3 | 17.08 ± 2.60 | 21.25 | 17.50 |

## Findings

1. **Large oracle headroom on the strong scaffolds.** On the proposed tier,
   pass@k roughly *doubles* avg@1: `codeact-chat` 39.5 → **63.8** (+24.2pp),
   `rvlm-vsearch` 36.7 → **66.3** (+29.6pp), `rvlm-minimal` 42.0 → **67.5**.
   Most failures are not "the scaffold can't" but "the scaffold is inconsistent"
   — a trial-selector / verifier / RL reward model has ~25pp of recoverable
   signal. This is the strongest argument for a verification or best-of-n stage.
2. **SC@k is a real, free lift on strong cells but flat on weak ones.** On the
   top tier SC@8 beats avg@1 by ~5pp (`codeact-chat` 39.5 → 45.0; `rvlm-minimal`
   42.0 → 47.5) — voting recovers part of the variance. On weak cells (avg@1 ≲
   12%) SC ≈ avg@1 or worse (e.g. `codeact-8b` 12.3 → SC 11.3): with mostly-wrong
   trials, the majority is wrong, so voting can't help. SC's value scales with
   base accuracy.
3. **pass@k separates "perception-budget-bound" from "reasoner-bound" cleanly.**
   Holding the reasoner weak and giving it the 27B VLM (v2 cells), pass@k is high
   even where avg@1 is low — `rvlm-minimal-4b-llm-27b` avg@1 21.1 but pass@8
   **56.3**; `react-4b-llm-27b` 18.1 → 52.5. The right answer is reachable on
   most questions with strong perception; the small reasoner just can't land it
   *consistently*. Where perception is the bottleneck instead (homog small, or
   8B-LM/27B), pass@k is markedly lower (`rvlm-minimal-8b` 13.9 → 40.0).
4. **Gemma E4B is a genuine floor, not just low-mean.** Even pass@8 stays tiny
   (rvlm 27.5, official 7.5, rawvlm 3.75) — the 4B model rarely gets it right in
   *any* of 8 trials, confirming the capacity-gate reading (no recoverable signal
   to re-rank), unlike the small-Qwen cells above.

## TODO

- **Re-run the headline `*-cmp-val` matrix** (n=8, Qwen 27B) to recover its
  pass@8 / SC@8 (artifacts not retained). The 27B server is up; gated on the
  paused-campaign resume decision.
