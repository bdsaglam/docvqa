# rvlm_rationale — Qwen 3.5 27B (val)

## Hypothesis / question

Fork of `rvlm`: the VLM perception sub-call (`batch_look`) returns, on EVERY
call, a clean `answer` PLUS a short **observation/uncertainty note** (a
separate dspy field), surfaced to the main agent as `"<answer>  [note:
<observation>]"`. The note says what's visible/legible and flags doubt
(blurry/ambiguous chars, a region clipped at the crop edge, the value not on
this image). Framing is **observation+uncertainty, not justification**, to
avoid the post-hoc-confidence trap.

Question: does this diagnostic signal help the main agent steer its next
action (re-crop wider, try another page, trust vs re-verify) — i.e. does a
richer perception channel beat the terse `rvlm` answer?

## Setup

- Solver: `rvlm_rationale` (clean `answer` field + `rationale` field; batch_look
  returns `answer [note: ...]`). Otherwise byte-identical scaffold to `rvlm`.
- Model: Qwen 3.5 27B local vllm 8927 (lm + vlm), `enable_thinking=false`.
- Profile: DocVQA-2026; max_iterations 25; max_concurrency 16; n=8.

## Result

| run | rvlm_rationale (n=8) | rvlm (n=8) | Δ |
|---|---|---|---|
| val (25 docs / 80 Q) | **39.22% ± 2.91** | 39.38% ± 1.49 | **−0.16pp** |

Per-trial: 38.75 / 37.50 / 43.75 / 40.00 / 38.75 / 40.00 / 41.25 / 33.75.

**Dead parity.** Adding the observation/uncertainty note to every perception
read **neither helps nor hurts** accuracy on DocVQA-2026 val.

**The channel is real and used by the VLM:** across the 8 trials there are
**19,632** `[note: ...]` annotations, of which **~7% (1,396)** explicitly
flag uncertainty (blur / clipped region / ambiguous char / value-not-present
/ illegible). So the VLM does surface doubt — the note is not vacuous.

**Why no accuracy change (read):** `rvlm`'s prompt *already* treats every VLM
read as unreliable and runs a verify loop (re-read, re-crop, cross-check)
regardless of any note. The explicit uncertainty signal is therefore
**redundant with a discipline the agent already applies** — it doesn't
trigger meaningfully more (or better-targeted) re-looks than the agent was
already doing. No sign of the confabulation failure mode either (a confident
note on a wrong read causing early-commit) — the −0.16pp is well within the
±2.9pp trial noise, in both directions.

This lands with the same shape as the other perception-sub-call enrichments
on this benchmark — `rvlm_subagent` (generalize the sub-call: Δ −0.16) and
`rvlm_subagent_full` (make it a full agent: Δ ~0 at n=8): **enriching the
perception sub-call — with generality, agency, or a rationale channel — does
not move accuracy on DocVQA-2026 val; the minimal `rvlm` sub-call is already
sufficient.** (A richer channel may still pay off where the agent must make
*harder routing decisions* — long multi-page docs — but that's not this set.)

## Status

**DONE (2026-06-07).** n=8 = 39.22 ± 2.91, Δ vs `rvlm` −0.16pp (parity).
Note channel works (~7% flag uncertainty) but is redundant with the existing
verify loop. Rolled into `docs/results.md` (ablations group).
