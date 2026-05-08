# No-loop multi-image baseline (raw VLM, native multi-image input)

**Hypothesis:** the original `no_loop` baseline scores 0% on long docs
(business_report 105–181 pages, science_paper 19–44 pages) because the
composite-image height cap (16384px) crushes per-page resolution to
~80–150px tall, making text illegible. A multi-image variant that sends
the first N pages at native resolution as separate images in one VLM
call should lift those scores — and is a stronger raw-VLM baseline that
preempts the reviewer objection "did you actually try the VLM properly?"

**Setup:** New solver `solver=no_loop_multi`. Sends `min(num_pages,
max_pages)` pages of the document at native resolution as separate
images in a single chat-completion request. No REPL, no tools, no agent
loop. Truncation policy: first N pages (head). Qwen 3.5 27B VLM via
Host B vllm 8928 (4-GPU). Val 80q, `lm.enable_thinking=false`,
`max_concurrency=8`, `question_concurrency=4`, `max_pages=10`.

vLLM probe (3-image multipart request) confirmed multi-image works
natively — model identified each image separately. Smoke test on 2 docs
ran end-to-end with no errors.

## Command

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-remote \
  vlm=qwen-3_5-27b-vllm-remote \
  lm.enable_thinking=false \
  solver=no_loop_multi \
  data.split=val data.num_samples=null \
  max_concurrency=8 \
  run_id=no-loop-multi-3_5-27b-val-t${i}
# Trials launched 2026-05-08 used run_id=no-loop-multi-val-t${i}.
```

## Per-trial scores

| Trial | run_id | Score | Correct | Wall | Sandbox errors |
|---|---|---|---|---|---|
| t1 | `no-loop-multi-val-t1` | _running_ | — | — | 0 (n/a — no REPL) |
| t2 | _pending (chained)_ | — | — | — | — |
| t3 | _pending (chained)_ | — | — | — | — |

## Summary

_To be filled in after t3 completes._

## Comparison (planned)

- **vs no_loop composite baseline:** 17.08% ± 2.60pp (n=3) — see
  `no-loop-baseline.md`. Hypothesis: multi-image lifts long-doc
  categories (business_report, science_paper) where composite was 0%.
  Comics narrative questions probably remain hard (model abstains
  honestly when narrative reading is needed).
- **vs scaffold (Flat Solo m=30):** 44.69% ± 2.81pp (n=8). Multi-image
  closes some of the gap but should still leave a substantial margin
  for the agent loop + VLM tool channel.

## Observations (preliminary)

- **vLLM accepts multi-image natively.** A test with 3 placeholder
  images returned a coherent enumeration. Default `--limit-mm-per-prompt`
  is sufficient (no flag set on Host B vllm); the binding constraint
  is tokens, not image count.
- **Token budget at max_pages=10.** Probe with 10 comics_2 pages
  (1200×1800ish each): 21881 prompt tokens. Comfortable under
  131072 max-model-len.
- **Multi-image is more "honest" than composite on hard questions.**
  Smoke on comics_2 (52pp, sent first 10) returned literal "Unknown"
  on all 4 narrative questions — the model correctly identifies the
  questions are unanswerable from 10 pages. Composite no_loop on the
  same 4 questions returned guesses ("Whodunnit", "red", "5", etc.).
  The "Unknown" rule comes from `prompts.py:10`: "If the question is
  unanswerable given the provided image, the response must be exactly:
  Unknown". When pages are legible, the model recognizes when it can't
  confirm an answer; when pages are crushed by rescaling, the model
  pattern-matches and guesses.

## Configuration knobs

- `max_pages` (default 10): pages to send. Higher → more coverage but
  more vision tokens. At 10, all single-page docs (maps, infographics,
  science_poster) and most engineering_drawing get full coverage;
  science_paper (19–44pp) and business_report (105–181pp) are
  head-truncated.
- `question_concurrency` (default 4): per-doc question parallelism.

## Status

**In progress.** 3-trial val chain launched 2026-05-08 16:11 in
tmux `docvqa-exps:multi`. Heartbeat cron `33c6fffe` will surface
completion. Solver source: `src/docvqa/solvers/no_loop_multi_solver.py`.
Config: `configs/solver/no_loop_multi.yaml`.
