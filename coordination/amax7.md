# amax7 queue (adaptive host)

amax7 is the adaptive host — runs critical-path experiments where the
result might change the experiment plan. Tighter feedback loops; one
cell at a time; replan after each result.

**vllm:** Qwen 3.5 27B at `localhost:8927`.

## In progress

### `[→]` Refill pass: skeletal (4 docs) + naked (1 doc) — 2026-05-29T18:??

Round out full-set numbers for the strip-chain so n=1 scores are on the
full 25-doc / 80-Q denominator. Hybrid was already 25/25 complete.
Sequential to avoid vllm contention.

Missing docs:
- skeletal-t1: `business_report_4 infographics_2 science_paper_1 science_poster_1`
- naked-t1: `business_report_4`

## Queued

### 1. `[ ]` rvlm_ocr n=1 val (task #14)

Locks the clean OCR-extension number. Current `rvlm_full` legacy data is
confounded with `look()` ergonomic wrapper. This is the clean cell.

```bash
uv run python evals.py \
  lm=qwen-3_5-27b-vllm-local vlm=qwen-3_5-27b-vllm-local \
  lm.enable_thinking=false \
  solver=rvlm_ocr \
  data.split=val data.num_samples=null \
  max_concurrency=16 \
  run_id=rvlm-ocr-val-t1
```

- Expected wall: ~50min
- Compare to: `rvlm` val (OCR-free); expect approximately equal on
  DocVQA-2026 (moderate-length docs).

### 2. `[→moved]` direct_vlm n=1 val (task #19) — MOVED to amax1 (2026-05-29)

Moved to `coordination/amax1.md` to run paired with `direct_vlm_minimal`
on the throughput host, at **`max_iterations=40`** (the minimal cell
surfaced that cap=20 is binding — ~56/80 questions hit 20/20). Running
both direct-VLM-architecture cells at the same cap keeps the
prompt-stripping (TOOL_HINTS) comparison clean.

**Caveat for the paper claim (decision rules below):** `direct_vlm`
will now be cap=40, but the `rvlm` headline (n=8) was cap=20. The
`direct_vlm < rvlm by 5+pp` / `≈ rvlm` rules below assume equal
iteration budget — interpret the direct_vlm-vs-rvlm comparison only
after confirming the cap difference doesn't drive it (an rvlm cap=40
spot-check may be needed).

## Done

### `[✓]` rvlm_skeletal + rvlm_naked + rvlm_hybrid n=1 val (tasks #32 #33 #35) — 2026-05-29

Strip-chain results, all n=1 on Qwen 3.5 27B local, c=32 (refill pending
for skeletal/naked long-tail). Compare to rvlm_minimal n=8 mean
40.94% (wait — that's unified; minimal is **42.03% ± 2.21pp**).

**Full-set numbers (no refill yet):**
- skeletal-t1: 45.45% (25/55) — 4 docs missed long-tail
- naked-t1: 32.00% (24/75) — 1 doc missed
- hybrid-t1: **35.00% (28/80)** — clean 25/25, no long-tail!

**Common 21-doc / 55-Q subset (clean 4-way compare):**

| Solver | common-21 | Δ vs minimal-t1 |
|---|---|---|
| minimal-t1 | 43.64% (24/55) | — |
| skeletal-t1 | 45.45% (25/55) | **+1.82pp** |
| naked-t1 | 34.55% (19/55) | **−9.09pp** |
| hybrid-t1 | 36.36% (20/55) | **−7.27pp** |

**Preliminary reads (n=1, refill pending):**
- **skeletal ≈ minimal** (+1.82pp on common 21): the 3 doc-shape
  patterns (high-density, many-page, counting) do not carry the method.
  Strips fine; promote-to-default candidate.
- **naked drops ~9pp**: removing the APPROACH steps + the verify-under-
  VLM-stochasticity principle costs real points. **APPROACH + verify
  are load-bearing.** The skeletal → naked delta isolates the
  contribution of these (skeletal 45.45% → naked 34.55% = −10.91pp on
  the same 55Q). Naked is a step too far.
- **hybrid drops ~7pp**: having `display()` as an alternative to
  `ask_vlm()` doesn't help on n=1 — might even hurt. Possible
  explanations: agent confused by choice; display() adds per-iteration
  context-bloat (images in conversation); or just an n=1 low draw.
  Worth n=8 for a clean read. Also hybrid was ~2× slower per doc than
  the rvlm-family (MultimodalRLM has heavier per-iteration cost from
  inline images).

**Operational note: GPU overlap.** The new `rvlm_overlap_orch.py`
pattern launched hybrid the moment naked hit 22/25, so no GPU idle
across the long-tail transition. Pattern is reusable for any future
strip chain.

### `[✓]` rvlm_minimal n=8 val — generality test (task #31) — 2026-05-29

run_ids: `rvlm-minimal-val-t{1..8}` · **n=8 mean 42.03% ± 2.21pp**
(range 38.75–45.00) on Qwen 3.5 27B local, c=32.

Paired vs `rvlm_unified` t1..t8: **Δ = +1.09pp [CI95: −3.14, +5.33]**,
paired t = 0.611, df = 7 (n.s.). Lands cleanly in the pre-set
"≈ 0pp / within paired noise" band → **`rvlm_minimal` is the proposed
method.** The 10.7 kB of hand-crafted DocVQA-2026 category tips and
the engineering-drawing-specific VLM sub-call signature are not
load-bearing; the recursive-perception mechanism is.

Variance note worth surfacing for the paper: minimal σ = 2.21pp,
unified σ = 4.05pp — almost 2× tighter trial-to-trial. Plausibly the
agent stops being yanked between competing per-category prescriptions
when the dispatch guess is wrong. Per-trial table + paired analysis:
`docs/experiments/rvlm-minimal-generality.md`.

Refill notes: 4/8 trials needed pass-2 refill — `science_paper_1`
systematically hits the 4h task_timeout (~50% per-attempt success
rate, agent-loop / vllm-hang specific to this doc). All cleared on
the sequential refill pass.

### `[✓]` unified-tips n=8 val (tasks #25 + #28) — 2026-05-28

run_ids: `rvlm-unified-val-t{1..8}` · **n=8 mean 40.94% ± 4.05pp**
(range 35.0–47.5) on Qwen 3.5 27B local, c=32.

Paired vs amax1's `rvlm` baseline at t1..t7: Δ mean = +0.71pp,
SE 1.72pp, 95% CI [−3.50, +4.93]pp — well inside noise. Per-trial
table + paired analysis: `docs/experiments/unified-category-tips-ablation.md`.

Variance asymmetry worth flagging for the promote-to-default
decision: unified σ=4.05pp vs rvlm σ=2.38pp (~1.7×). Final paired
analysis lands when amax1's t8 commits.

## Decision rules (set in advance)

- **unified-tips Δ ≈ 0pp** → promote unified to default; replace rvlm
  cells with rvlm_unified in subsequent cells.
- **rvlm_ocr ≈ rvlm on val** → OCR neutral on moderate-length docs;
  paper's §B doc-length-axis claim holds. Push to MMLongBench-Doc next.
- **direct_vlm < rvlm by 5+pp** → recursive sub-call is load-bearing;
  paper §C prediction 3 supported.
- **direct_vlm ≈ rvlm** → architecture-agnostic; reframing needed
  (the sub-call may not be the load-bearing piece — context-rationing
  is, regardless of architecture).
