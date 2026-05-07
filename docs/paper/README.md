# DocVQA Scaffold — Paper Project

Working title: TBD (placeholder name in drafts: "the scaffold")

This directory holds planning, intermediate artifacts, and (later) the
paper draft itself. The paper applies the **Recursive Language Models**
paradigm (Zhang, Kraska, Khattab, arXiv:2512.24601, Dec 2025) to
multimodal document VQA, validated by competitive results on the
ICDAR 2026 DocVQA challenge.

## Positioning vs RLM

We are **not** introducing RLM. We are introducing a focused instantiation
of RLM for **document VQA**:

- **Main LLM** operates in a REPL with the document accessible via OCR
  retrieval (BM25 over per-page OCR markdown) — symbolic exploration of
  context too large to fit raw.
- **Sub-call is specialized as a VLM**, exposed to the main agent as a
  visual lookup tool. The VLM perceives page regions / images too detailed
  or numerous to fit the main agent's context window — the same
  context-rot motivation as the original RLM paper, lifted to the
  multimodal setting.
- **Single level of recursion.** Main → VLM, no deeper nesting.
- **Same model class for main + VLM** in our headline runs (Qwen 27B for
  both, Gemini Pro+Flash for the closed-frontier point) — driven by the
  ICDAR tier rules, not by the method. The framework is general; main and
  VLM can be different models.

Novelty: not the method (RLM exists), but the application — taking
RLM into the multimodal document setting where the "context too long"
problem is acute (many pages, dense images, tables, infographics) and
showing that this lifts every model class.

## Headline claim (working version)

Applying RLM to document VQA — a code-capable LLM in a REPL with OCR
retrieval and a VLM sub-call — delivers double-digit accuracy improvements
across model classes (≤8B / 8–35B / >35B), requiring no specialized
document encoder, no proprietary OCR, and no domain-specific training.

Current evidence (val/test, ICDAR 2026 DocVQA, **mean ± std across
independent trials** — no self-consistency voting; see `decisions.md` D-003):

- **Qwen 3.6 27B (open):** **44.1% val ± 3.0pp** (8 trials, range
  40.0–47.5%). Test mean across the 8 trials needs to be computed
  (currently only SC-8 test 43.75% is summarized in
  `docs/results.md:144-187`).
- **Gemini 3 Pro (closed):** baseline 37.5% test (official); with scaffold
  59.4% test (single trial — needs ≥3-trial replication).
  (`docs/results.md:15`, `docs/results.md:23`)
- **Gemini 3 Flash:** baseline 33.75% test; with scaffold ~50% (mentioned;
  needs trials and to land in `docs/results.md`).

## Strawman abstract (v1 — to be refined)

> Document visual question answering remains hard for general-purpose
> multimodal models even at frontier scale, with leading proprietary models
> below 40% on the ICDAR 2026 DocVQA challenge test set. The challenge is
> a context-rot problem in disguise: documents have many pages, dense
> images, and fine detail that overwhelm a multimodal model's context
> window when fed raw. We apply the Recursive Language Model paradigm
> (Zhang et al., 2025) to this setting: a code-capable LLM operates in a
> REPL where the document is symbolically accessible via OCR retrieval,
> and a VLM is exposed as a recursive sub-call for visual perception of
> regions the main agent chooses to inspect. The resulting method is
> model-agnostic and lifts performance substantially across the model
> size spectrum — it lifts open ≤8B models (Qwen 3.5 9B, Gemma) into
> competitive range, turns an open 27B model into a competitive entry in
> the ICDAR 2026 8–35B tier, and lifts frontier closed models (Gemini 3
> Pro+Flash) by 20+ percentage points on the test set. We use no
> specialized document encoder, no proprietary OCR, and no domain-specific
> training. All scores are reported as mean ± std across independent
> trials. We validate generality across N additional benchmarks and
> ablate the method's components, isolating the contribution of OCR
> retrieval, the VLM sub-call, and turn budget.

Open in this draft:
- Method name (placeholder: "RLM-DocVQA" or similar)
- Confirmed multi-trial numbers with error bars (replace single-trial 59.4%)
- Number of benchmarks (decide after lit review)
- "≤8B" claim depends on small-model experiments landing

## Target venue & deadline

TODO — user to decide. Top candidates:
- **EMNLP / NAACL / ACL Findings** — empirical study framing
- **ICDAR 2026 challenge proceedings** — system description; lower bar but lower visibility
- **TMLR** — rigor-leaning, no novelty bar

## Methodology — frame-first, not experiment-first

1. Strawman abstract + figure list locked before scaling experiments
2. Lit review in parallel — kicked off via `lit-review-brief.md`
3. Risk-ranked experiments — falsification first, polish later
4. Multi-trial discipline (≥3 trials per headline number, error bars always)
5. Write while running — draft sections as data lands

## File index

- `README.md` — this file (overview, status, principles)
- `decisions.md` — append-only decision log (framing, scope, method
  presentation, experimental design). Read before re-opening a settled
  question.
- `lit-review-brief.md` — internal planning doc for the lit-review task
  (references repo files; for our own reasoning)
- **Two parallel self-sufficient lit-review prompts** (each runs an
  independent agent — they don't depend on each other):
  - `lit-review-prompt.md` — **RLM-focused** prompt. Asks the agent to
    map the RLM paradigm, code-as-reasoning vision agents, and
    adjacent paradigms; identify any prior RLM-on-multimodal-docs work
    that would weaken our novelty. Output: `lit-review-rlm.md`.
  - `lit-review-docvqa-prompt.md` — **DocVQA-focused** prompt. Asks the
    agent to map document VQA methods, per-benchmark SOTA, ICDAR
    challenge history, and recommend baselines. Output:
    `lit-review-docvqa.md`.
- `experiment-plan.md` — main experiments + ablations + execution order
- `related-works.md` — running index of relevant papers, with connection
  notes and obsidian paths (no paper files copied into repo)
- (later: `outline.md`, `figures/`, `draft.md`)

## Status tracker

- [ ] Strawman abstract reviewed (RLM framing applied 2026-05-01)
- [ ] Method name picked
- [ ] Target venue / deadline picked
- [x] RLM-focused lit review delivered → `lit-review-1.md`
- [x] DocVQA-focused lit review delivered → `lit-review-2.md`
- [x] Related-works index updated from both lit reviews (2026-05-01)
- [x] RVLM positioning decided: **concurrent work** (D-005)
- [x] MADQA framing decided: verify + use as baseline (D-005)
- [ ] Verify RVLM (arXiv:2603.24224) on arXiv; download to obsidian
- [ ] Verify MADQA (arXiv:2603.12180) on arXiv; download to obsidian; user reads
- [ ] Verify other lit-review citations before they land in the paper
- [ ] Read ARIAL (arXiv:2511.18192) — closest agentic-DocVQA competitor
- [ ] Pick a method name that clearly differs from "Recursive Vision-Language Model" (D-005)
- [ ] Benchmark + baseline picks finalized after RVLM/MADQA reads
- [ ] Experiment plan signed off
- [ ] Matched-baseline runs (Qwen 27B without scaffold) on ICDAR val/test
- [ ] Multi-trial Gemini 3 Pro test replication (≥3 trials)
- [ ] Small-model runs (≤8B: Qwen 3.5 9B, Gemma)
- [ ] Generality experiments on second benchmark
- [ ] Ablations (no-loop baseline, OCR on/off, VLM sub-call on/off, VLM cropping on/off, turn budget, category tips)
- [ ] Compute multi-trial test mean ± std for Qwen 3.6 27B (replace SC-8 number)
- [ ] Error analysis + qualitative trace examples
- [ ] First draft

## Working principles

- **Variance discipline:** ≥3 trials per headline number, std reported, no
  single-trial headlines. Reuse multi-trial setup from `docs/results.md:144`.
- **Verify before claiming:** numbers in this directory cite run IDs from
  `docs/results.md`. Don't drift from source.
- **Update this README's status tracker** as work lands. This file is the
  project ground truth.
- **Plan changes go here**, not in conversation. If we change scope or
  drop an experiment, update `experiment-plan.md` with a dated note.
