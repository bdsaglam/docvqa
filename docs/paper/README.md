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

## Headline claim (per D-006, 2026-05-27)

**Mid-sized open VLMs are bottlenecked by visual context budget on
document VQA, not by reasoning capacity.** The right architectural fix
is recursive perception — letting the main agent ration its
high-resolution VLM calls — rather than scaling model size or training
data. This is a focused application of the Recursive Language Models
paradigm (Zhang et al., 2025), with the recursive sub-call specialized
as a VLM. The proposed method is **OCR-free**: a code-capable LLM in
a REPL whose only tool is a recursive `batch_look` call to a VLM. OCR
retrieval is reported as an **extension**, not a contribution; it
matters on long-doc benchmarks and not on moderate-doc ones.

Three falsifiable predictions (with the supporting data):

1. **Model-size axis.** Lift scales with model code-writing capability.
   Supported: E4B +5.83pp, 9B +6.25pp, 27B +20.94pp, 31B +25.00pp.
2. **Document-length axis.** Lift scales with effective doc length.
   Supported: MMLongBench-Doc +16.84pp judge, MP-DocVQA 11-20pp bucket
   +13.68pp, DocVQA-2026 +20.94pp.
3. **Mechanism axis.** Removing the recursive VLM sub-call (but keeping
   REPL + agent loop) collapses the lift. **Not yet measured — the
   critical missing experiment** (task list #9).

Current headline numbers (Qwen 3.5 27B, n=8 SC-8, val-leak-scrubbed
prompts):

- **OCR-free method (proposed):** val 48.8% / **test 39.0%** ICDAR
- **OCR-extension (legacy `flat_solo`, confounded):** val 47.5% / test 38.0%
- **Raw-VLM baseline:** val 20.0% / test 11.0%

Split-calibration check: the val→test gap is dominated by
split-difficulty (~9pp floor at the no-scaffold baseline), not by
prompt overfitting. Post-scrub headline cells sit at the floor — no
measurable generalization gap remains.

## Solver taxonomy (per D-006)

The paper measures four cells; engineering names are placeholders, paper
names picked later.

| Engineering name | Paper role | Tool surface |
|---|---|---|
| `leanest_solo` | **Proposed method (M)** | `batch_look` only |
| `<m_ocr>` (TBD, new fork) | **+OCR extension** | `batch_look` + `search` + `page_texts` |
| `rvlm` | **Alternative angle** | REPL with `display()` — single multimodal model, no sub-call |
| `no_loop_multi` | **Raw-VLM baseline** | one forward pass, no scaffold |
| `official_baseline` | **Competition baseline** | kit MASTER_PROMPT |

The legacy `flat_solo` is *not* the clean OCR extension (it bundles
`look()` ergonomic wrapper with the OCR channel). A new fork of
`leanest_solo` is built that adds OCR/search only. Existing `flat_solo`
data lives in `docs/experiments/` for reproducibility but does not
anchor a paper number.

## Strawman abstract (per D-006)

> Document visual question answering remains hard for general-purpose
> multimodal models even at frontier scale — leading proprietary models
> sit below 40% on the ICDAR 2026 DocVQA challenge test set. We show
> that this is a *visual context-budget* problem: mid-sized open VLMs
> have ample reasoning capacity but cannot allocate their finite visual
> context across many-page documents with dense fine-grained content.
> We apply the Recursive Language Models paradigm (Zhang et al., 2025)
> with the recursive sub-call specialized as a VLM — a code-capable
> main agent in a REPL invokes batched VLM perception of pages or
> arbitrary crops, rationing its visual budget per question. The method
> uses no symbolic retrieval channel in its core form; on
> moderate-length-document benchmarks this is sufficient, and on
> long-document benchmarks an OCR retrieval extension provides an
> additional lift. We validate three predictions: lift scales with
> model code-writing capability across four open models (4B–31B);
> lift scales with effective document length across three benchmarks
> (DocVQA-2026, MP-DocVQA, MMLongBench-Doc); and removing the
> recursive sub-call (but keeping the REPL and agent loop) collapses
> the lift to the raw-VLM baseline. All scores are reported as mean ±
> std across independent trials. The result is a single architectural
> addition that lifts a 27B open model into competitive range with
> closed frontier models on document VQA, requiring no specialized
> document encoder, no proprietary OCR, and no domain-specific
> training.

Open in this draft:
- Paper name for the method (placeholder: M)
- Final ablation table layout (waiting on VLM-sub-call-off result)
- Whether OCR-extension cells get re-run with the clean M+OCR fork
  before the draft locks

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

## Status tracker (updated 2026-05-27 for D-006 pivot)

### Framing & planning
- [x] Headline reframed around visual-context-budget hypothesis (D-006)
- [x] Per-solver inline prompts principle accepted (D-007)
- [x] Trial-budget escalation policy accepted (D-008)
- [x] RLM-focused lit review delivered → `lit-review-1.md`
- [x] DocVQA-focused lit review delivered → `lit-review-2.md`
- [x] Prompt-parity audit done (2026-05-27 — `aa857907bec1c90ba`, `a3c24b9c9f1e6688b`)
- [ ] Method/paper name picked
- [ ] Target venue / deadline picked
- [ ] Verify RVLM (arXiv:2603.24224) + MADQA (arXiv:2603.12180) + ARIAL (arXiv:2511.18192) — deferred until experiments lock

### Code refactor (D-007, in flight)
- [ ] Reconcile RVLM_CATEGORY_TIPS asymmetries with canonical (task #18)
- [ ] Inline category tips per-solver — drop shared dict (task #4)
- [ ] Guard `get_profile()` against unknown-id fallback
- [ ] Rename engineering solver concepts to paper-facing names (later)

### Experiments — locked headline cells (n=8 with clean prompts)
- [x] Raw-VLM baseline (`no_loop_multi`): val 20.0% / test 11.0% — split-calibration anchor
- [x] OCR-free method (`leanest_solo` scrubbed): val 48.8% / test 39.0%
- [x] Cross-benchmark generality on MP-DocVQA + MMLongBench-Doc (Qwen 27B, n=3, DA profile)
- [x] Model-size lift signs across E4B/9B/27B/31B (with pre-scrub prompts; see caveat)

### Experiments — pending under new framing
- [ ] Build new M+OCR solver (clean fork of leanest with OCR/search, no `look()`) — task #13
- [ ] Run M+OCR on DocVQA-2026 val + test — task #14
- [ ] Re-do MMLongBench-Doc + MP-DocVQA OCR-extension cells with M+OCR — task #15
- [ ] Build VLM-sub-call-off ablation (fork of leanest minus batch_look) — task #7
- [ ] Run VLM-sub-call-off ablation — **critical for prediction 3** — task #9
- [ ] Re-run model-axis cells (Gemma E4B, Qwen 9B, Gemma 31B) on clean prompts — task #8
- [ ] Halt + reconcile + restart RVLM chain — task #17
- [ ] Decide flat_solo's paper role (drop vs appendix) — task #16

### Writing
- [ ] Draft new abstract — task #11
- [ ] Restructure experiment-plan.md around D-006 — task #5
- [ ] First draft
- [ ] Error analysis + qualitative trace examples

## Working principles

- **Hypothesis-first.** Frame, ablations, and figures derive from the
  three predictions in D-006. Competition optimizations (SC voting,
  FLAT_SOLO_TOOL_HINTS, multi-image extension) are out of scope.
- **Prompt parity (D-007).** All solvers in the paper pass the same
  audit standard. Per-solver inline prompts — no shared dict across
  solvers. Only `ANSWER_FORMATTING_RULES` is shared.
- **Trial-budget escalation (D-008).** n=1 first, n=2 if direction
  holds, n=8 only after the paper headline locks. Variance discipline
  (≥3 trials for headline numbers, mean ± std) preserved.
- **Verify before claiming:** numbers cite run IDs from
  `docs/results.md` and `docs/experiments/*.md`. Don't drift.
- **No prompt-iteration narrative in the paper.** Readers see the
  end-state, not the v1/v2/scrub history.
- **No engineering solver names in the paper.** `leanest_solo`,
  `flat_solo`, `rvlm`, etc. are development names. Paper-facing names
  picked separately.
- **Update this README's status tracker** as work lands. This file is
  the project ground truth.
- **Plan changes go here**, not in conversation. Update
  `experiment-plan.md` with a dated note when scope shifts.
