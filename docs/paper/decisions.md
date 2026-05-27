# Decision Log — DocVQA Scaffold Paper

Append-only record of consequential decisions about scope, framing, method
presentation, and experimental design. Each entry: decision + date +
reasoning + implications. Reasoning is recorded so we don't relitigate
or quietly re-introduce things we already decided to drop.

Entries are roughly chronological. Status: `accepted` unless we explicitly
revise an earlier decision.

---

## D-001: Frame the paper as an application of RLM

- **Date:** 2026-05-01
- **Status:** accepted

**Decision.** Position the paper as a focused application of the
Recursive Language Models paradigm (Zhang, Kraska, Khattab,
arXiv:2512.24601, Dec 2025) to multimodal document VQA, with the
recursive sub-call specialized as a VLM. Not as "a custom agentic
scaffold."

**Reasoning.** RLM provides the architectural template our method
instantiates. Citing it directly:

- Acknowledges where the architectural idea comes from.
- Sharpens what is and isn't novel — the contribution is **application +
  empirical results**, not architecture.
- Connects the work to a coherent research line, giving reviewers a
  clean target for "what's the delta."
- Mirrors the RLM ablation structure (no-loop, no-symbolic-access,
  no-sub-call), which makes our ablations more legible.

**Implications.**

- Method components described in RLM language: REPL, symbolic context
  access, recursive sub-call (specialized as VLM in our case).
- Novelty claim is explicit and bounded — multimodal/document
  application, with VLM specialization of the sub-call.
- Lit review must aggressively check for any prior RLM-on-multimodal
  work that would weaken this claim.

---

## D-002: Skip OCR quality sensitivity ablation

- **Date:** 2026-05-01
- **Status:** accepted

**Decision.** Ablate OCR on/off (binary) only. Do not vary OCR quality
or compare OCR engines.

**Reasoning.** Tangential to the main claims. Binary on/off already
isolates whether the OCR-retrieval channel matters; quality sensitivity
adds analysis without strengthening the headline.

**Implications.**

- Removed from `experiment-plan.md` ablation table.
- If a reviewer specifically asks for OCR robustness, treat as
  rebuttal-time work.

---

## D-003: Drop self-consistency from the paper's method framing

- **Date:** 2026-05-01
- **Status:** accepted

**Decision.** Do not present self-consistency (SC-8 majority voting) as
part of our method in the paper. Report **mean ± std across independent
trials** as the headline reporting. Do not cite Wang et al. 2023 SC paper.

**Reasoning.**

- SC was a competition tactic to pick a single submission. It is not
  part of the method's substance — it's post-hoc voting that boosts a
  final number.
- Mean ± std across independent trials is the honest scientific
  reporting and gives readers the actual signal (per-trial performance
  with its variance).
- Including SC as a contribution would require defending it as novel,
  which it isn't (Wang et al. 2023). Better to take it out of scope.

**Implications.**

- Headline numbers in abstract / tables / figures: 8-trial mean ± std,
  not SC-8 voted scores.
- Need to compute mean ± std across the 8 individual test trials for
  Qwen 3.6 27B (currently `docs/results.md` summarizes only the SC-8
  test number; per-trial test scores must be aggregated). Update
  `docs/results.md` once computed.
- **Drop SC budget ablation** (the k=1,2,4,8 curve) from the experiment
  plan.
- **Drop SC budget figure** from the figure list.
- **Drop SC as a search topic** in the lit review.
- **Drop Wang et al. 2023 SC entry** from `related-works.md`.
- Competition submission strategy (using SC-8) can be a brief footnote,
  not a contribution.

---

## D-004: Add arbitrary-cropping ablation

- **Date:** 2026-05-01
- **Status:** accepted

**Decision.** Add an ablation where the VLM `look` / `batch_look` tool is
restricted to **whole pages** — accepting only a page index, not an
arbitrary PIL Image. This tests whether the agent's ability to crop /
zoom into arbitrary image regions contributes to performance, separately
from the VLM-on/off ablation (which removes the VLM entirely).

**Reasoning.** Active visual perception — the agent choosing to zoom
into specific regions — is plausibly important for:

- **High-resolution pages.** Some test images exceed 246M pixels.
  Whole-page processing risks exceeding VLM context or losing fine
  detail; cropping lets the agent attend to a region at native
  resolution.
- **Small-detail extraction.** Chart cells, fine labels, sub-diagrams,
  table cells embedded in dense pages.
- **Multi-step visual reasoning.** Look at a region, follow a reference,
  look at a related region.

If cropping does not matter, that's an interesting finding too — would
let us simplify the tool surface in future work and would give a cleaner
story about what the VLM sub-call is actually doing.

**Implications.**

- Add row to ablation table in `experiment-plan.md`: "VLM cropping
  on/off" (full method vs page-only).
- Add to lit-review prompt and brief ablations list.
- **Implementation needed.** Current `look` tool accepts any PIL Image
  (see `docs/solvers/flat-solo.md`, `docs/solvers/leanest-solo.md`).
  Need a page-index-only variant. Effort is low — restrict the tool
  signature or prompt-enforce.

---

## D-005: Position vs RVLM and MADQA

- **Date:** 2026-05-01
- **Status:** accepted (RVLM positioning final pending arXiv verification;
  MADQA framing final pending verification + read)

**Decision.**

- **RVLM (arXiv:2603.24224, Mar 2026):** treat as **concurrent work**,
  not prior art. Position alongside, not against. Acknowledge in related
  work; spell out the domain delta (medical imaging vs multi-page
  documents); do not claim novelty over RVLM on the architectural idea.
- **MADQA / Borchmann et al. (arXiv:2603.12180, 2026):** verify and read.
  Working plan: **use their agentic method as a baseline** in our
  experiments. Engage with their "unconstrained RLMs are an efficiency
  catastrophe" critique by framing our scaffold explicitly as a
  *focused / constrained* instantiation.

**Reasoning.**

- Concurrent-work framing for RVLM is defensible at most venues: a two-
  month gap is within the typical concurrent window, and the domain delta
  (medical scans → multi-page documents) is substantive. Our scaffold
  also introduces document-specific components (BM25 OCR retrieval,
  multi-page navigation, category-tip prompting) that aren't in RVLM as
  far as we currently know.
- Abandoning the project on a name-and-lineage overlap is premature. The
  delta in evaluation domain and scaffold design carries real weight.
- MADQA-as-baseline turns a positioning threat into a positive empirical
  comparison. If our scaffold beats or matches their constrained-agent
  numbers on MADQA, we get a second venue for the lift claim.

**Implications.**

- `related-works.md`: move RVLM out of "novelty threat" framing into
  "concurrent work" subsection. MADQA stays in a positioning-target
  subsection until verified, with the baseline plan noted.
- `experiment-plan.md`: add MADQA to the secondary-benchmark candidate
  list and add the MADQA constrained-agent method to the baseline
  shortlist.
- After RVLM is verified and read, write the explicit
  delta paragraph for related work (medical vs multi-page docs;
  OCR-retrieval channel; multi-page navigation; document-type diversity
  across 8 categories).
- **Name choice:** even under concurrent framing, pick a method name
  that clearly differs from "Recursive Vision-Language Model." Reduces
  reviewer confusion regardless of how RVLM is positioned.

---

## D-006: Reframe paper around visual-context-budget hypothesis

- **Date:** 2026-05-27
- **Status:** accepted

**Decision.** Shift the paper's headline from "we built an agentic
scaffold that wins ICDAR 2026 DocVQA" to a research-mode hypothesis:
**mid-sized open VLMs are bottlenecked by visual context budget on
document VQA; recursive perception (RLM with a VLM sub-call) lifts
them to frontier-level accuracy without scaling model size or training
data.**

Three falsifiable predictions support the hypothesis:

1. **Model-size axis.** Lift scales with model code-writing capability
   (a code-driven recursive sub-call needs a model that can drive it).
   Supported by E4B +5.83pp, 9B +6.25pp, 27B +20.94pp, 31B +25.00pp.
2. **Document-length axis.** Lift scales with effective document length
   (the context-budget hypothesis itself). Supported by MMLongBench-Doc
   +16.84pp judge, MP-DocVQA 11-20pp bucket +13.68pp, DocVQA-2026
   +20.94pp.
3. **Mechanism axis.** Removing the recursive VLM sub-call (but keeping
   the REPL + agent loop) collapses the lift. **Not yet measured — the
   critical missing experiment.**

The proposed method is the **OCR-free** variant (current engineering
name: `leanest_solo`). The **OCR-extension** variant is reported as an
extension, not a contribution, and requires a *new* solver — distinct
from `flat_solo`, which confounds OCR with a single-image `look()`
ergonomic wrapper.

**Reasoning.**

- Empirical headline: OCR-free wins on DocVQA-2026 test (39.0%) over
  the OCR variant after prompt-scrubbing (38.0%). OCR's value
  concentrates on truly long documents (MMLongBench-Doc +2pp on top of
  leanest, MP-DocVQA 11-20pp bucket +13.68pp). On moderate-length docs
  it is neutral or a small drag.
- This isolates the recursive VLM sub-call as the load-bearing
  mechanism, mirroring the original RLM paper (Zhang et al., 2025).
  Cleaner research story than the competition framing: a single
  architectural addition explains the lift; everything else is
  benchmark-conditional.
- Competition-tactic items (SC voting, FLAT_SOLO_TOOL_HINTS, multi-image
  extension, GEPA, pyai port) are out-of-scope for the paper — they
  optimized for a single number on a single split, not research
  contributions.
- The prompt-iteration history (v1/v2/scrub) is not part of the paper.
  The reader sees the end-state, not the process.
- Engineering solver names (`leanest_solo`, `flat_solo`, `rvlm`,
  `no_loop_multi`) are not used in the paper; paper-facing names
  picked later. They reflect the development history, not the
  research finding.

**Implications.**

- Build a new solver = `leanest_solo` tool surface (`batch_look` only)
  + `search()` + `page_texts` in scope. Replaces `flat_solo` as the
  OCR-extension cell. Working name TBD; placeholder `m_ocr`.
- Build the VLM-sub-call-off solver = fork of `leanest_solo` with
  `batch_look` removed. Tests prediction 3.
- Re-run model-axis cells (Gemma 4 E4B, Qwen 3.5 9B, Gemma 4 31B) with
  current clean prompts — their original runs (2026-05-09/10) used
  pre-scrub prompts.
- Re-run MP-DocVQA + MMLongBench-Doc OCR-extension cells with the new
  M+OCR solver, not the confounded `flat_solo`.
- Existing `flat_solo` data goes to `docs/experiments/` only, or
  becomes an appendix "kitchen-sink" cell after we learn whether
  `look()` adds anything on top of M+OCR.
- `experiment-plan.md` and `paper/README.md` restructured around the
  new headline. Strawman abstract rewritten.
- The map from existing experimental data to the new hypothesis is
  documented in `paper/README.md`'s "evidence map" section — most
  data carries over; the framing changes, not the data.

---

## D-007: Per-solver inline prompts (DRY-undo) + prompt-parity rule

- **Date:** 2026-05-27
- **Status:** accepted

**Decision.** Move category-tip prompts out of `src/docvqa/prompts.py`
into each solver's own file. Each solver owns its prompts inline — no
shared `CATEGORY_TIPS` dict across solvers. Only competition-invariant
content (`ANSWER_FORMATTING_RULES`) stays in `prompts.py`.

Coupled with this: all solvers reported in the paper must pass the
**same prompt-parity audit standard**. Same val-leak scrub, same audit
effort, no solver heavily reviewed relative to others. The shared
inheritance from a single source-of-truth (via `get_category_tips`) is
replaced by an inline-then-audit discipline.

**Reasoning.**

- Shared prompt dicts caused the v1/v2 mess (see
  `docs/experiments/scrub-audit.md`). Stripping tool-routing verbs for
  leanest (where they were dead references) accidentally stripped them
  for `flat_solo` too (where they steered real tools), costing 2pp on
  test. A v2 overlay was then needed to restore them. All of this came
  from cross-solver coupling.
- We are a research project, not a production codebase. The DRY benefit
  (one place to edit) is outweighed by the coupling cost (one solver's
  prompt needs leak into another).
- The 2026-05-27 parity audit found that even after the v1 scrub,
  `RVLM_CATEGORY_TIPS` is asymmetric vs `CATEGORY_TIPS` along
  non-tool-surface lines (missing TEXT TRUNCATION on business_report,
  missing COUNTING-OBJECTS on maps, missing CITED PAPER FINDINGS on
  science_paper; extra Leader-lines bullet on engineering_drawing).
  Inline ownership makes these asymmetries visible and auditable,
  rather than masked by inheritance hierarchy.

**Implications.**

- Refactor: move `CATEGORY_TIPS`, `BASELINE_CATEGORY_TIPS`,
  `FLAT_SOLO_TOOL_HINTS` (or relevant fragments) into the solver files
  that use them. Each solver gets its own inline dict.
- `ANSWER_FORMATTING_RULES` stays in `prompts.py` as the only shared
  invariant.
- After the refactor, reconcile asymmetries between solvers' tips
  along semantic / question-interpretation lines. Tool-surface
  differences are legitimate; semantic differences must be deliberate
  decisions, documented per-solver. Per-bullet decisions for the
  current asymmetries are tracked in task list (#18).
- The DA profiles in `datasets/profile.py` already follow this
  principle (each benchmark profile owns its content). The
  2026-05-27 audit confirmed parity-clean — no DocVQA-2026 leakage
  into MP-DocVQA or MMLongBench-Doc. No changes there.
- Document the inline location in each solver's module docstring so
  future readers find prompts where they're used, not where they're
  centrally defined.
- Add a small guard in `get_profile()` so unknown dataset ids do not
  silently fall back to `DOCVQA_2026_PROFILE`. (Minor finding from
  the profile audit.)

---

## D-008: Trial-budget escalation policy (n=1 → n=2 → n=8)

- **Date:** 2026-05-27
- **Status:** accepted

**Decision.** New experimental cells use a budget-escalation policy:
n=1 first across all cells in an experiment, n=2 if the n=1 direction
holds, n=8 only after the paper headline framing is locked. Avoids
wasted compute when a pivot or revision is still possible.

**Reasoning.**

- This conversation produced the third framing pivot in ~3 weeks
  (competition mode → scrub-audit → research-mode hypothesis).
  Committing to n=8 per cell each pivot is expensive.
- One trial gives a directional signal (sign of the lift, gross
  magnitude). Two trials catch the largest variance surprises. Eight
  trials are paper-table grade.
- The existing variance discipline (≥3 trials per headline number,
  mean ± std) is preserved — the policy only changes WHEN we escalate,
  not WHETHER.

**Implications.**

- Headline cells already at n=8 with clean prompts (Qwen 27B
  leanest/flat scrubbed, no_loop_multi split-calibration,
  MMLongBench-Doc DA, MP-DocVQA DA) stay at n=8.
- Model-axis re-runs (Gemma E4B, Qwen 9B, Gemma 31B) start at n=1.
  Direction check vs the pre-scrub n=3 data. Escalate to n=2 if it
  matches; n=8 only after the paper headline is locked.
- New M+OCR solver cells: n=1 val first; then n=2; then n=8.
- VLM-sub-call-off ablation: n=1 val (prediction: collapse to
  ~17-25%); n=2 if collapse confirms; n=8 for the paper table.
- RVLM chain (restart after prompt reconciliation): n=1 val first
  rather than full n=8; escalate based on direction.
- Variance discipline rule in `experiment-plan.md` updated to reflect
  escalation, not blanket n=8.

---

## How to add entries

1. Allocate next D-NNN id.
2. Capture: decision (one sentence), date, status, reasoning,
   implications (what changes in other docs).
3. Walk through `README.md`, `experiment-plan.md`, `lit-review-prompt.md`,
   `lit-review-brief.md`, `related-works.md` and apply the
   implications. Don't leave the decision uncodified.
4. If reversing a prior decision, add a new entry referencing the old
   one and mark the old one `revised by D-NNN`. Do not edit the old
   entry's body.
