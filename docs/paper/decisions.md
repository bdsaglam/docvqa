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
3. **Active-perception mechanism.** The lift comes specifically from
   *active, iterative* VLM sub-calls — the agent choosing what region
   to inspect, at what resolution, across multiple turns — not from
   giving more compute to a single VLM call. Three ablations triangulate
   this: cropping-off −7.81pp (active region selection matters);
   m=5 turn budget −15pp vs m=30 (iteration matters); leanest 48.8%
   vs no_loop_multi 20.0% (the recursive sub-call carries +28.8pp over
   one-shot). All measured; no new experiment needed.

   *Earlier draft of prediction 3* called for a "REPL-only" cell
   (REPL + agent loop with the VLM sub-call removed). Built and smoke-
   tested 2026-05-27 (`src/docvqa/solvers/repl_only_solver.py`); on a
   2-doc smoke the agent SUBMITs "Unknown" in 1 iteration per question
   (0/5). The result is mechanistically obvious — "no perception → no
   answer" — and tests a strawman rather than a sharp prediction. The
   reframing above uses the three existing ablations as the mechanism
   evidence instead. The REPL-only solver code stays in the tree as
   documentation; it is not a paper cell.

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
- **Status:** accepted (prompt-parity portion); **partially revised by D-009**
  — the "per-solver inline CATEGORY_TIPS" implementation was found to
  over-apply the principle by baking tool-routing into 5 redundant
  per-solver inline dicts. D-009 refines: tool-agnostic semantic
  content per-dataset (in profile); tool-routing per-solver (in
  solver's `TASK_INSTRUCTIONS` + optional overlay). The parity rule
  carries forward unchanged.

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

## D-009: Refine D-007 — split semantic-per-profile from tool-routing-per-solver

- **Date:** 2026-05-27
- **Status:** accepted
- **Supersedes:** the per-solver-inline-CATEGORY_TIPS portion of D-007.
  The parity rule (all paper solvers audited to the same standard) is
  preserved.

**Decision.** Split prompt ownership along two axes:

- **Tool-agnostic semantic content** (per-dataset, per-category) lives
  in the dataset profile (`src/docvqa/datasets/profile.py`). Example:
  "for engineering drawings, verify each label is correctly associated
  with the part it connects to" is dataset-level guidance.
- **Tool-routing** (per-solver) lives in the solver — in
  `TASK_INSTRUCTIONS` for documenting the tool surface, plus an
  optional per-category overlay for tool-routing examples (similar to
  the old `FLAT_SOLO_TOOL_HINTS`). Example: "use `batch_look` to
  inspect the region" is solver-level.
- **Shared:** nothing per-category. `ANSWER_FORMATTING_RULES` is now
  considered part of the dataset profile, not a globally shared
  constant.

All paper solvers become **dataset-aware by default**. The
`solo`/`_da` distinction is dropped; merged solvers use the profile
system with DocVQA-2026 as the default profile.

**Reasoning.**

- D-007 (2026-05-27 earlier) was applied as "every solver has its own
  inline CATEGORY_TIPS dict." That over-shot the actual problem.
- The problem D-007 was solving (v1/v2 mess from prompt-scrub audit)
  was specifically about **tool-routing verbs leaking between
  solvers** when stripped for one solver's needs. Removing
  `search()` references from CATEGORY_TIPS to clean up leanest's dead
  references accidentally removed them for flat_solo (where they
  steered real tools).
- Tool-routing IS the cross-solver coupling problem; semantic content
  is not. Telling the agent "count carefully" is dataset-level
  guidance — it should be the same advice regardless of whether the
  solver uses `batch_look`, `display()`, or single-shot.
- The 2026-05-27 inline refactor produced 5 redundant copies of the
  same dataset-level semantic content across solver files (with
  tool-routing baked in differently per copy). That's exactly the
  duplication research code should avoid.
- The DA solvers (`*_da`) already do the right split via the profile
  system. Generalizing this everywhere — making it the default — is
  the cleanest end state.
- Cross-benchmark eval (MP-DocVQA, MMLongBench-Doc) becomes
  first-class: solvers are dataset-parameterized, not
  DocVQA-2026-coupled.

**Implications.**

- **Profile changes** (`src/docvqa/datasets/profile.py`):
  - DocVQA-2026 profile owns the canonical per-category semantic
    content (currently scattered across 5 solver files). Move the
    tool-agnostic content from `leanest_solo_solver.py`'s inline
    `CATEGORY_TIPS` into the profile (since leanest's version has the
    minimal tool routing of the five).
  - Strip tool-routing verbs from the moved content (`batch_look`
    references become tool-agnostic phrasings like "inspect the
    relevant region").
  - The 4 reconciled bullets (Leader-lines, TEXT TRUNCATION,
    COUNTING-OBJECTS protocol, CITED PAPER FINDINGS) all live in the
    DocVQA-2026 profile now.
  - `ANSWER_FORMATTING_RULES` content moves into the DocVQA-2026
    profile's `answer_formatting_rules` slot (it already is — just
    the source-of-truth moves from `prompts.py` to the profile).
- **Solver changes:**
  - Merge `leanest_solo` ← `leanest_solo_da` into a single
    DA-by-default solver. Same pattern for `no_loop_multi` ←
    `no_loop_multi_da` and other paired solvers.
  - New solvers (`leanest_ocr`, `repl_only`) made DA-by-default at
    creation. Drop inline `CATEGORY_TIPS` from them; replace with
    `profile.category_tips_fn()` calls.
  - Each solver keeps `TASK_INSTRUCTIONS` (documenting its tool
    surface) and an optional per-category tool-routing overlay (for
    OCR-bearing solvers, this is the `FLAT_SOLO_TOOL_HINTS`-equivalent
    that's solver-owned).
  - `rvlm_solver.py` becomes DA-capable.
- **`src/docvqa/prompts.py`:** can be reduced to nothing once
  back-compat is no longer needed. For now, keep the DEPRECATED block
  for shelved solvers; remove later.
- **Audit standard preserved:** parity rule from D-007 still applies —
  every paper solver passes the same audit. Now the parity question
  is "do the solvers correctly inherit from the profile?" instead of
  "are the 5 inline dicts consistent?" — easier to audit since
  there's only one source-of-truth.

**Migration plan** (task #21, #22):

1. Move tool-agnostic content from leanest_solo's inline
   `CATEGORY_TIPS` into DocVQA-2026 profile. Strip tool-routing.
2. Merge `leanest_solo_solver.py` ← `leanest_solo_da_solver.py`.
   Keep DA structure. Default profile = DocVQA-2026.
3. Same merge for `no_loop_multi`, and for the two new solvers
   (`leanest_ocr`, `repl_only`) at creation.
4. Update `rvlm_solver.py` to be DA-capable (currently DocVQA-2026
   only).
5. Update hydra configs to expose the dataset parameter.
6. Smoke test: every solver runs on DocVQA-2026, and at least leanest
   runs on MMLongBench-Doc and MP-DocVQA (verifies the DA path).

---

## D-010: Solver renames — behavior-based engineering names

- **Date:** 2026-05-27
- **Status:** accepted

**Decision.** Rename solvers from history-based to behavior-based
names. The current names trace development lineage (`flat_solo` →
`lean_solo` → `leanest_solo` reads as "started big, stripped down")
rather than describing what each solver actually does.

### Rename map

| Current | New | Role in paper |
|---|---|---|
| `leanest_solo` | **`rvlm`** | proposed method — code-LM in REPL + recursive VLM sub-call |
| `leanest_ocr` | **`rvlm_ocr`** | proposed method + OCR extension |
| `flat_solo` | **`rvlm_full`** (if kept per task #16) | kitchen-sink: rvlm + look + OCR |
| `rvlm` (current) | **`direct_vlm`** | single multimodal model in REPL — no sub-call, "direct" perception via `display()` |
| `no_loop_multi` | **`raw_vlm_multi`** | raw VLM baseline, multi-image |
| `no_loop` | **`raw_vlm_single`** | raw VLM baseline, single-image |
| `repl_only` | unchanged | documentation-only ablation; REPL with no perception |
| `official_baseline` | unchanged | competition kit prompt, verbatim |

Coupled with the D-009 merge: each `*_solo` ← `*_da` pair becomes a
single DA-by-default solver under the new name. The new names absorb
the merge.

### Why these names

- **`rvlm` (proposed method).** Recursive Language Model applied to
  VLM. Code-writing LLM in REPL invoking a VLM as a recursive sub-call.
  The name describes the architecture directly. Note D-005 caveat:
  "RVLM" is also the name of a concurrent arXiv paper
  (arXiv:2603.24224); this is an **engineering name only** — the
  paper-facing method name still needs to differ to avoid reviewer
  confusion, per D-005's pick-a-different-name implication.
- **`direct_vlm` (single-model alt angle).** Pairs lexically with
  `rvlm` along the architectural axis: recursive vs direct. In the
  current `rvlm` solver, a single multimodal model perceives images
  directly via `display()` into its own context — no delegation, no
  sub-call. "Direct" is the absence-of-recursion property that
  distinguishes it from `rvlm`.
- **`rvlm_ocr`, `rvlm_full`.** Compose with `rvlm` to indicate
  extensions: OCR-channel only (`rvlm_ocr`) and kitchen-sink with
  `look()` + OCR (`rvlm_full`).
- **`raw_vlm_*`.** The "raw" prefix marks unscaffolded baselines —
  one VLM call, no scaffold. Renames from history-suggestive
  `no_loop_*`.

### What stays as old names

- Historical docs in `docs/experiments/*.md` keep the old names —
  they record what happened at the time. Same principle as git commit
  messages.
- Existing run IDs in `output/runs/leanest-solo-*`, `flat-solo-*`,
  etc. keep their old names. New runs use new names.
- The `CLAUDE.md` "Best Results" table currently shows
  "Flat Solo SC-8" etc.; updated separately (entry-by-entry choice on
  whether to backport new names or note both).

### Implications

- File renames in `src/docvqa/solvers/`. Order matters: rename current
  `rvlm_solver.py` → `direct_vlm_solver.py` BEFORE renaming
  `leanest_solo_da_solver.py` → `rvlm_solver.py`, to avoid clobbering
  the `rvlm_solver.py` namespace.
- Config renames in `configs/solver/`. Each `*_da.yaml` is dropped
  after the merge folds its content into the single merged config.
- Class and factory-function renames throughout the codebase
  (`LeanestSoloDAProgram` → `RvlmProgram`, etc.).
- Hydra solver-config choice names change (`solver=leanest_solo` →
  `solver=rvlm`). Run scripts in `scripts/` need updating where they
  reference solver= choices.
- Update `paper/README.md` solver taxonomy table and `experiment-plan.md`
  to use new names. CLAUDE.md updated entry-by-entry.

---

## D-011: Deprioritize `rvlm_full` (kitchen-sink) cells

- **Date:** 2026-05-28
- **Status:** accepted

**Decision.** Defer the `rvlm_full` (kitchen-sink: `batch_look` +
`look` + `search` + `page_texts`) cells from the paper's critical
path. The `look()` ergonomic wrapper alongside `batch_look()` is
unlikely to make a meaningful difference; the OCR/search channel
already accounts for the kitchen-sink lift. Existing `rvlm_full` data
from the prompt-scrub audit and the MMLB/MPDV legacy DA cells stays as
supporting evidence (footnoted as "kitchen-sink, confounded with
look()" per D-006), but **no new `rvlm_full` cells are queued**.

**Reasoning.**

- Capability test: `look(image, query)` is essentially sugar for
  `batch_look([(image, query)])[0]`. Both call the same VLM with the
  same prompt path. The ergonomic difference is whether the agent
  writes `look(crop, q)` or `batch_look([(crop, q)])[0]`.
- The leanest_solo (now rvlm) prompt already documents the single-call
  idiom (`batch_look([(image, query)])[0]`), so removing `look()`
  doesn't restrict capability — only ergonomic. Likely <1pp effect.
- Adding a paper cell to test this ergonomic effect adds compute cost
  with low expected information value.
- The paper's OCR-extension story rests on `rvlm_ocr` (clean fork:
  batch_look + search + page_texts, no look). That cell is the right
  test for prediction 2.

**Implications.**

- Resolves task #16 (rvlm_full's paper role): **drop from headline,
  keep existing data as footnote/appendix only**.
- No queued rvlm_full cells in `coordination/amax7.md` or
  `coordination/amax1.md`.
- `docs/results.md`, `docs/paper/README.md`, `docs/paper/experiment-plan.md`
  mark rvlm_full as "deferred / appendix-only."
- Existing `flat-solo-da-mmlb-remote-*` and `flat-solo-da-mpdv-remote-*`
  run dirs (the rvlm_full DA cells on MMLB/MPDV) stay until `rvlm_ocr`
  DA cells supersede them (task #15); then deletable.
- Solver source files `rvlm_full_solver.py` + config + doc stay in the
  live codebase (not archived). Reactivation is cheap if a reviewer
  asks for the look-vs-batch_look isolation experiment.

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
