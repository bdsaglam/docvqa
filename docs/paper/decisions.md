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
