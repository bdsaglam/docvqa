# Literature Review Brief — DocVQA Scaffold Paper

## Task

Conduct a focused literature review for a paper that **applies the
Recursive Language Models (RLM) paradigm to multimodal document VQA**.
Goal: map the landscape, identify must-cite work, find positioning, and
surface comparisons that reviewers will demand.

**Output:** save a 2–4 page markdown report to `docs/paper/lit-review.md`
and add new paper entries to `docs/paper/related-works.md` (use the
conventions documented at the bottom of that file).

The foundational paper our work builds on:

- **Zhang, A. L., Kraska, T., Khattab, O.** *Recursive Language Models.*
  arXiv:2512.24601, Dec 2025.
- Local copy at `~/obsidian/Knowledge/Sources/Papers/rlm-2512.24601/`
  — read `overview.md` for the summary, `full.md` for the full text.
- We are **not** introducing RLM. We are introducing an application of
  RLM to multimodal document VQA.

## Repo context — read these first

You have read access to this repo. Skim these files before searching:

- `/home/baris/repos/docvqa/CLAUDE.md` — project overview, best results, infra
- `/home/baris/repos/docvqa/docs/results.md` — full experiment results
- `/home/baris/repos/docvqa/docs/dataset.md` — dataset description
- `/home/baris/repos/docvqa/docs/solvers/` — solver design docs
- `/home/baris/repos/docvqa/src/docvqa/solvers/` — actual solver implementations
- `/home/baris/repos/docvqa/src/docvqa/prompts.py` — prompts and tool descriptions
- `/home/baris/repos/docvqa/docs/paper/README.md` — paper goals and headline claim

## Method summary (so you know what we're positioning)

Our method is a focused instantiation of RLM for multimodal document VQA:

- **Main LLM in a REPL loop.** Code-capable LLM writes Python, it's
  executed, output feeds back in, repeat for a bounded number of turns.
- **Document accessible via OCR retrieval.** Per-page OCR markdown is
  indexed with BM25 and queryable from the REPL — the symbolic
  exploration channel.
- **VLM as recursive sub-call.** A vision-language model is exposed to
  the main agent as a tool for visual perception of pages or regions
  too detailed/numerous to fit the main agent's context window. This
  is the multimodal specialization of RLM's `llm_query`.
- Per-category prompt tips (e.g., infographics, maps, science_paper) that
  steer answer formatting.
- **Model-agnostic.** Tested with Qwen 3.5/3.6 27B (open) and Gemini 3
  Pro/Flash (closed). Will be tested on smaller open models (≤8B: Qwen
  3.5 9B, Gemma).
- **In headline runs we use the same model for main + VLM** (driven by
  ICDAR competition tier rules), but the framework permits any pairing.
- **No specialized document encoder, no proprietary OCR, no
  domain-specific training.**

The paper's headline claim: **applying RLM to document VQA lifts every
model class** — small open, mid open, frontier closed — by double-digit
accuracy on the ICDAR 2026 DocVQA challenge, with generality validated
on additional benchmarks (TBD).

The novelty is in the **application**, not the architecture. Reviewers
will probe: how clearly novel is the application vs RLM? What changes
when the sub-call is a VLM rather than an LLM? What's specific about
DocVQA that makes the RLM idea pay off?

## Search topics

Primary (must cover):

1. **Recursive Language Models and adjacent paradigms** — the foundational
   RLM paper (Zhang et al. 2512.24601) plus any follow-ups, predecessors,
   or contemporaneous work on REPL-based / context-as-environment LLMs.
   Especially: papers that apply RLM-style ideas to multimodal settings.
2. **Code-as-reasoning agents for vision** — VisProg, ViperGPT, Chameleon,
   HuggingGPT-style work where an LLM orchestrates vision tools via code.
   How do they differ structurally from RLM? Do any treat the prompt /
   document as a symbolic environment?
3. **Agentic / tool-using approaches to document VQA specifically** —
   anything where the model orchestrates tools rather than answering
   directly, applied to docs. Critical: has anyone applied RLM-style
   ideas to DocVQA already? If yes, our novelty claim is weaker.
4. **Document VQA architectures** — both encoder-based (Donut, LayoutLMv3,
   UDOP, Pix2Struct, mPLUG-DocOwl, DocFormer) and recent prompt-based or
   training-free approaches.
5. **Small-model + tool-use empirically beats bigger models** — Toolformer
   and successors; any work showing tool scaffolds make small models
   competitive with frontier models.

Secondary (cover if relevant work exists):

6. **Multi-page document understanding** — MP-DocVQA and related.
7. **Adjacent VQA benchmarks** — InfographicVQA, ChartQA, SlideVQA, VisualMRC,
   AI2D — what's their format, what's the SOTA, what's the typical baseline?
8. **OCR-free vs OCR-based document understanding** — the Donut-era debate
   and where it's landed.
9. **ICDAR 2026 DocVQA challenge** — official paper / overview if any,
   prior winners' system descriptions, baseline methodology.

Excluded (per `decisions.md` D-003): self-consistency / majority voting.
We are dropping SC from the paper's framing and reporting mean ± std
across independent trials instead.

## Tools to use

- `mcp__plugin_scholarly_paper-search__*` for arXiv / Semantic Scholar /
  Crossref / Google Scholar searches and downloads
- Prefer recent (2023–2026) work but include foundational older papers
- Cite arxiv IDs / DOIs inline so claims can be verified

## Required output structure (`docs/paper/lit-review.md`)

1. **Landscape map** — 3–5 paragraphs categorizing the space into clusters
   (e.g., trained doc-VLMs, prompt-based vision agents, RLM and adjacent
   paradigms, code-as-reasoning vision agents).
2. **Closest prior work** — 5–10 papers, ranked by closeness, with 2–3
   sentence summaries. Be explicit about what's similar and what's
   different from our RLM-for-DocVQA application.
3. **Where this paper sits** — 1 paragraph positioning vs the closest
   prior work. What's our delta given that RLM itself already exists?
4. **Must-cite list** — bibtex-ready entries, ~20–40 papers, grouped by
   topic. Mirror the categories in `docs/paper/related-works.md`.
5. **Comparisons reviewers will demand** — concrete list of baselines and
   experimental comparisons we should include.
6. **Risks / gaps** — claims our paper makes that someone has already made,
   or near-misses where we'd lose novelty. *Special focus*: has anyone
   applied RLM (or close paradigms) to multimodal docs already?
7. **Naming inspiration** — what naming conventions similar methods use
   (helps us pick our own).
8. **Suggested venues** — based on what's published where in this space.

Additionally: **append entries** to `docs/paper/related-works.md` for the
papers worth tracking long-term, following that file's conventions
(citation, obsidian path if downloaded, connection note). Don't duplicate
the full lit-review report there — it's an index, not a copy of the
report.

## Constraints

- **Don't fabricate.** If unsure, write "needs verification" and cite the
  source you found.
- **Length:** 2–4 pages of markdown. Don't pad.
- **Verify dates / authors** for the headline-cited papers — fabricated
  citations are worse than missing ones.

## Success criteria

After we read the report, we should be able to:

- Decide whether our novelty claim survives.
- Know what the closest 3–5 papers are and how to position against them.
- Have a concrete list of ~5 baselines to add to `experiment-plan.md`.
- Pick 2 secondary benchmarks for the generality experiments.
- Have a sense of which venue is right.
