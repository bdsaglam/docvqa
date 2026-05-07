# Literature Review Task — RLM Applied to Document VQA

You are conducting a literature review for a research paper. **You do not
have access to our codebase or files.** Everything you need to understand
our work is in this document. Your tools are paper-search MCPs (arXiv,
Semantic Scholar, Crossref, Google Scholar) and the open web.

## What we are publishing

A paper that **applies the Recursive Language Models (RLM) paradigm to
multimodal document visual question answering (DocVQA)**.

We are **not** introducing RLM. RLM was published by Zhang, Kraska,
Khattab in December 2025 (arXiv:2512.24601). We are introducing a focused
instantiation of RLM where the recursive sub-call is specialized to a
**Vision-Language Model (VLM)** for visual perception of document pages,
and demonstrating empirically that this lifts performance on the ICDAR
2026 DocVQA challenge across the full model size spectrum (≤8B, 8–35B,
>35B parameter tiers).

Our novelty claim is in the **application + empirical results**, not the
architectural idea. The paper's success therefore depends on:

1. Clean novelty of the application (no one has applied RLM to DocVQA already).
2. Strong empirical lift across model classes.
3. Generalization beyond the headline benchmark.
4. Honest ablations isolating where the lift comes from.

## Your goal

Map the surrounding literature, identify must-cite work, find the closest
prior work, and surface comparisons reviewers will demand. The output is
a 2–4 page markdown report.

---

## Background you need: the foundational paper (RLM)

**Citation:** Zhang, A. L., Kraska, T., Khattab, O. *Recursive Language
Models.* arXiv:2512.24601, December 2025.

**Core idea.** LLMs have fixed context windows and suffer "context rot"
on long inputs. RLM addresses this by treating the prompt as an
**external REPL environment** rather than something to fit into the
context window. Specifically:

- The user's prompt P is loaded as a Python variable (e.g., `context`)
  in a persistent REPL. Only constant-size metadata (length, prefix,
  access instructions) is shown to the model directly.
- The LLM generates Python code to **symbolically explore** the prompt
  (e.g., `context[:10000]`, string ops, search).
- An `llm_query` function lets the LLM invoke itself or designated
  sub-models on transformed slices of the prompt — **recursive
  sub-calling**.
- The system iterates code-generation → execution → state-update until
  the LLM signals completion via a sentinel variable.

**What they evaluate.** Long-context **text** tasks: S-NIAH (single
needle), OOLONG (linear-complexity transformations), OOLONG-Pairs
(quadratic), BrowseComp-Plus (multi-hop QA over 6M–11M tokens),
LongBench-v2 CodeQA (23K–4.2M tokens). Models: GPT-5 and Qwen3-Coder-480B,
with smaller models (Qwen3-8B) used for sub-calls.

**Headline results.**
- BrowseComp-Plus 6M–11M tokens: RLM(GPT-5) 91.3% vs base GPT-5 0.0%.
- OOLONG-Pairs (quadratic): RLM(GPT-5) 58.0% F1 vs base <0.1%.
- Cost: RLM(GPT-5) ~$0.99/query on BrowseComp-Plus, **29% cheaper** than
  summarization baselines while handling far more data.
- Native RLM training: a Qwen3-8B fine-tuned on 1,000 RLM trajectories
  beat the base by **+28.3%** average across tasks and approached vanilla
  GPT-5 quality on three long-context tasks.

**Their ablations to know about.**
- REPL-only (no sub-calling) still beats task-agnostic baselines.
- Adding sub-calling lifts further by **10–59%** on info-dense tasks.
- Different base models use the recursion differently (GPT-5 conservative,
  Qwen3-Coder more liberal).

**Why this matters for our paper.** RLM is text-only. We extend it to the
multimodal document setting where the "context too long" problem is
acute (many pages, large detailed images, tables, infographics that don't
fit a multimodal model's context window). Our `llm_query` analogue is a
**VLM call** — visual perception, not just additional text-context budget.

---

## Background you need: the benchmark

**ICDAR 2026 Document VQA Challenge** — competition track at ICDAR 2026.

- **Dataset:** `VLR-CVC/DocVQA-2026` on HuggingFace.
- **Splits:** val (25 docs, 80 questions), test (48 docs, 160 questions).
  No train split — zero-shot / few-shot only.
- **Metric:** ANLS (Average Normalized Levenshtein Similarity, threshold 0.80).
- **Categories:** 8 document types, balanced at 10 questions/category in
  val and 20 in test:
  `business_report`, `comics`, `engineering_drawing`, `infographics`,
  `maps`, `science_paper`, `science_poster`, `slide`.
- **Document scale:** avg 36 pages/doc on val, 33 pages/doc on test.
  Some business reports exceed 280 pages.
- **Image scale:** test images average 8.5M pixels; some maps and posters
  exceed 246M pixels per page.
- **OCR availability:** 100% of docs have OCR coverage via `docling-serve`,
  but quality is uneven. Maps, posters, and minimal-text slides often
  yield 0 or very few OCR characters — **OCR alone is insufficient** for
  these categories. This motivates the multimodal approach.
- **Question style:** mostly extraction (what/which/how — ~60%), with
  significant spatial reasoning (in/on/locate) and multi-hop reasoning
  (~20% requires combining information across pages or arithmetic).
- **Tier structure:** the competition splits submissions into three
  parameter tiers: ≤8B, 8B–35B, >35B. Each tier has its own leaderboard.

**Sample questions illustrating difficulty:**

- *infographics:* "Assuming you get 15 USD/hour in Angola what's the
  number of coffees you could buy for the price of an iPod?"
- *maps:* "If I'm standing at the Pantheon and looking toward the
  Colosseum, which hill will I see most clearly to the right of it?"
- *slide:* "Assuming Q3 revenue is distributed by customer in the same
  way as Q3 backlog, and customers are equally distributed between them,
  what would be the Q3 revenue of the second largest customer?"

**Official baselines (test set, direct prompting, no scaffold):**

| Model | Test score |
|---|---|
| Gemini 3 Pro | 37.50% |
| GPT-5.2 | 35.00% |
| Gemini 3 Flash | 33.75% |
| GPT-5 Mini | 22.50% |

The benchmark is **not saturated** — even frontier proprietary models are
below 40%.

---

## Background you need: our method

**One-line summary:** A code-capable LLM operates in a REPL with the
document accessible via OCR retrieval and a VLM exposed as a recursive
sub-call for visual perception of pages.

**Components:**

1. **Main LLM in a REPL loop.** Code-capable LLM writes Python, code is
   executed, output feeds back into the next prompt, repeated for up to
   ~25–30 turns.
2. **Symbolic document access.** Per-page OCR markdown is indexed with
   BM25. The LLM calls `search(query, k)` from the REPL to retrieve
   relevant page snippets.
3. **VLM as recursive sub-call.** The LLM has `look(image, query)` and
   `batch_look([(image, query), ...])` tools, which call a VLM on a
   chosen page (or region) with a focused question and return the VLM's
   text answer. This is the multimodal analogue of RLM's `llm_query`.
4. **Per-category prompt tips.** Hand-crafted formatting hints loaded
   based on document category (e.g., infographics, maps, science_paper).
5. **One question per session** (not batched). Each question gets its own
   independent RLM session with the full tool suite and turn budget.

We do **not** present self-consistency / majority voting as part of the
method. We report mean ± std across independent trials. (Self-consistency
was used internally to pick a single competition submission; it is not a
contribution of this paper. Do not include the SC literature as part of
our positioning.)

**Variants we have studied (these become ablation points in the paper):**

- **Flat Solo** (full method): REPL + OCR retrieval + VLM tools + SC.
- **Lean Solo** (slightly trimmed budget, same components).
- **Leanest Solo** (no OCR retrieval, no `search()`, only VLM lookup):
  removes the symbolic-text channel and forces all information through
  the visual sub-call.
- **Flat Batch** (all questions for a doc in one session): worse by
  ~10pp; one question at a time wins by a large margin.

**Models we use.**

- **Open mid-size:** Qwen 3.5 27B, Qwen 3.6 27B (same model class for
  main LLM and VLM, hosted via vLLM on 3×A100).
- **Closed frontier:** Gemini 3 Pro / Gemini 3 Flash via Vertex AI.
- **Planned:** Qwen 3.5 9B, a Gemma variant (≤8B tier), and an
  alt-frontier closed model (GPT-5.x or Claude).

In the headline runs we use the **same model** for the main LLM and the
VLM (driven by the ICDAR tier rules — using a larger VLM would push the
submission into a different tier). The framework is general; main and
VLM can be different.

**Crucially: no specialized document encoder, no proprietary OCR, no
domain-specific training, no fine-tuning.** Off-the-shelf VLM +
off-the-shelf OCR + RLM scaffold.

---

## Background you need: our results so far

All paper-headline numbers are reported as **mean ± std across
independent trials** (no self-consistency voting).

**ICDAR 2026 DocVQA, Qwen 3.6 27B (open, both LLM and VLM):**

- 8-trial val mean: **44.06% ± 3.04pp** (range 40.0–47.5%).
- 8-trial test mean: needs to be computed from per-trial test scores
  (currently only the SC-8 voted score 43.75% is summarized). Variance
  expected to be ~3pp like val.

**ICDAR 2026 DocVQA, frontier closed (single-trial, will be replicated to ≥3):**

- Gemini 3 Pro baseline (no scaffold): **37.5% test** (official).
- Gemini 3.1 Pro main + Gemini 3 Flash VLM, our scaffold: **59.4% test**
  (single trial). This is a +21.9pp lift over the direct-prompt
  baseline; we will replicate before headlining.

**Per-category struggle:** the `maps` category stays around 20% even
after model upgrades. Spatial path-tracing on map imagery is a clear
failure mode.

**Cross-validation needed (planned for paper):**
- A second benchmark (candidates: original DocVQA 2020, InfographicVQA,
  SlideVQA, ChartQA, MP-DocVQA — final pick depends on your lit review).
- A third frontier closed model (GPT-5 / Claude) to show non-Gemini
  generality.
- Ablations: no-loop direct prompting baseline; OCR on/off; VLM sub-call
  on/off (OCR-only); VLM cropping on/off (page-only vs arbitrary-image
  zoom — isolates the "active perception" contribution); turn budget
  curve.

---

## Your specific search topics

**Primary (must cover thoroughly):**

1. **Recursive Language Models and adjacent paradigms.** The foundational
   paper is Zhang et al. arXiv:2512.24601 (Dec 2025). Find any follow-ups,
   contemporaneous work, or predecessors that frame "prompt as external
   environment + symbolic exploration." **Critically: search for any work
   applying RLM (or close paradigms) to multimodal / vision / document
   settings — if it exists, our novelty claim is at risk.**
2. **Code-as-reasoning vision agents.** VisProg (Gupta & Kembhavi,
   CVPR 2023), ViperGPT (Surís et al., ICCV 2023), Chameleon (Lu et al.,
   NeurIPS 2023), HuggingGPT-style work. How do they differ structurally
   from RLM? Do any treat the document/prompt as a symbolic environment?
3. **Agentic / tool-using approaches to document VQA.** Anything where an
   LLM/VLM orchestrates tools rather than answering directly, applied to
   documents. Especially recent (2024–2026) work.
4. **Document VQA architectures.** Both encoder-trained (Donut,
   LayoutLMv3, UDOP, Pix2Struct, mPLUG-DocOwl, DocFormer) and recent
   prompt-based or training-free approaches.
5. **Small-model + tool-use empirically beats bigger models.** Toolformer
   (Schick et al., NeurIPS 2023) and successors. Anything that
   demonstrates a tool scaffold making a small model competitive with
   frontier models — directly relevant to our cross-tier claim.

**Secondary (cover if relevant work exists):**

6. **Multi-page document understanding.** MP-DocVQA and related — what's
   been done for documents that don't fit in a single context?
7. **Adjacent VQA benchmarks.** InfographicVQA, ChartQA, SlideVQA,
   VisualMRC, AI2D — what's their format, current SOTA, typical baseline,
   how big is the test set? Goal: pick 2 secondary benchmarks for our
   generality claim.
8. **OCR-free vs OCR-based document understanding.** The Donut-era debate
   and where it's landed in 2024–2026.
9. **ICDAR 2026 DocVQA challenge.** Any official paper / overview, prior
   winners' system descriptions for DocVQA challenges (2023, 2024, 2025
   editions if they exist), baseline methodology.

---

## Required output

Save your report as a 2–4 page markdown file. Structure:

1. **Landscape map** — 3–5 paragraphs categorizing the space (e.g.,
   trained doc-VLMs; prompt-based vision agents; RLM and adjacent
   paradigms; code-as-reasoning vision agents).
2. **Closest prior work** — 5–10 papers, ranked by closeness to our
   RLM-for-DocVQA application, with 2–3 sentence summaries each. Be
   explicit about what's similar and what's different from our work.
3. **Where this paper sits** — 1 paragraph positioning vs the closest
   prior work. What's our delta given that RLM itself already exists?
4. **Must-cite list** — bibtex-ready entries, ~20–40 papers, grouped by
   topic. Match the topic structure in the search-topics section above.
5. **Comparisons reviewers will demand** — concrete list of baselines
   and experimental comparisons we should include. Be specific:
   "compare against method X on benchmark Y."
6. **Risks / gaps** — claims our paper makes that someone has already
   made, or near-misses where we'd lose novelty. **Special focus: has
   anyone applied RLM (or close paradigms) to multimodal docs already?**
7. **Naming inspiration** — what naming conventions similar methods use
   (helps us pick our own — currently a placeholder).
8. **Suggested venues** — based on what's published where in this space
   (ACL/EMNLP/NAACL Findings, CVPR/ICCV workshops, TMLR, ICDAR
   proceedings, etc.).

---

## Tools and constraints

- **Prefer recent (2023–2026)** but include foundational older work.
- **Cite arxiv IDs / DOIs inline** so claims can be verified later.
- **Don't fabricate.** If unsure, write "needs verification" and cite
  the source you found. Fabricated citations are worse than missing ones.
- **Verify titles, authors, and dates** for the headline-cited papers.
- **Length:** 2–4 pages of markdown is the target. Don't pad.

## Success criteria

After we read your report, we should be able to:

- Decide whether our novelty claim survives.
- Know what the closest 3–5 papers are and how to position against them.
- Have a concrete list of ~5 baselines to add to our experiment plan.
- Pick 2 secondary benchmarks for our generality experiments.
- Have a sense of which venue is right.
