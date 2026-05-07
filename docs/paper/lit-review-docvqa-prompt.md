# Literature Review Task — Document VQA Methods and Benchmarks

You are conducting a focused literature review for a research paper.
**You do not have access to our codebase or files.** Everything you need
is in this document. Your tools are paper-search MCPs (arXiv, Semantic
Scholar, Crossref, Google Scholar) and the open web.

## Scope and goal

This is the **second of two parallel lit reviews** for the same paper.

- **A separate lit review** covers the Recursive Language Models (RLM)
  paradigm and code-as-reasoning agents, since our method is framed as
  an application of RLM. Do **not** focus on that here.
- **This lit review** focuses on **document visual question answering
  (DocVQA) methods, benchmarks, and SOTA**. Map the landscape of how
  the field actually solves DocVQA — what models, what training, what
  retrieval, what evaluation — so we know who to compare against, what
  to cite, and what numbers to reference.

**Output:** save a 2–4 page markdown report as `lit-review-docvqa.md`
following the structure at the bottom of this brief.

---

## Background you need: our paper

We are publishing a paper that applies an agentic scaffold (a
code-capable LLM in a REPL with OCR retrieval and a VLM exposed as a
recursive sub-call) to multimodal document VQA. Headline empirical
result: the scaffold **lifts every model class** on the ICDAR 2026
DocVQA challenge — small open ≤8B, mid open 27B, frontier closed.

We use **off-the-shelf models with no domain-specific training, no
specialized document encoder, and no proprietary OCR**. We use
docling-serve OCR. The scaffold's tools are: BM25 search over per-page
OCR markdown, a VLM `look` tool that accepts arbitrary PIL Images
(pages or crops) with a question and returns text. One question per
session; a turn budget of ~25–30. Headline numbers reported as mean ± std
across independent trials.

**Why this lit review matters.** To position the paper, we need:

- A clear picture of how the field currently solves DocVQA — what's
  been tried, what works.
- A list of competitive **baselines** to compare against (or report
  alongside our results).
- Per-benchmark **SOTA numbers** so reviewers see we know the field.
- Awareness of any approaches structurally similar to ours that we
  must cite (so we don't accidentally claim novelty against work that
  exists).

---

## Background you need: the headline benchmark

**ICDAR 2026 Document VQA Challenge** — competition track at ICDAR 2026.

- **Dataset:** `VLR-CVC/DocVQA-2026` on HuggingFace.
- **Splits:** val (25 docs, 80 questions), test (48 docs, 160 questions).
  No train split — zero-shot / few-shot only.
- **Metric:** ANLS (Average Normalized Levenshtein Similarity, threshold
  0.80).
- **Categories:** 8 document types, balanced at 10 questions/category in
  val and 20 in test:
  `business_report`, `comics`, `engineering_drawing`, `infographics`,
  `maps`, `science_paper`, `science_poster`, `slide`.
- **Document scale:** avg 36 pages/doc on val, 33 on test. Some business
  reports exceed 280 pages.
- **Image scale:** test images average 8.5M pixels; some maps and
  posters exceed 246M pixels per page.
- **OCR coverage:** all docs OCR'd via docling-serve. Some categories
  yield 0 chars (maps, posters), motivating multimodal handling.
- **Question style:** mostly extraction (~60% what/which/how), with
  significant spatial reasoning and ~20% multi-hop arithmetic / cross-page.
- **Tier structure:** competition splits submissions into ≤8B, 8–35B,
  >35B parameter tiers; each has its own leaderboard.
- **Official baselines (test, direct prompting):** Gemini 3 Pro 37.50%,
  GPT-5.2 35.00%, Gemini 3 Flash 33.75%, GPT-5 Mini 22.50%. The
  benchmark is **not saturated**.

The ICDAR DocVQA challenge has prior editions in earlier years — these
prior winners and their system descriptions are part of what we want to
catalog.

---

## What we want from you

A structured map of the DocVQA literature. Specifically:

1. **The method landscape.** Group methods into clusters and describe
   each. We expect at least these clusters; add more if the literature
   demands:
   - **Trained document encoders / VLMs:** Donut, LayoutLMv2/v3, UDOP,
     Pix2Struct, mPLUG-DocOwl (1, 1.5, 2, ...), DocFormer, ERNIE-Layout,
     GeoLayoutLM, TILT, BROS, LiLT, StructuralLM, etc. What's the
     current SOTA among these?
   - **General multimodal LLMs applied to DocVQA:** GPT-4V/4o, Gemini,
     Claude, Qwen-VL, InternVL, LLaVA-family, MiniCPM-V, etc., used
     directly with prompting (no fine-tuning).
   - **Tool-using / agentic / retrieval-augmented DocVQA approaches:**
     anything that orchestrates OCR, retrieval, code execution, or
     external tools to solve DocVQA. This is the cluster closest to
     our method — be especially thorough.
   - **Specialized methods per document type:** chart-specific (UniChart,
     ChartLlama, ChartGemma...), infographic-specific, table-specific,
     etc.
   - **OCR-free vs OCR-augmented methods:** how is this debate
     currently positioned?

2. **Per-benchmark SOTA table.** For each benchmark below, identify
   current SOTA (best published number), the method that achieves it,
   and any "strong baseline" that's commonly cited. **Cite arxiv IDs
   inline so we can verify.**
   - **DocVQA (Mathew et al., 2020, original)** — the canonical
     single-page document VQA benchmark.
   - **MP-DocVQA** — multi-page document VQA.
   - **InfographicVQA** — infographic-specific.
   - **ChartQA** — chart-specific.
   - **SlideVQA** — slide-deck VQA.
   - **VisualMRC** — visual machine reading comprehension.
   - **DUDE** — document understanding for diverse documents.
   - **AI2D / ScienceQA-V** — science-question multimodal.
   - **TAT-DQA / TabFact / FinQA** — table-style document QA.
   - **(Others if relevant — e.g., MMLongBench-Doc, LongDocURL.)**

3. **ICDAR DocVQA challenge series.** Identify and summarize the prior
   editions and their winning approaches. Goal: understand what the
   challenge has historically rewarded and what trends are visible across
   years.

4. **Evaluation conventions.** Specifically ANLS — its definition,
   thresholding, why it's used over exact match for DocVQA, any known
   weaknesses or alternatives. Other DocVQA-relevant metrics that show
   up.

5. **Long-document and multi-page handling.** How do existing methods
   handle documents longer than a model's context window? Retrieval?
   Page-level pooling? Hierarchical summarization? Cropping? Patch-based
   pretraining?

6. **OCR pipelines used in the field.** What OCR engines are commonly
   paired with DocVQA methods? (PaddleOCR, Tesseract, Azure Document
   Intelligence, Amazon Textract, docling, Mistral OCR, etc.) Any
   recent work showing a specific OCR makes a measurable difference
   on DocVQA outcomes?

7. **Recommended baselines for our paper.** Based on what you find,
   propose a concrete shortlist (5–8) of methods or method numbers we
   should report alongside ours. For each, say *why* — what comparison
   does it enable?

8. **What we shouldn't claim.** Anything we'd be tempted to say in the
   paper (e.g., "no prior work uses an LLM with OCR retrieval and VLM
   tools on DocVQA") that the literature contradicts.

---

## Required output structure

1. **Landscape map** — 4–6 short subsections (one per method cluster
   above), each summarizing the cluster + naming 3–5 representative
   papers with arxiv IDs.

2. **SOTA table** — single table covering the benchmarks above:

   | Benchmark | Method (arxiv id) | SOTA | Year | Notes |

   Where SOTA is unclear or contested, say so.

3. **ICDAR DocVQA challenge history** — a short subsection per prior
   edition with the winner's method and key idea.

4. **Evaluation conventions** — a paragraph on ANLS and adjacent metrics.

5. **Long-document handling** — 2–3 paragraphs reviewing the strategies.

6. **OCR pipelines in the field** — short paragraph or table.

7. **Recommended baselines for our paper** — concrete bulleted list with
   reasoning.

8. **What we shouldn't claim** — bulleted list of common-but-false claims
   to avoid.

9. **Must-cite list** — bibtex-ready entries, ~20–40 papers, grouped by
   the same clusters as the landscape map.

---

## Constraints

- **Don't fabricate.** If unsure about a SOTA number or paper, say so
  and cite the source you found. Fabricated citations are worse than
  missing ones.
- **Verify titles, authors, dates** for headline-cited papers.
  Scholar). Open-web verification when in doubt.
- **Cite arxiv IDs / DOIs inline** for every claim, every SOTA number.
- **Length:** 2–4 pages of markdown is the target. Don't pad. The SOTA
  table can extend the length but should be dense.
- **Recency bias.** Prefer 2023–2026 work, but include foundational
  older papers (e.g., DocVQA 2020 paper itself, LayoutLM family
  pre-2022) where they are part of the standard reference set.

---

## Success criteria

After we read your report, we should be able to:

- Pick **2 secondary benchmarks** to evaluate our method on (from the
  candidate list above), with knowledge of their SOTA and standard
  baselines.
- Have a concrete shortlist of **5–8 baseline methods/numbers** to
  report alongside our own results.
- Understand the **ICDAR challenge history** well enough to position
  our submission in the series.
- Know what claims to avoid making in the related-work section.
- Have ~30 citations ready to slot into the related-work section,
  grouped by cluster.
