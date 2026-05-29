---
title: "Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections"
shorthand: madqa
arxiv: "2603.12180"
authors: "Lukasz Borchmann, Jordy Van Landeghem, Michał Turski, Shreyansh Padarha, Ryan Othniel Kearns, Adam Mahdi, Niels Rogge, Clémentine Fourier, Siwei Han, Huaxiu Yao, Artemis Llabrés, Yiming Xu, Dimosthenis Karatzas, Hao Zhang, Anupam Datta"
date: "2026-03"
venue: ""
section: "Concurrent and adjacent work — primary positioning targets"
verification: "claims verified from PDF 2026-05-27 — benchmark stats, RLM critique, and headline numbers confirmed"
date_added: 2026-05-27
---

# Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections

**Position in our paper:** Planned baseline + benchmark (D-005).

**Connection (from docs/paper/related-works.md):**

Introduces the MADQA benchmark for multimodal agentic document QA and warns that "unconstrained" RLMs are an "efficiency catastrophe." Their critique is exactly what our paper's "focused/constrained instantiation" framing answers — we engage with it directly rather than reinventing the framing. We add their constrained-agent baseline to our experiment baselines, and frame our paper as the affirmative case for "constrained RLM works" on multimodal documents.

**Verified from PDF (2026-05-27).** Benchmark = **2,250 human-authored questions over 800 heterogeneous PDFs**, document-**collection** QA (cross-page/cross-doc multi-hop). Metric = LLM-judged **Accuracy** + novel **Kuiper effort-calibration** statistic (not ANLS). Best system *Gemini 3 Pro BM25 MLLM Agent* **82.2%**; ~18% oracle gap; thesis = **retrieval, not reasoning, is the bottleneck**. Documents are **fresh, not recycled** → low reuse risk with DocVQA-2026.

**Corrections / sharpening:** "efficiency catastrophe" is a paraphrase; actual = "catastrophic effort overhead of RLMs" (§5). They run unconstrained RLM **citing the same Zhang et al. 2025 RLM paper we instantiate** (Claude 4.5 Sonnet RLM: 270M tokens / ~$850, still loses to BM25-agent). **Tension:** this pre-empts a "constraining RLM helps" contribution (their result), and MADQA's collection-scale regime is where our **OCR extension**, not the OCR-free core, is relevant. Reframe our affirmative case to "**visual** recursive perception is the fix where perception, not retrieval, bounds the model." See `candidate-datasets.md` MADQA entry.
