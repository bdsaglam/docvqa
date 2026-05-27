---
title: "Recursive Language Models"
shorthand: rlm
arxiv: "2512.24601"
authors: "Alex L. Zhang, Tim Kraska, Omar Khattab"
date: "2025-12"
venue: ""
section: "Foundational"
verification: "verified 2026-05-27 — arXiv title matches related-works.md claim"
date_added: 2026-05-27
---

# Recursive Language Models

**Position in our paper:** Foundational — the paradigm our method instantiates.

**Connection (from docs/paper/related-works.md):**

Our method is a focused instantiation of RLM applied to multimodal document VQA. We borrow the central architectural idea — the prompt as an external REPL environment, accessed symbolically by a code-capable LLM, with a recursive sub-call available for delegated processing. We specialize the sub-call as a VLM for visual perception of pages too detailed/numerous to fit the main agent's context window. Differences: (modality) RLM evaluates text long-context tasks (BrowseComp-Plus, LongBench-v2 CodeQA, OOLONG, S-NIAH); we evaluate multimodal document VQA. (sub-call type) RLM's llm_query is LLM→LLM; ours is LLM→VLM, bringing a different capability (vision), not just more context budget. (recursion depth) single level main→VLM. (context source) RLM's context is the raw long input; ours is the document's pages exposed via OCR retrieval (BM25 over per-page OCR markdown) plus image lookup. Findings we align with: REPL alone significantly lifts baseline; sub-calling adds 10–59% on information-dense tasks; RLM-trained Qwen3-8B outperformed base by 28.3% (supports our "lifts every model class" thesis at the small-model end). Ablations to mirror: REPL-without-sub-calling (≈ our no-VLM/OCR-only ablation), REPL+sub-calling vs no-REPL (≈ our no-loop baseline). Note: cite as the source of the paradigm — we claim the application + empirical results in the multimodal document setting, not the architectural idea.
