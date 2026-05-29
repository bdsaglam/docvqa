---
title: "ARIAL: An Agentic Framework for Document VQA with Precise Answer Localization"
shorthand: arial
arxiv: "2511.18192"
authors: "Pinaki Prasad Guha Neogi, Dheeraj Kulshrestha, Rajiv Ramnath"
date: "2025-11"
venue: ""
section: "Closest prior work — Agentic / tool-using DocVQA frameworks"
verification: "claims verified from PDF 2026-05-27 — benchmarks, numbers, and method confirmed"
date_added: 2026-05-27
---

# ARIAL: An Agentic Framework for Document VQA with Precise Answer Localization

**Position in our paper:** Closest prior work — the most direct competitor surfaced.

**Connection (from docs/paper/related-works.md):**

Reportedly achieves 0.887 ANLS on DocVQA via a modular pipeline (TrOCR + retrieval + answer generation) with pixel-grounded localization. This is the most direct competitor we surfaced. Our delta is the RLM framing (Python REPL state + recursive VLM sub-call) and the multi-page / multi-doc-type focus.

**Verified from PDF (2026-05-27).** Four **single-page** benchmarks: DocVQA **88.7 ANLS** / 50.1 mAP, FUNSD 90.0, CORD 85.5, SROIE 93.1. Pipeline = DB+TrOCR OCR → MiniLM retrieval → **fine-tuned Gemma 3-27B** QA (70k DocVQA/CORD/FUNSD pairs) → box grounding, planner = LLaMA 4 Scout. Caveats for head-to-head: **fine-tuned + single-page only**; only DocVQA-SP overlaps our space and we lean-exclude it. Byline is Neogi/Kulshrestha/Ramnath (Ohio State + Flairsoft), not "Mohammadshirazi" (that's DLaVA, the prior work ARIAL extends).
