---
title: "RVLM: Recursive Vision-Language Models with Adaptive Depth"
shorthand: rvlm
arxiv: "2603.24224"
authors: ""
date: "2026-03"
venue: ""
section: "Concurrent and adjacent work — primary positioning targets"
verification: "verified 2026-05-27 — arXiv title matches related-works.md claim"
date_added: 2026-05-27
---

# RVLM: Recursive Vision-Language Models with Adaptive Depth

**Position in our paper:** Concurrent work (D-005). ~2-month gap is within the concurrent window for our target venues; treat as alongside, not prior art.

**Connection (from docs/paper/related-works.md):**

Recursive vision-language model with adaptive depth, reportedly on single-image medical scans (X-ray, MRI). Domain delta we own even if RVLM is real: (1) multi-page documents up to 280+ pages with diverse layouts (tables, infographics, maps, slides, comics), vs single-image medical imaging; (2) symbolic + visual hybrid retrieval — our scaffold combines BM25 over per-page OCR markdown with arbitrary-image VLM lookup; RVLM has no equivalent OCR-retrieval channel as far as we know; (3) multi-page navigation, a core challenge for us, not relevant in single-scan medical imaging. Action items: write the explicit "concurrent and complementary" paragraph; pick a method name clearly distinct from "Recursive Vision-Language Model" to reduce reviewer confusion.
