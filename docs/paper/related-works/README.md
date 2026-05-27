# Related-works paper library

Local copies of every paper indexed in [`../related-works.md`](../related-works.md).
Each subfolder is `{shorthand}-{arxiv_id}/` and follows the `paper-lookup`
convention:

| File | Contents |
|------|----------|
| `notes.md` | Frontmatter (title, arxiv, position, verification) + our connection note, seeded from `related-works.md`. **This is the file to read first.** |
| `overview.md` | AlphaXiv overview (when server-rendered content was available). |
| `paper.md` | Full paper as markdown via `hf papers read` (when an HTML version existed). |
| `{id}.pdf` | The PDF (always present). |

**Verification (2026-05-27).** Every arXiv ID below was probed against
`arxiv.org/abs/{id}` and the returned title matched the `related-works.md`
claim. **All 33 entries are real — no fabricated citations**, despite the
`(needs verification)` flags those entries still carry in `related-works.md`.
ID corrections made during download: Toolformer `2302.04363`→`2302.04761`;
VideoAtlas resolved to `2603.17948`; the Nourbakhsh groundedness paper to
`2503.19120` (both had no ID in the source doc).

Some folders lack `overview.md` (AlphaXiv had no server-rendered overview) or
`paper.md` (no arXiv HTML version) — the PDF is the fallback in those cases.

---

## Concurrent and adjacent work — primary positioning targets

| Paper | arXiv | Position | Folder |
|-------|-------|----------|--------|
| RVLM: Recursive Vision-Language Models with Adaptive Depth | 2603.24224 | Concurrent work (D-005) | [rvlm-2603.24224](rvlm-2603.24224/) |
| Strategic Navigation or Stochastic Search? (MADQA) | 2603.12180 | Planned baseline + benchmark (D-005) | [madqa-2603.12180](madqa-2603.12180/) |

## Foundational

| Paper | arXiv | Position | Folder |
|-------|-------|----------|--------|
| Recursive Language Models (RLM) | 2512.24601 | The paradigm our method instantiates | [rlm-2512.24601](rlm-2512.24601/) |

## Closest prior work — Agentic / tool-using DocVQA frameworks

| Paper | arXiv | Position | Folder |
|-------|-------|----------|--------|
| ARIAL: Agentic Framework for Document VQA | 2511.18192 | Most direct competitor | [arial-2511.18192](arial-2511.18192/) |
| VISOR: Agentic Visual RAG | 2604.09508 | Closest | [visor-2604.09508](visor-2604.09508/) |
| AgenticOCR: Parsing Only What You Need | 2602.24134 | Closest | [agentic-ocr-2602.24134](agentic-ocr-2602.24134/) |
| MDocAgent: Multi-Modal Multi-Agent Framework | 2503.13964 | Closest | [mdocagent-2503.13964](mdocagent-2503.13964/) |
| DocDancer: Agentic Document-Grounded Information Seeking | 2601.05163 | Closest (adjacent) | [docdancer-2601.05163](docdancer-2601.05163/) |
| ORCA: Orchestrated Reasoning with Collaborative Agents | 2603.02438 | Closest | [orca-2603.02438](orca-2603.02438/) |
| SlideAgent: Hierarchical Agentic Framework | 2510.26615 | Closest / multi-page baseline | [slideagent-2510.26615](slideagent-2510.26615/) |
| Doc-V*: Coarse-to-Fine Interactive Visual Reasoning | 2604.13731 | Closest | [doc-vstar-2604.13731](doc-vstar-2604.13731/) |

## Closest prior work — RLM applied to other modalities

| Paper | arXiv | Position | Folder |
|-------|-------|----------|--------|
| VideoAtlas: Navigating Long-Form Video in Logarithmic Compute | 2603.17948 | RLM in a non-document modality | [videoatlas-2603.17948](videoatlas-2603.17948/) |

## Closest prior work — Code-as-reasoning vision agents

| Paper | arXiv | Position | Folder |
|-------|-------|----------|--------|
| Visual Programming (VisProg) | 2211.11559 | Lineage | [visprog-2211.11559](visprog-2211.11559/) |
| ViperGPT: Visual Inference via Python Execution | 2303.08128 | Lineage | [vipergpt-2303.08128](vipergpt-2303.08128/) |
| Chameleon: Plug-and-Play Compositional Reasoning | 2304.09842 | Lineage | [chameleon-2304.09842](chameleon-2304.09842/) |

## Closest prior work — Trained document encoders and VLMs

| Paper | arXiv | Position | Folder |
|-------|-------|----------|--------|
| LayoutLMv3 (LayoutLM family) | 2204.08387 | Trained-encoder baseline class | [layoutlmv3-2204.08387](layoutlmv3-2204.08387/) |
| Donut: OCR-free Document Understanding Transformer | 2111.15664 | Trained-encoder baseline class | [donut-2111.15664](donut-2111.15664/) |
| Pix2Struct | 2210.03347 | Trained-encoder baseline class | [pix2struct-2210.03347](pix2struct-2210.03347/) |
| UDOP: Unifying Vision, Text, and Layout | 2212.02623 | Trained-encoder baseline class | [udop-2212.02623](udop-2212.02623/) |
| mPLUG-DocOwl (DocOwl family) | 2307.02499 | Trained-VLM baseline class | [docowl-2307.02499](docowl-2307.02499/) |
| Qwen2.5-VL (Qwen-VL family) | 2502.13923 | Trained-VLM baseline / our backbone family | [qwen2.5-vl-2502.13923](qwen2.5-vl-2502.13923/) |

## Related — Adjacent / long-document benchmarks

| Paper | arXiv | Position | Folder |
|-------|-------|----------|--------|
| DocVQA (original) | 2007.00398 | Benchmark | [docvqa-2007.00398](docvqa-2007.00398/) |
| MP-DocVQA | 2212.05935 | Candidate second benchmark | [mp-docvqa-2212.05935](mp-docvqa-2212.05935/) |
| MMLongBench-Doc | 2407.01523 | Long-doc generality experiment | [mmlongbench-doc-2407.01523](mmlongbench-doc-2407.01523/) |
| InfographicVQA | 2104.12756 | Benchmark | [infographicvqa-2104.12756](infographicvqa-2104.12756/) |
| SlideVQA | 2301.04883 | Benchmark | [slidevqa-2301.04883](slidevqa-2301.04883/) |
| DUDE | 2305.08455 | Benchmark | [dude-2305.08455](dude-2305.08455/) |
| ChartQA | 2203.10244 | Benchmark | [chartqa-2203.10244](chartqa-2203.10244/) |
| VisualMRC | 2101.11272 | Benchmark | [visualmrc-2101.11272](visualmrc-2101.11272/) |

## Related — Tool-using small models

| Paper | arXiv | Position | Folder |
|-------|-------|----------|--------|
| Toolformer | 2302.04761 | Small-models-with-tools evidence | [toolformer-2302.04761](toolformer-2302.04761/) |

## Related — Evaluation conventions

| Paper | arXiv | Position | Folder |
|-------|-------|----------|--------|
| Scene Text VQA (original ANLS) | 1905.13648 | ANLS metric source | [anls-stvqa-1905.13648](anls-stvqa-1905.13648/) |
| ANLS* | 2402.03848 | Structured-output metric | [anls-star-2402.03848](anls-star-2402.03848/) |
| Where is this coming from? (groundedness) | 2503.19120 | Groundedness-aware evaluation | [groundedness-2503.19120](groundedness-2503.19120/) |
