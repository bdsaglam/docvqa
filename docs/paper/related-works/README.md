# Related-works paper notes

Project-specific notes on every paper indexed in
[`../related-works.md`](../related-works.md) — one `{shorthand}-{arxiv_id}.md`
file each, holding frontmatter (title, arxiv, position, verification) and
our connection note. Paper artifacts (PDF, overview, full markdown, code
repo) are not stored in this project.

**Verification (2026-05-27).** Every arXiv ID below was probed against
`arxiv.org/abs/{id}` and the returned title matched the
`related-works.md` claim. **All entries are real — no fabricated
citations**, despite the `(needs verification)` flags those entries
still carry in `related-works.md`. ID corrections made during the
existence check: Toolformer `2302.04363`→`2302.04761`; VideoAtlas
resolved to `2603.17948`; the Nourbakhsh groundedness paper to
`2503.19120` (both had no ID in the source doc).

---

## Concurrent and adjacent work — primary positioning targets

| Paper | arXiv | Position | Notes |
|-------|-------|----------|-------|
| RVLM: Recursive Vision-Language Models with Adaptive Depth | 2603.24224 | Concurrent work (D-005) | [rvlm-2603.24224](rvlm-2603.24224.md) |
| Strategic Navigation or Stochastic Search? (MADQA) | 2603.12180 | Planned baseline + benchmark (D-005) | [madqa-2603.12180](madqa-2603.12180.md) |

## Foundational

| Paper | arXiv | Position | Notes |
|-------|-------|----------|-------|
| Recursive Language Models (RLM) | 2512.24601 | The paradigm our method instantiates | [rlm-2512.24601](rlm-2512.24601.md) |

## Closest prior work — Agentic / tool-using DocVQA frameworks

| Paper | arXiv | Position | Notes |
|-------|-------|----------|-------|
| ARIAL: Agentic Framework for Document VQA | 2511.18192 | Most direct competitor | [arial-2511.18192](arial-2511.18192.md) |
| VISOR: Agentic Visual RAG | 2604.09508 | Closest | [visor-2604.09508](visor-2604.09508.md) |
| AgenticOCR: Parsing Only What You Need | 2602.24134 | Closest | [agentic-ocr-2602.24134](agentic-ocr-2602.24134.md) |
| MDocAgent: Multi-Modal Multi-Agent Framework | 2503.13964 | Closest | [mdocagent-2503.13964](mdocagent-2503.13964.md) |
| DocDancer: Agentic Document-Grounded Information Seeking | 2601.05163 | Closest (adjacent) | [docdancer-2601.05163](docdancer-2601.05163.md) |
| ORCA: Orchestrated Reasoning with Collaborative Agents | 2603.02438 | Closest | [orca-2603.02438](orca-2603.02438.md) |
| SlideAgent: Hierarchical Agentic Framework | 2510.26615 | Closest / multi-page baseline | [slideagent-2510.26615](slideagent-2510.26615.md) |
| Doc-V*: Coarse-to-Fine Interactive Visual Reasoning | 2604.13731 | Closest | [doc-vstar-2604.13731](doc-vstar-2604.13731.md) |

## Closest prior work — Think-with-images trained VLMs

| Paper | arXiv | Position | Notes |
|-------|-------|----------|-------|
| DeepEyes: Incentivizing "Thinking with Images" via RL | 2505.14362 | RL-trained analog of our `direct_vlm` (alt-angle method) | [deepeyes-2505.14362](deepeyes-2505.14362.md) |

## Closest prior work — RLM applied to other modalities

| Paper | arXiv | Position | Notes |
|-------|-------|----------|-------|
| VideoAtlas: Navigating Long-Form Video in Logarithmic Compute | 2603.17948 | RLM in a non-document modality | [videoatlas-2603.17948](videoatlas-2603.17948.md) |

## Closest prior work — Code-as-reasoning vision agents

| Paper | arXiv | Position | Notes |
|-------|-------|----------|-------|
| Visual Programming (VisProg) | 2211.11559 | Lineage | [visprog-2211.11559](visprog-2211.11559.md) |
| ViperGPT: Visual Inference via Python Execution | 2303.08128 | Lineage | [vipergpt-2303.08128](vipergpt-2303.08128.md) |
| Chameleon: Plug-and-Play Compositional Reasoning | 2304.09842 | Lineage | [chameleon-2304.09842](chameleon-2304.09842.md) |

## Closest prior work — Trained document encoders and VLMs

| Paper | arXiv | Position | Notes |
|-------|-------|----------|-------|
| LayoutLMv3 (LayoutLM family) | 2204.08387 | Trained-encoder baseline class | [layoutlmv3-2204.08387](layoutlmv3-2204.08387.md) |
| Donut: OCR-free Document Understanding Transformer | 2111.15664 | Trained-encoder baseline class | [donut-2111.15664](donut-2111.15664.md) |
| Pix2Struct | 2210.03347 | Trained-encoder baseline class | [pix2struct-2210.03347](pix2struct-2210.03347.md) |
| UDOP: Unifying Vision, Text, and Layout | 2212.02623 | Trained-encoder baseline class | [udop-2212.02623](udop-2212.02623.md) |
| mPLUG-DocOwl (DocOwl family) | 2307.02499 | Trained-VLM baseline class | [docowl-2307.02499](docowl-2307.02499.md) |
| Qwen2.5-VL (Qwen-VL family) | 2502.13923 | Trained-VLM baseline / our backbone family | [qwen2.5-vl-2502.13923](qwen2.5-vl-2502.13923.md) |

## Related — Adjacent / long-document benchmarks

| Paper | arXiv | Position | Notes |
|-------|-------|----------|-------|
| DocVQA (original) | 2007.00398 | Benchmark | [docvqa-2007.00398](docvqa-2007.00398.md) |
| MP-DocVQA | 2212.05935 | Candidate second benchmark | [mp-docvqa-2212.05935](mp-docvqa-2212.05935.md) |
| MMLongBench-Doc | 2407.01523 | Long-doc generality experiment | [mmlongbench-doc-2407.01523](mmlongbench-doc-2407.01523.md) |
| InfographicVQA | 2104.12756 | Benchmark | [infographicvqa-2104.12756](infographicvqa-2104.12756.md) |
| SlideVQA | 2301.04883 | Benchmark | [slidevqa-2301.04883](slidevqa-2301.04883.md) |
| DUDE | 2305.08455 | Benchmark | [dude-2305.08455](dude-2305.08455.md) |
| ChartQA | 2203.10244 | Benchmark | [chartqa-2203.10244](chartqa-2203.10244.md) |
| VisualMRC | 2101.11272 | Benchmark | [visualmrc-2101.11272](visualmrc-2101.11272.md) |

## Related — Tool-using small models

| Paper | arXiv | Position | Notes |
|-------|-------|----------|-------|
| Toolformer | 2302.04761 | Small-models-with-tools evidence | [toolformer-2302.04761](toolformer-2302.04761.md) |

## Related — Evaluation conventions

| Paper | arXiv | Position | Notes |
|-------|-------|----------|-------|
| Scene Text VQA (original ANLS) | 1905.13648 | ANLS metric source | [anls-stvqa-1905.13648](anls-stvqa-1905.13648.md) |
| ANLS* | 2402.03848 | Structured-output metric | [anls-star-2402.03848](anls-star-2402.03848.md) |
| Where is this coming from? (groundedness) | 2503.19120 | Groundedness-aware evaluation | [groundedness-2503.19120](groundedness-2503.19120.md) |
