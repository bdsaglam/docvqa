# Related Works — DocVQA Scaffold Paper

Running index of papers relevant to this work. Paper PDFs / notes are kept
in obsidian, not copied into this repo. Each entry: citation, obsidian
path (if downloaded), and a short note on the connection.

**Last updated:** 2026-05-01, from `lit-review-1.md` (RLM-focused) and
`lit-review-2.md` (DocVQA-focused).

**Verification status.** Most entries below come from the lit-review
agents; they have **not been independently verified**. Entries flagged
**(needs verification)** are high-stakes — verify against arXiv before
acting on them in the paper. Lit reviews are known to fabricate
plausible-looking citations.

Convention:
- **Foundational** — the paper our work directly builds on.
- **Novelty threat** — papers that, if real, materially shrink our
  contribution. Verify these first.
- **Closest prior work** — adjacent methods we position against.
- **Related** — broader context.
- **Baseline** — a method we compare against in experiments.

---

## Concurrent and adjacent work — primary positioning targets

These two papers were surfaced as the highest-stakes positioning items
in the lit reviews. Per `decisions.md` D-005, neither warrants
abandoning or dramatically reframing the project, but both need verification
on arXiv before they land in the paper.

### RVLM: Recursive Vision-Language Models with Adaptive Depth — concurrent work

- **Citation:** *RVLM: Recursive Vision-Language Models with Adaptive
  Depth.* arXiv:2603.24224 (Mar 2026). **(needs verification)**
- **Position:** **concurrent work** (D-005). Two-month gap is within the
  typical concurrent window for our target venues; treat as alongside,
  not prior art.
- **Domain delta** (what we own even if RVLM is real):
  - **Medical imaging vs multi-page documents.** RVLM reportedly
    handles single-image medical scans (X-ray, MRI). Our setting is
    multi-page documents up to 280+ pages with diverse layouts (tables,
    infographics, maps, slides, comics, etc.).
  - **Symbolic + visual hybrid retrieval.** Our scaffold combines BM25
    over per-page OCR markdown with arbitrary-image VLM lookup. RVLM
    has no equivalent OCR-retrieval channel as far as we currently know.
  - **Multi-page navigation.** A core challenge in our setting; not
    relevant in single-scan medical imaging.
- **Action items:**
  1. Verify arXiv:2603.24224 exists. Download to obsidian if real.
  2. After reading, write the explicit "concurrent and complementary"
     paragraph for related work.
  3. **Pick a method name that clearly differs from "Recursive
     Vision-Language Model"** — even under concurrent framing, distinct
     naming reduces reviewer confusion.

### MADQA — Borchmann et al. *Strategic Navigation or Stochastic Search?*

- **Citation:** Borchmann, Ł., et al. *Strategic Navigation or
  Stochastic Search? How Agents and Humans Reason Over Document
  Collections.* arXiv:2603.12180 (2026). **(needs verification)**
- **Position:** **planned baseline + benchmark** (D-005). Reportedly
  introduces the MADQA benchmark for multimodal agentic doc QA *and*
  warns that "unconstrained" RLMs are an "efficiency catastrophe."
- **Why this matters for us:** their critique is exactly what our
  paper's "focused instantiation" framing answers. We engage with it
  directly rather than reinventing the framing.
- **Action items:**
  1. Verify arXiv:2603.12180 exists. Download to obsidian; user reads.
  2. Add MADQA to the benchmark candidate list in
     `experiment-plan.md` (done).
  3. Add their **constrained-agent baseline** to our experiment
     baselines (done in experiment-plan.md baseline shortlist).
  4. Frame our paper as the affirmative case for "constrained RLM
     works" on multimodal documents.

---

## Foundational

### Recursive Language Models (RLM)

- **Citation:** Zhang, A. L., Kraska, T., Khattab, O. *Recursive Language
  Models.* arXiv:2512.24601 (Dec 2025).
- **Obsidian:** `~/obsidian/Knowledge/Sources/Papers/rlm-2512.24601/`
  (`overview.md`, `notes.md`, `full.md`, `paper.pdf`)
- **Connection:** Our method is a focused instantiation of RLM applied
  to multimodal document VQA. We borrow the central architectural idea —
  the prompt as an external REPL environment, accessed symbolically by a
  code-capable LLM, with a recursive sub-call available for delegated
  processing. We specialize the sub-call as a **VLM** for visual perception
  of pages too detailed/numerous to fit the main agent's context window.
- **Differences:**
  - **Modality:** RLM evaluates on text long-context tasks (BrowseComp-Plus,
    LongBench-v2 CodeQA, OOLONG, S-NIAH). We evaluate on multimodal document
    VQA (ICDAR 2026 DocVQA + others).
  - **Sub-call type:** RLM's `llm_query` is general (LLM → LLM). Ours is
    specialized: LLM → VLM. The sub-call brings a different *capability*
    (vision), not just additional context budget.
  - **Recursion depth:** Single level (main → VLM); RLM also primarily
    operates at single level but is general.
  - **Context source:** RLM's `context` variable is the raw long input. Ours
    is the document's pages exposed via OCR retrieval (BM25 over per-page
    OCR markdown) plus image lookup.
- **Their findings we align with:**
  - REPL environment alone (no sub-call) significantly lifts baseline.
  - Sub-calling adds 10–59% additional improvement on information-dense tasks.
  - RLM-trained Qwen3-8B outperformed base by 28.3% — supports our
    "lifts every model class" thesis at the small-model end.
- **Their ablations to mirror:**
  - REPL-without-sub-calling (≈ our no-VLM / OCR-only ablation).
  - REPL+sub-calling vs no-REPL (≈ our no-loop baseline).
- **Note for the paper:** Cite as the source of the paradigm. Be
  explicit that we are not claiming the architectural idea — we are
  claiming the application + empirical results in the multimodal
  document setting.

---

## Closest prior work

### Agentic / tool-using DocVQA frameworks

This is the cluster most likely to challenge our novelty. Sourced from
`lit-review-2.md` — all need verification.

- **ARIAL: An Agentic Framework for Document VQA with Precise Answer
  Localization.** Mohammadshirazi et al. arXiv:2511.18192 (NeurIPS 2025).
  **(needs verification)** Reportedly achieves **0.887 ANLS on DocVQA**
  via a modular pipeline (TrOCR + retrieval + answer generation) with
  pixel-grounded localization. **This is the most direct competitor we
  surfaced.** Position: our delta is the RLM framing (REPL + recursive
  sub-call) and the multi-page / multi-doc-type focus.
- **VISOR: Agentic Visual Retrieval-Augmented Generation via Iterative
  Search and Over-horizon Reasoning.** Wu et al. arXiv:2604.09508
  (2026). **(needs verification)** Iterative visual RAG with structured
  "Evidence Space" and "Intent Injection" to prevent search drift.
  Closest to our scaffold's iterative refinement loop. Position: our
  delta is the explicit RLM lineage (Python REPL state) and OCR-augmented
  retrieval channel.
- **AgenticOCR: Parsing Only What You Need for Efficient
  Retrieval-Augmented Generation.** Jin et al. arXiv:2602.24134 (2026).
  **(needs verification)** Query-driven OCR — selectively recognizes
  regions of interest rather than pre-processing every page. Conceptually
  parallel to our agent's targeted `look()` calls but on the OCR side.
- **MDocAgent: A Multi-Modal Multi-Agent Framework for Document
  Understanding.** Wang et al. (Aiming Lab, 2025/2026).
  **(needs verification)** Five specialized agents (general, critical,
  text, image, summarizing) collaborating via multimodal context
  retrieval. Strong baseline on MMLongBench-Doc. Position: our delta is
  the single-agent RLM-style design vs their multi-agent orchestration.
- **DocDancer: Towards Agentic Document-Grounded Information Seeking.**
  arXiv:2601.05163 (2026). **(needs verification)** Search + read tools
  for iterative document exploration. Adjacent in spirit.
- **ORCA: Orchestrated Reasoning with Collaborative Agents for Document
  VQA.** Zhang et al. arXiv:2603.02438 (2026). **(needs verification)**
  "Thinker" agent generates a reasoning path, routes sub-tasks to
  specialized agents (tables/figures/forms/handwritten).
- **SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual
  Document Understanding.** arXiv:2510.26615 (2025).
  **(needs verification)** Multi-level (global / page / element) agent
  for slide decks. Not RLM-style but a strong baseline for multi-page
  agentic doc work.
- **Doc-V*: Coarse-to-Fine Interactive Visual Reasoning for Multi-Page
  Document VQA.** arXiv:2604.13731 (2026). **(needs verification)**
  Iterative coarse-to-fine visual reasoning for multi-page DocVQA.

### RLM applied to other modalities

Adjacent applications of the RLM idea in non-document settings — useful
for showing RLM is being instantiated across modalities, with our paper
covering the document slot.

- **VideoAtlas: Navigating Long-Form Video in Logarithmic Compute.**
  (2026) **(needs verification)** Reportedly applies RLM to video
  understanding via structured visual grids. Confirms the broader pattern
  (RLM-for-non-text-modalities). If real, our paper joins this line.

### Code-as-reasoning vision agents

Foundational to the broader paradigm; we cite for lineage but they're
not direct competitors on DocVQA.

- **VisProg.** Gupta & Kembhavi. *Visual Programming: Compositional
  visual reasoning without training.* CVPR 2023. — LLM generates Python
  programs orchestrating vision modules.
- **ViperGPT.** Surís, Menon, Vondrick. *ViperGPT: Visual Queries as
  Python Programs.* ICCV 2023. — Python execution + vision tools.
- **Chameleon.** Lu et al. NeurIPS 2023. — Plug-and-play tool composition
  for multimodal reasoning.

### Trained document encoders and VLMs

These are the architecture-trained baseline class. Not direct competitors
to our scaffold (we don't train), but standard reference set.

- **LayoutLM family** (LayoutLM/v2/v3, Xu et al., KDD 2020 → ACM MM 2022).
- **Donut.** Kim et al. *OCR-free Document Understanding Transformer.*
  ECCV 2022.
- **Pix2Struct.** Lee et al. ICML 2023.
- **UDOP.** Tang et al. CVPR 2023.
- **mPLUG-DocOwl** family. Ye et al. 2023+.
- **Qwen-VL family** (Qwen2.5-VL, Qwen3-VL). Qwen3-VL 235B reportedly
  reaches **0.971 ANLS on the original DocVQA** (near-saturation).
  **(needs verification)**

---

## Related

### ICDAR DocVQA challenge series

Prior editions and reported winners (from `lit-review-2.md` — verify
before citing).

- **ICDAR 2021 Task 2 (Document Collection VQA)** — Infrrd / Infrrd-RADAR
  (retrieval-augmented). **(needs verification)**
- **ICDAR 2021 Task 3 (Infographics VQA)** — Applica.ai, 0.6120 ANLS.
  **(needs verification)**
- **ICDAR 2023 (VQAonBD, business documents)** — Team Upstage KR, 95.9%
  accuracy via fine-tuned Donut. **(needs verification)**
- **ICDAR 2024 (Handwritten VQA)** — Team PA_VCG, 0.643 ANLS.
  **(needs verification)**
- **ICDAR 2026** — current edition; we're an entrant in the 8–35B tier.

### Adjacent / long-document benchmarks (candidates for our second benchmark)

- **DocVQA (original).** Mathew et al. WACV 2021. Single-page extractive.
  Reported near-saturation by frontier VLMs.
- **MP-DocVQA.** Multi-page DocVQA. Reported SOTA: AVIR Framework, 0.8458
  ANLS (2025). **(needs verification)**
- **MMLongBench-Doc.** Ma et al. NeurIPS 2024 (arXiv:2407.01523). 1,082
  expert-annotated long-context questions. Reported GPT-4o ≈ 44.9% F1.
  Strong candidate for our long-doc generality experiment.
- **InfographicVQA.** Mathew et al. — used as the ICDAR 2021 Task 3.
- **SlideVQA.** Tanaka et al. AAAI 2023.
- **DUDE.** Document Understanding Dataset and Evaluation. Diverse doc
  types.
- **ChartQA.** Specialized chart reasoning.
- **VisualMRC.** Visual machine reading comprehension.

### Tool-using small models

- **Toolformer.** Schick et al. NeurIPS 2023. — Early demonstration that
  small models with tools can be competitive.

### Evaluation conventions

- **ANLS metric definition.** Biten et al. *Scene Text Visual Question
  Answering.* ICCV 2019. — Original ANLS definition.
- **ANLS\*.** arXiv:2402.03848 (2024). — Extension for structured
  outputs (dicts, lists).
- **Groundedness in DocVQA evaluation.** Nourbakhsh et al. NAACL Findings
  2025. *Where is this coming from? Making groundedness count.*

---

## Baselines (in our experiments)

### Official ICDAR 2026 DocVQA baselines (test set)

From `docs/results.md:21-26`:

- Gemini 3 Pro: 37.50%
- GPT-5.2: 35.00%
- Gemini 3 Flash: 33.75%
- GPT-5 Mini: 22.50%

Direct-prompt (no scaffold) — the "raw model" comparison.

### SOTA on adjacent benchmarks (for context, not direct competition)

| Benchmark | Best reported | Method | Notes |
|---|---|---|---|
| DocVQA (original) | 0.971 ANLS | Qwen3-VL 235B | self-reported, near-saturation |
| DocVQA (original) | 0.887 ANLS | ARIAL (agentic) | ⚠ closest-style competitor |
| MP-DocVQA | 0.8458 ANLS | AVIR Framework | retrieval-augmented |
| MMLongBench-Doc | ~44.9% F1 | GPT-4o | long-context, hard |
| InfographicVQA (ICDAR'21) | 0.6120 ANLS | Applica.ai | task-specific winner |

All numbers from `lit-review-2.md` and **need verification**.

---

## Claims to avoid making in the paper

From `lit-review-2.md` — these are common-but-false claims the literature
contradicts. We should not write any of these in the paper.

- **"No prior work uses an LLM with OCR retrieval and visual tools on
  DocVQA."** This is exactly what ARIAL, AgenticOCR, and VISOR do. The
  delta we own is the RLM framing (Python REPL state, recursive
  sub-call), not "we orchestrate tools."
- **"We are the first to handle multi-page DocVQA reasoning."** Infrrd
  (2021), MP-DocVQA (2022), and many subsequent works have addressed
  multi-page reasoning explicitly.
- **"We are the first to use code for DocVQA reasoning."** ARIAL,
  MDocAgent, ORCA already incorporate code/tool execution.
- **"OCR-free is universally worse / better."** The literature shows a
  capacity-dependent trade-off — depends on model scale and document
  complexity.
- **"Recursive vision-language models for documents are unexplored."**
  Verify RVLM first (above) but we should already assume the recursive-
  visual idea is occupied at least in adjacent modalities.

---

## Conventions for adding entries

When the lit review (or future reading) surfaces a relevant paper:

1. Place it under the right section (Novelty threat / Foundational /
   Closest / Related / Baseline).
2. Always include: full citation (with arxiv ID or DOI), obsidian path
   if downloaded, connection note.
3. **Mark `(needs verification)` if the citation came from an LLM /
   lit-review agent and you haven't checked arXiv yourself.** Do not
   silently treat unverified citations as load-bearing.
4. If you haven't read it yet, mark **TBD** in the connection field —
   don't fabricate.
5. Update `docs/paper/README.md` status tracker if the entry changes
   paper positioning.
