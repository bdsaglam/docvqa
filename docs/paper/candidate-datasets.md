# Candidate Datasets for Generality Experiments

Evaluation of every dataset in `tmp/datasets.md` as a potential
**additional generality benchmark** for the paper. One section per
dataset with a fit verdict and reasoning.

Grounded in the **D-006 framing** (`decisions.md`): the headline method
is OCR-free **recursive visual perception** (a code-capable LLM in a
REPL whose load-bearing tool is a `batch_look` VLM sub-call). OCR/search
is an extension, not a contribution. So a dataset's relevance now hinges
on the **visual context-budget** mechanism, *not* on OCR retrieval.

## Fit rubric

A dataset is a good generality target only if it passes all of:

1. **QA with measurable accuracy.** Rules out captioning and pretraining
   corpora — no per-question answer to score, no lift to measure.
2. **Perception-budget-bound.** The benchmark must stress a VLM's *finite
   visual context*: fine detail at high resolution, visually dense
   content, and/or many pages. This is the exact mechanism the paper
   claims (rationed recursive VLM perception). If a raw VLM already
   nails it in one forward pass, there is no lift to demonstrate.
3. **Non-circular (no document *reuse*).** The test is literal document/
   image overlap with DocVQA-2026's 8 categories (`business_report,
   comics, engineering_drawing, infographics, maps, science_paper,
   science_poster, slide`, from `VLR-CVC` = CVC/UAB), **not** shared
   authorship. CVC/UAB is central to the entire DocVQA field — MP-DocVQA
   and MMLongBench-Doc share its lineage yet are accepted because their
   documents are separate. Reuse risk is highest for a benchmark that is
   the *canonical source* of a DocVQA-2026 category (infographics →
   InfographicVQA; slide → SlideVQA). Default to verify-then-use, not
   exclude-on-lineage.
4. **Axis coverage (bonus).** Extends one of D-006's three predictions:
   model-size, **document-length** (the thin axis — only 3 benchmarks
   today), or simply adds a distinct document type for the perception
   mechanism.

**Verdict scale:** ✅ strong candidate · ⚠️ conditional/weak ·
❌ exclude — circular · ❌ off-target.

## Summary

| Dataset | QA? | Perception-bound? | Circular? | Verdict |
|---|---|---|---|---|
| InfographicVQA | ✓ | ✓ | **yes** (infographics) | ❌ circular |
| DocVQA (SP) | ✓ | weak | lineage | ❌ lean-exclude |
| ChartQA | ✓ | ✓ | skill overlap | ❌ exclude |
| TextVQA | ✓ | partial | no | ⚠️ conditional |
| MMMU | ✓ | weak | no | ⚠️ weak |
| MRAG-Bench | ✓ | no (retrieval) | no | ❌ off-target |
| VQA v2 | ✓ | no | no | ❌ off-target |
| GQA | ✓ | no | no | ❌ off-target |
| VCR | ✓ | no | no | ❌ off-target |
| NLVR2 | ✓ (NLI) | no | no | ❌ off-target |
| CLEVR | ✓ | no (synthetic) | no | ❌ off-target |
| Visual Genome | partial | no | no | ❌ off-target |
| MindBench | no (parsing) | n/a | no | ❌ off-target |
| INQUIRE | no (retrieval) | n/a | no | ❌ off-target |
| Flickr30K | no | n/a | n/a | ❌ off-target |
| MS COCO Captions | no | n/a | n/a | ❌ off-target |
| Conceptual Captions | no | n/a | n/a | ❌ off-target |
| Conceptual 12M | no | n/a | n/a | ❌ off-target |
| RedCaps | no | n/a | n/a | ❌ off-target |
| WIT | no | n/a | n/a | ❌ off-target |
| YFCC100M | no | n/a | n/a | ❌ off-target |
| COYO-700M | no | n/a | n/a | ❌ off-target |
| LAION-5B | no | n/a | n/a | ❌ off-target |
| Re-LAION-5B | no | n/a | n/a | ❌ off-target |
| DataComp-1B | no | n/a | n/a | ❌ off-target |
| MMC4 | no | n/a | n/a | ❌ off-target |

**Bottom line:** the non-circularity constraint removes every clean
document-VQA fit, because DocVQA-2026 is deliberately comprehensive
(8 doc types). Nothing in this list is a strong candidate. The best
generality evidence comes from the **related-works benchmark pool**
(next section), not from `tmp/datasets.md`.

---

## Benchmarks from the related-works library (the relevant pool)

`docs/paper/related-works/` indexes the document-VQA benchmarks the
field actually uses — a *better* pool than `tmp/datasets.md` for
generality experiments. Two reasons: several are **multi-page** (the
document-length axis, D-006 prediction 2, where we currently have only
3 benchmarks), and several are what our **direct competitors report
on**, which enables head-to-head positioning rather than standalone
lift.

Already handled: DocVQA-SP (lean-exclude, below), InfographicVQA
(circular, below), ChartQA (skill overlap, below), **MP-DocVQA +
MMLongBench-Doc** (in use — the two length-axis legs). New candidates
surfaced here:

| Benchmark | arXiv | Pages | Reuse risk | Competitor reports on it | Verdict |
|---|---|---|---|---|---|
| DUDE | 2305.08455 | multi (→70+) | low | — | ✅ strong |
| MADQA | 2603.12180 | doc collections | low (fresh docs) | MADQA (our planned baseline) | ✅ strong; regime favors OCR ext. |
| SlideVQA | 2301.04883 | multi (slides) | **high** (slide cat.) | SlideAgent | ⚠️ verify reuse |
| VisualMRC | 2101.11272 | single | low | — | ⚠️ conditional |
| ST-VQA | 1905.13648 | single (scene) | low | — | ⚠️ redundant w/ TextVQA |

### DUDE (2305.08455) — ✅ strong candidate

Document Understanding Dataset and Evaluation (Van Landeghem et al.,
ICCV 2023). Multi-domain, multi-industry documents spanning single page
to 70+ pages, with extractive + abstractive + list + unanswerable
answers. **The best generality benchmark available to us:** it hits the
document-length axis *and* the perception mechanism (diverse dense
layouts), it's a recognized standard, and its document collection is
distinct from DocVQA-2026 (shares CVC organizers, but its own docs — low
reuse risk; no single DocVQA-2026 category maps onto it). Caveat:
abstractive + list + unanswerable answers mean it needs the profile
`score_fn`, not plain ANLS (per the cross-benchmark methodology rule in
`CLAUDE.md`). Recommend adding it.

### MADQA (2603.12180) — ✅ strong + strategic, with a framing tension

Borchmann et al., *Strategic Navigation or Stochastic Search?*
**Verified from the PDF (2026-05-27):** 2,250 human-authored questions
over 800 heterogeneous PDFs, framed as document-**collection** QA
(corpus retrieval + cross-page/cross-doc multi-hop subsets). Metric is
LLM-judged **Accuracy** plus a novel **Kuiper effort-calibration**
statistic — not ANLS. Best system *Gemini 3 Pro BM25 MLLM Agent* =
82.2%; ~18% oracle gap; the paper's thesis is that **retrieval, not
reasoning, is the bottleneck**. Reuse risk **explicitly low** — the
paper advertises *fresh documents not recycled from existing
benchmarks*. Collection-scale, so strong on the context-budget
mechanism.

**Two tensions to position against, not gloss (per CLAUDE.md "surface
blind spots"):**

1. **Regime favors our OCR *extension*, not our OCR-free core.** MADQA is
   collection-scale retrieval — exactly where our own data says OCR/search
   helps (MMLongBench-Doc, MP-DocVQA long bucket) and where a pure
   visual-perception core would look retrieval-bound. Leading here with
   the OCR-free method risks a weak number; we'd lead with the extension.
2. **They already tested unconstrained RLM and found it cost-catastrophic.**
   §5: "constrained agency … avoids the catastrophic effort overhead of
   RLMs." They run RLM citing the *same* Zhang et al. 2025 paper we
   instantiate — e.g. Claude 4.5 Sonnet RLM burned 270M input tokens /
   ~$850 and still lost to its BM25-agent counterpart. This **supports**
   our "focused/constrained instantiation" framing (D-005) but also
   **pre-empts any "constraining RLM helps" claim** — that's their result.
   Our defensible delta is the *visual* sub-call specialization + the
   perception-budget hypothesis on benchmarks where visual perception
   (not collection retrieval) is the bottleneck.

Verdict: include as benchmark + baseline (D-005 stands), but lead with
the OCR extension and frame explicitly against their RLM-efficiency
result. Read the full PDF before drafting the positioning paragraph.

### SlideVQA (2301.04883) — ⚠️ candidate, verify slide-document reuse

Tanaka et al. (AAAI 2023). Multi-page slide-deck QA with multi-hop
reasoning — good length + perception fit, and SlideAgent (a related-works
competitor) reports on slide tasks, giving positioning value. **But**
DocVQA-2026 has a `slide` category and SlideVQA is *the* canonical
slide-QA dataset — highest document-reuse risk after InfographicVQA. Use
only after confirming DocVQA-2026's slide documents aren't drawn from
SlideVQA.

### VisualMRC (2101.11272) — ⚠️ conditional

Tanaka et al. (AAAI 2021). Single web-page screenshots, abstractive
machine reading comprehension. Distinct domain (web pages), low reuse
risk. But single-image → no document-length axis, and web pages are only
moderately perception-dense → weak mechanism stress. A domain-breadth
point at best, not headline.

### ST-VQA (1905.13648) — ⚠️ conditional, redundant with TextVQA

Biten et al. (ICCV 2019); the original ANLS source. Scene-text VQA on
natural images — the same out-of-domain "reads fine scene text" probe
role as TextVQA. CVC-authored but natural-scene domain (low reuse risk).
Pick TextVQA *or* ST-VQA, not both.

---

## Document-VQA benchmarks (the real candidates)

### InfographicVQA (2021) — ❌ exclude, circular

Single high-resolution infographic images; dense layout + text +
graphics. On the rubric this is a *textbook* perception-budget benchmark
(fine labels, dense charts) — it would fit the mechanism well. But
DocVQA-2026 has a dedicated `infographics` category, and both come from
the **same group (CVC/UAB, the DocVQA authors)**. Using it tests
generality on data almost certainly drawn from the same pool as the
headline benchmark. Circular. *Action: confirm whether DocVQA-2026's
infographic images are literally reused or merely same-domain before
fully discarding — but default is exclude.*

### DocVQA / SP-DocVQA (2021) — ❌ lean-exclude

The original single-page document VQA benchmark (scanned industry
documents). Two problems: (a) it's the **direct predecessor** of
DocVQA-2026 from the same lab — reviewers read it as the same benchmark
family; and (b) it's **single-page, moderate-resolution scanned text**,
so a raw VLM is only weakly perception-bound on it. Weak mechanism fit
*and* lineage overlap. Low standalone value.

### ChartQA (2022) — ❌ exclude, skill overlap

Chart images requiring value reading + reasoning — genuinely
perception-bound (small tick labels, dense plots), and a *different*
lab from CVC. But chart-reading is the explicit task in DocVQA-2026's
`science_poster` (the sample val question is literally a chart-%
comparison) and recurs in `science_paper`/`infographics`. It wouldn't
demonstrate anything beyond content the headline benchmark already
covers. Exclude on skill overlap.

### TextVQA (2019) — ⚠️ conditional

Scene text in natural photographs; reading small/oblique text. Genuinely
**disjoint domain** from DocVQA-2026 (no natural-scene category), so
non-circular, and it does stress visual perception (reading fine text
in cluttered images) — partial mechanism fit. Caveats: **single image,
no document length axis**, and it's a recognition-heavy task more than a
multi-page-budget task. Best role: an optional *out-of-domain* probe to
show the recursive-perception mechanism isn't document-specific. Keep on
the shortlist; not load-bearing.

### MMMU (2023) — ⚠️ weak

College-level multimodal reasoning (multiple-choice) across disciplines,
including charts/tables/diagrams but also chemistry structures, medical
images, music notation. Mostly disjoint from DocVQA-2026, so
non-circular. But it's a **reasoning** benchmark; visual content is
often a single moderate-resolution panel, so the perception-budget
mechanism is only lightly stressed. Defensible as a "does the scaffold
help hard multimodal reasoning generally" point, but it tests reasoning
breadth, not the paper's mechanism. Weak fit.

### MRAG-Bench (2024) — ❌ off-target

Multimodal RAG benchmark: 1,353 MCQs answered by retrieving from an
**external image corpus** (16,130 images). It's multimodal QA, but the
task is *cross-image retrieval*, not perception-budget allocation within
a document. Different mechanism. Off-target.

---

## Natural-image VQA / reasoning (wrong mechanism)

These are QA benchmarks but on **single natural images** where a raw VLM
is not perception-budget-bound — the bottleneck is recognition or
commonsense reasoning, not rationing visual context across dense/long
documents. The method's machinery (recursive crop/zoom over many dense
pages) doesn't activate, so there's no lift to show.

### VQA v2.0 (2017) — ❌ off-target
Open-ended VQA on COCO images. Recognition + commonsense, single image.

### GQA (2019) — ❌ off-target
Compositional VQA over scene graphs of natural images. Reasoning, not
perception budget.

### VCR (2019) — ❌ off-target
Commonsense reasoning + rationale on movie stills. Single image, reasoning.

### NLVR2 (2019) — ❌ off-target
Visual NLI over image pairs (true/false), not document QA.

### CLEVR (2017) — ❌ off-target
Synthetic compositional reasoning. No real-document perception at all.

### Visual Genome (2017) — ❌ off-target
Primarily a scene-graph / region-annotation resource. Region-QA exists
but the dataset is built for grounding, not perception-budget document QA.

---

## Document parsing / retrieval (not QA-accuracy)

### MindBench (2024) — ❌ off-target
Mind-map / structured-document *parsing* — output is structured
representations, not scored short answers. Document-domain but wrong task
shape; no clean accuracy lift to report.

### INQUIRE (2024) — ❌ off-target
Natural-world image **retrieval** benchmark (250 queries). Retrieval,
not document QA.

---

## Captioning & pretraining corpora (no QA target)

None of these have per-question answers to score, so there is no lift to
measure — they fail rubric criterion 1 outright. Listed for completeness:

- **Flickr30K (2014)** — captioning. ❌
- **YFCC100M (2014)** — multimedia pretraining corpus. ❌
- **MS COCO Captions (2014)** — captioning. ❌
- **Conceptual Captions (2018)** — caption-pretraining pairs. ❌
- **WIT (2021)** — Wikipedia image-text pretraining. ❌
- **Conceptual 12M (2021)** — web captioning. ❌
- **RedCaps (2021)** — user-generated captions. ❌
- **COYO-700M (2022)** — URL-text pretraining. ❌
- **LAION-5B (2022)** — CLIP-filtered pretraining. ❌
- **DataComp-1B (2023)** — curated pretraining. ❌
- **MMC4 (2023)** — interleaved multimodal pretraining docs. ❌
- **Re-LAION-5B (2024)** — safety-updated pretraining corpus. ❌

---

## Recommendation

1. **Drop** InfographicVQA, DocVQA(SP), and ChartQA from consideration —
   circular or skill-overlapping with DocVQA-2026.
2. **Keep TextVQA** as an optional out-of-domain perception probe (shows
   the mechanism generalizes beyond curated documents). Not headline.
3. **MMMU** only if a reviewer wants breadth; weak mechanism fit.
4. **Everything else is off-target.**

**The generality story must lean on off-list benchmarks.** This list
contains no long-document benchmark, which is exactly where D-006
prediction 2 (document-length axis) lives. The strongest non-circular
evidence already in flight:

- **MMLongBench-Doc** — long-doc, the context-budget leg (+16.84pp judge,
  Qwen 27B). Already run (`docs/experiments/mmlongbench-doc-qwen27b.md`).
- **MP-DocVQA** — multi-page, with the per-length-bucket breakdown
  (+13.68pp in the 11–20pp bucket). Already run
  (`docs/experiments/mp-docvqa-qwen27b.md`).

Both are distinct datasets from DocVQA-2026 (non-circular), and both
exercise the document-length axis a single-image benchmark cannot.

**Ranked plan for additional benchmarks** (drawn from the related-works
pool, not `tmp/datasets.md`):

1. **MP-DocVQA + MMLongBench-Doc** — in use, the two length-axis legs.
2. **DUDE** — best new add: multi-page, diverse, standard, low reuse risk.
3. **MADQA** — strategically strongest (competitor benchmark + planned
   baseline), gated on arXiv verification + a read.
4. **SlideVQA** — usable only after confirming no slide-document reuse.
5. **TextVQA** *or* **ST-VQA** — one optional out-of-domain perception
   probe; not headline.

`tmp/datasets.md` contributes nothing beyond the optional TextVQA probe.
