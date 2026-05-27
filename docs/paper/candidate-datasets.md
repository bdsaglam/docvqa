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
3. **Non-circular.** Not drawn from the same source as DocVQA-2026's 8
   categories (`business_report, comics, engineering_drawing,
   infographics, maps, science_paper, science_poster, slide`, from
   `VLR-CVC` = CVC/UAB). Using same-source data measures generality on
   the headline benchmark itself.
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
generality evidence comes from **off-list long-document benchmarks**
(see final section).

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
Candidates to round out coverage if more breadth is wanted, all
non-circular and on-mechanism: **DUDE**, **SlideVQA**, **TAT-DQA**.
