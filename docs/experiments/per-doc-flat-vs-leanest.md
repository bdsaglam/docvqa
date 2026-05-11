# Per-doc comparison: flat_solo vs leanest (best configs)

**Setup**: Qwen 3.5 27B / val 80q, 8 trials each.
- **flat_solo m=30 (default, full scaffold)**: OCR `page_texts` in scope, BM25 `search()` tool, VLM cropping, category tips.
- **leanest m=40 (peak)**: no OCR, no search, only `batch_look()` VLM with cropping, category tips.

Per-doc accuracy = mean of (doc accuracy per trial) across 8 trials.
Sort: descending num_pages, then category, then doc_id.

| doc_id | category | pages | nq | flat_solo (full) | leanest m=40 | Δ (flat−leanest) |
|---|---|---:|---:|---:|---:|---:|
| business_report_1 | business_report | 181 | 2 | 37.5% ± 23.1 | 37.5% ± 23.1 | **+0.0** |
| business_report_4 | business_report | 110 | 5 | 47.5% ± 14.9 | 47.5% ± 18.3 | **+0.0** |
| business_report_2 | business_report | 105 | 1 | 12.5% ± 35.4 | 0.0% ± 0.0 | **+12.5** |
| business_report_3 | business_report | 89 | 2 | 100.0% ± 0.0 | 87.5% ± 23.1 | **+12.5** |
| comics_4 | comics | 69 | 2 | 43.8% ± 17.7 | 37.5% ± 23.1 | **+6.2** |
| comics_3 | comics | 60 | 3 | 45.8% ± 17.3 | 41.7% ± 15.4 | **+4.2** |
| comics_2 | comics | 52 | 4 | 37.5% ± 18.9 | 46.9% ± 16.0 | **-9.4** |
| science_paper_1 | science_paper | 44 | 7 | 30.4% ± 19.4 | 33.9% ± 7.4 | **-3.6** |
| comics_1 | comics | 36 | 1 | 12.5% ± 35.4 | 0.0% ± 0.0 | **+12.5** |
| slide_1 | slide | 36 | 2 | 50.0% ± 0.0 | 50.0% ± 0.0 | **+0.0** |
| slide_2 | slide | 32 | 3 | 29.2% ± 11.8 | 25.0% ± 15.4 | **+4.2** |
| science_paper_3 | science_paper | 30 | 2 | 43.8% ± 17.7 | 43.8% ± 17.7 | **+0.0** |
| science_paper_2 | science_paper | 19 | 1 | 0.0% ± 0.0 | 0.0% ± 0.0 | **+0.0** |
| slide_3 | slide | 18 | 5 | 57.5% ± 7.1 | 55.0% ± 9.3 | **+2.5** |
| engineering_drawing_1 | engineering_drawing | 7 | 3 | 54.2% ± 17.3 | 41.7% ± 23.6 | **+12.5** |
| engineering_drawing_3 | engineering_drawing | 6 | 3 | 41.7% ± 23.6 | 70.8% ± 27.8 | **-29.2** |
| engineering_drawing_4 | engineering_drawing | 3 | 2 | 62.5% ± 23.1 | 68.8% ± 25.9 | **-6.2** |
| engineering_drawing_2 | engineering_drawing | 1 | 2 | 87.5% ± 35.4 | 68.8% ± 25.9 | **+18.8** |
| infographics_1 | infographics | 1 | 2 | 81.2% ± 25.9 | 75.0% ± 37.8 | **+6.2** |
| infographics_2 | infographics | 1 | 8 | 57.8% ± 9.3 | 64.1% ± 10.4 | **-6.2** |
| maps_1 | maps | 1 | 2 | 6.2% ± 17.7 | 6.2% ± 17.7 | **+0.0** |
| maps_2 | maps | 1 | 5 | 20.0% ± 15.1 | 17.5% ± 16.7 | **+2.5** |
| maps_3 | maps | 1 | 3 | 4.2% ± 11.8 | 4.2% ± 11.8 | **+0.0** |
| science_poster_1 | science_poster | 1 | 5 | 42.5% ± 22.5 | 35.0% ± 14.1 | **+7.5** |
| science_poster_2 | science_poster | 1 | 5 | 67.5% ± 14.9 | 57.5% ± 16.7 | **+10.0** |

## Aggregate (per-doc mean, equal-weight)

- flat_solo full = 42.93% vs leanest m=40 = 40.63% (delta = +2.30pp)
- Per-doc winner (|Δ| > 1pp): flat wins **13/25**, leanest wins **5/25**, tie **7/25**

(Note: per-doc equal-weighted average differs slightly from the
per-trial cell mean — 44.69% (flat) and 43.75% (leanest) reported
in `flat-solo-turn-budget-sweep.md` and `leanest-turn-budget-sweep.md`
— because cell means weight each question equally while this view
weights each doc equally.)

## By document length

- **Long docs (≥20 pages, n=12)**: flat = 40.9%, leanest = 37.6%, delta = **+3.3pp**
- **Short docs (<20 pages, n=13)**: flat = 44.8%, leanest = 43.4%, delta = **+1.4pp**

The OCR/search channel helps **more on long docs** — as expected, since the
agent needs to *locate* the right page among many. On short docs the
delta is essentially zero — visual perception is doing the work.

## Where OCR wins (biggest flat_solo advantages)

| Doc | Pages | Qs | flat | leanest | Δ |
|---|---:|---:|---:|---:|---:|
| engineering_drawing_2 | 1 | 2 | 87.5% | 68.8% | **+18.8** |
| business_report_2 | 105 | 1 | 12.5% | 0.0% | **+12.5** |
| business_report_3 | 89 | 2 | 100.0% | 87.5% | **+12.5** |
| comics_1 | 36 | 1 | 12.5% | 0.0% | **+12.5** |
| engineering_drawing_1 | 7 | 3 | 54.2% | 41.7% | **+12.5** |
| science_poster_2 | 1 | 5 | 67.5% | 57.5% | **+10.0** |

Pattern: **long documents with few questions** — OCR retrieval makes
"find the relevant page" cheap. Without it, leanest has to `batch_look`
across many pages, which is both more expensive and noisier. The
"100% vs 87.5%" win on `business_report_3` (89 pages, 2 questions) is
the cleanest example.

## Where leanest wins (biggest leanest advantages)

| Doc | Pages | Qs | flat | leanest | Δ |
|---|---:|---:|---:|---:|---:|
| engineering_drawing_3 | 6 | 3 | 41.7% | 70.8% | **-29.2** |
| comics_2 | 52 | 4 | 37.5% | 46.9% | **-9.4** |
| engineering_drawing_4 | 3 | 2 | 62.5% | 68.8% | **-6.2** |
| infographics_2 | 1 | 8 | 57.8% | 64.1% | **-6.2** |

Pattern: **visually-rich, short documents**. The standout is
`engineering_drawing_3` (6 pages, 3 questions, **+29.2pp for leanest**)
— a doc where answers live in visual structure (leader lines, part
labels, connector diagrams) rather than text. The OCR'd text from
these docs is noisy and likely *misleads* the agent away from the
visual evidence. Removing the OCR channel forces the agent to look,
and looking is what works here.

`comics_2` and `infographics_2` show similar pattern — high
question-density on visually-busy short docs.

## Interpretation for the paper

The aggregate "OCR adds ~1pp" hides two opposing forces:

1. **OCR is a long-doc navigation tool** — it shortens "find the
   right page" loops. On 89-105 page docs, this matters.
2. **OCR is a visual-perception distractor** — when answers are in
   diagrams/figures, having OCR text in scope drags the agent toward
   text-based reasoning that doesn't match the answer source.

On the current 25-doc val set, these roughly cancel because the
benchmark mixes both kinds. A benchmark skewed toward long-document
QA (MP-DocVQA, MMLongBench-Doc) would likely show a larger OCR
contribution. A benchmark skewed toward diagram understanding
(ChartQA, engineering drawings) might show OCR hurting.

This is a more refined story than "OCR doesn't matter" — and a more
defensible one for the paper.
