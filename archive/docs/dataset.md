# DocVQA 2026 Dataset

ICDAR 2026 competition dataset for Document Visual Question Answering.

- **Source**: `VLR-CVC/DocVQA-2026` (HuggingFace)
- **Splits**: `val` (25 docs, 80 questions), `test` (48 docs, 160 questions)
- **No train split** — zero-shot / few-shot only
- **Metric**: ANLS (Average Normalized Levenshtein Similarity, threshold 0.80)
- **Categories**: 8 document types, balanced at 10 questions per category per split

## Split Overview

| | Val | Test |
|---|---|---|
| Documents | 25 | 48 |
| Questions | 80 | 160 |
| Pages (total) | 905 | 1,598 |
| Avg pages/doc | 36.2 | 33.3 |
| Questions/doc | 1–8 (median 3) | 1–7 (median 3) |
| Unanswerable | 5/80 (6.2%) | — (answers hidden) |

### Document IDs by Split

| Category | Val | Test |
|---|---|---|
| business_report | `_1, _2, _3, _4` | `_4, _7, _9, _10, _11, _14` |
| comics | `_1, _2, _3, _4` | `_1, _4, _10, _13, _15` |
| engineering_drawing | `_1, _2, _3, _4` | `_1, _6, _8, _9, _10, _11, _12` |
| infographics | `_1, _2` | `_2, _5, _6, _7, _8` |
| maps | `_1, _2, _3` | `_3, _5, _6, _7, _8, _9, _10, _12` |
| science_paper | `_1, _2, _3` | `_4, _5, _6, _7, _8, _9` |
| science_poster | `_1, _2` | `_3, _4, _5, _6, _7` |
| slide | `_1, _2, _3` | `_3, _4, _5, _6, _7, _8` |

Note: some doc IDs appear in both splits (e.g., `comics_1`, `engineering_drawing_1`) — the images overlap but questions differ.

## Category Statistics

### Val Set

| Category | Docs | Pages | Questions | Avg Pg/Doc | Avg Q/Doc |
|---|---|---|---|---|---|
| business_report | 4 | 485 | 10 | 121.2 | 2.5 |
| comics | 4 | 217 | 10 | 54.2 | 2.5 |
| engineering_drawing | 4 | 17 | 10 | 4.2 | 2.5 |
| infographics | 2 | 2 | 10 | 1.0 | 5.0 |
| maps | 3 | 3 | 10 | 1.0 | 3.3 |
| science_paper | 3 | 93 | 10 | 3.3 | 3.3 |
| science_poster | 2 | 2 | 10 | 1.0 | 5.0 |
| slide | 3 | 86 | 10 | 2.8 | 3.3 |
| **Total** | **25** | **905** | **80** | **36.2** | **3.2** |

### Test Set

| Category | Docs | Pages | Questions | Avg Pg/Doc | Avg Q/Doc |
|---|---|---|---|---|---|
| business_report | 6 | 837 | 20 | 139.5 | 3.3 |
| comics | 5 | 327 | 20 | 65.4 | 4.0 |
| engineering_drawing | 7 | 30 | 20 | 4.3 | 2.9 |
| infographics | 5 | 5 | 20 | 1.0 | 4.0 |
| maps | 8 | 8 | 20 | 1.0 | 2.5 |
| science_paper | 6 | 138 | 20 | 23.0 | 3.3 |
| science_poster | 5 | 5 | 20 | 1.0 | 4.0 |
| slide | 6 | 248 | 20 | 41.3 | 3.3 |
| **Total** | **48** | **1,598** | **160** | **33.3** | **3.3** |

## Page-Level Detail (Val)

| Category | Doc ID | Pages | OCR Chars | Chars/Page |
|---|---|---|---|---|
| business_report | business_report_1 | 181 | 895,908 | 4,951 |
| business_report | business_report_2 | 105 | 317,036 | 3,019 |
| business_report | business_report_3 | 89 | 406,846 | 4,569 |
| business_report | business_report_4 | 110 | 468,936 | 4,263 |
| comics | comics_1 | 36 | 232,342 | 6,454 |
| comics | comics_2 | 52 | 724,875 | 13,940 |
| comics | comics_3 | 60 | 980,011 | 16,334 |
| comics | comics_4 | 69 | 461,135 | 6,683 |
| engineering_drawing | engineering_drawing_1 | 7 | 9,005 | 1,286 |
| engineering_drawing | engineering_drawing_2 | 1 | 2,993 | 2,993 |
| engineering_drawing | engineering_drawing_3 | 6 | 99,708 | 16,618 |
| engineering_drawing | engineering_drawing_4 | 3 | 19,251 | 6,417 |
| infographics | infographics_1 | 1 | 3,724 | 3,724 |
| infographics | infographics_2 | 1 | 7,227 | 7,227 |
| maps | maps_1 | 1 | 3,033 | 3,033 |
| maps | maps_2 | 1 | 2,918 | 2,918 |
| maps | maps_3 | 1 | 2,820 | 2,820 |
| science_paper | science_paper_1 | 44 | 271,639 | 6,174 |
| science_paper | science_paper_2 | 19 | 87,186 | 4,589 |
| science_paper | science_paper_3 | 30 | 120,080 | 4,003 |
| science_poster | science_poster_1 | 1 | 44,214 | 44,214 |
| science_poster | science_poster_2 | 1 | 26,899 | 26,899 |
| slide | slide_1 | 36 | 157,090 | 4,364 |
| slide | slide_2 | 32 | 175,520 | 5,485 |
| slide | slide_3 | 18 | 29,287 | 1,627 |

## Page-Level Detail (Test)

| Category | Doc ID | Pages | OCR Chars | Chars/Page |
|---|---|---|---|---|
| business_report | business_report_4 | 281 | 1,442,861 | 5,133 |
| business_report | business_report_7 | 264 | 1,194,626 | 4,525 |
| business_report | business_report_9 | 75 | 712,730 | 9,503 |
| business_report | business_report_10 | 52 | 483,608 | 9,300 |
| business_report | business_report_11 | 25 | 102,862 | 4,114 |
| business_report | business_report_14 | 140 | 593,768 | 4,241 |
| comics | comics_1 | 53 | 441,452 | 8,329 |
| comics | comics_4 | 67 | 222,420 | 3,320 |
| comics | comics_10 | 68 | 552,335 | 8,123 |
| comics | comics_13 | 68 | 109,631 | 1,612 |
| comics | comics_15 | 71 | 366,638 | 5,164 |
| engineering_drawing | engineering_drawing_1 | 9 | 136,999 | 15,222 |
| engineering_drawing | engineering_drawing_6 | 1 | 3,592 | 3,592 |
| engineering_drawing | engineering_drawing_8 | 2 | 6,510 | 3,255 |
| engineering_drawing | engineering_drawing_9 | 2 | 6,041 | 3,021 |
| engineering_drawing | engineering_drawing_10 | 6 | 18,511 | 3,085 |
| engineering_drawing | engineering_drawing_11 | 5 | 18,692 | 3,738 |
| engineering_drawing | engineering_drawing_12 | 5 | 26,365 | 5,273 |
| infographics | infographics_2 | 1 | 9,675 | 9,675 |
| infographics | infographics_5 | 1 | 1,391 | 1,391 |
| infographics | infographics_6 | 1 | 4,269 | 4,269 |
| infographics | infographics_7 | 1 | 5,303 | 5,303 |
| infographics | infographics_8 | 1 | 3,787 | 3,787 |
| maps | maps_3 | 1 | 3,390 | 3,390 |
| maps | maps_5 | 1 | 0 | 0 |
| maps | maps_6 | 1 | 0 | 0 |
| maps | maps_7 | 1 | 3,396 | 3,396 |
| maps | maps_8 | 1 | 0 | 0 |
| maps | maps_9 | 1 | 3,019 | 3,019 |
| maps | maps_10 | 1 | 3,181 | 3,181 |
| maps | maps_12 | 1 | 3,047 | 3,047 |
| science_paper | science_paper_4 | 10 | 39,060 | 3,906 |
| science_paper | science_paper_5 | 5 | 25,740 | 5,148 |
| science_paper | science_paper_6 | 69 | 307,278 | 4,453 |
| science_paper | science_paper_7 | 8 | 52,044 | 6,506 |
| science_paper | science_paper_8 | 16 | 97,699 | 6,106 |
| science_paper | science_paper_9 | 30 | 145,594 | 4,853 |
| science_poster | science_poster_3 | 1 | 9,597 | 9,597 |
| science_poster | science_poster_4 | 1 | 0 | 0 |
| science_poster | science_poster_5 | 1 | 0 | 0 |
| science_poster | science_poster_6 | 1 | 0 | 0 |
| science_poster | science_poster_7 | 1 | 0 | 0 |
| slide | slide_3 | 41 | 68,430 | 1,669 |
| slide | slide_4 | 62 | 200,343 | 3,231 |
| slide | slide_5 | 36 | 111,371 | 3,094 |
| slide | slide_6 | 40 | 16,335 | 408 |
| slide | slide_7 | 35 | 17,687 | 505 |
| slide | slide_8 | 34 | 13,987 | 411 |

## Image Resolution

### Val Set

| Category | Pages | Avg Px/Page | Est. Resolution |
|---|---|---|---|
| business_report | 485 | 1,026,864 | ~1013×1013 |
| comics | 217 | 1,879,292 | ~1371×1371 |
| engineering_drawing | 17 | 8,700,384 | ~2950×2950 |
| infographics | 2 | 1,084,600 | ~1041×1041 |
| maps | 3 | 10,675,759 | ~3267×3267 |
| science_paper | 93 | 945,513 | ~972×972 |
| science_poster | 2 | 10,220,800 | ~3197×3197 |
| slide | 86 | 895,640 | ~946×946 |

- **Largest**: maps_2 page_0 — 3732×3732 (13.9M px)
- **Smallest**: comics_1 page_31 — 816×816 (0.67M px)

### Test Set

| Category | Pages | Avg Px/Page | Est. Resolution |
|---|---|---|---|
| business_report | 837 | 9,927,908 | ~3151×3151 |
| comics | 327 | 1,776,172 | ~1333×1333 |
| engineering_drawing | 30 | 8,700,384 | ~2950×2950 |
| infographics | 5 | 3,718,606 | ~1928×1928 |
| maps | 8 | 65,081,730 | ~8067×8067 |
| science_paper | 138 | 8,436,418 | ~2905×2905 |
| science_poster | 5 | 116,503,390 | ~10794×10794 |
| slide | 248 | 8,858,885 | ~2976×2976 |

- **Largest**: maps_5 page_0 — 15695×15695 (246M px!)
- **Smallest**: comics_4 page_33 — 815×815 (0.66M px)

> Test set images are significantly larger (avg 8.5M px vs 1.4M px in val). Business reports, maps, science papers, posters, and slides are all much higher resolution in test. Some images exceed 246M pixels — requires `Image.MAX_IMAGE_PIXELS = 500_000_000`.

## OCR Coverage

OCR processed via `docling-serve` (Docling pipeline with GPU). Stored as markdown per page: `data/{split}/ocr/{doc_id}/page_N.md`.

| Split | Docs with OCR | Total OCR Chars | Avg Chars/Page |
|---|---|---|---|
| Val | 25/25 (100%) | 5,549,683 | 6,132 |
| Test | 48/48 (100%) | 7,585,264 | 4,747 |

### OCR Gaps (Test)

Some test docs have zero or very low OCR output — likely image-heavy content that OCR cannot extract:

- **maps_5, maps_6, maps_8**: 0 chars (map images, text-only via legends)
- **science_poster_4, _5, _6, _7**: 0 chars (visual posters, no extractable text)
- **slide_6, _7, _8**: ~400-500 chars/page (minimal text slides)

## Question Analysis

### By Starting Word

| Starter | Val | Test | Combined |
|---|---|---|---|
| what | 26 | 37 | 63 |
| which | 13 | 15 | 28 |
| how | 11 | 22 | 33 |
| in | 10 | 29 | 39 |
| on | 3 | 5 | 8 |
| by | 2 | 1 | 3 |
| if | 2 | 3 | 5 |
| assuming | 2 | 0 | 2 |
| starting | 0 | 5 | 5 |
| identify | 0 | 4 | 4 |
| locate | 0 | 4 | 4 |

Most questions require extraction (what/which/how) or spatial reasoning (in/on/locate/starting). Multi-hop questions (assuming, if...then) are common.

### Answer Characteristics (Val only)

- **Answer length**: 1–304 characters, mean 14.4
- **Unanswerable**: 5/80 (6.2%) — answer is "Unknown" or similar
- Answers are typically short entity names, numbers, or short phrases

### Sample Questions by Category

**business_report** — multi-page navigation, financial data extraction:
> "In Fiscal 2025, by how many dollars does NVIDIA's TSR value exceed the Nasdaq-100 Index TSR value?"

**comics** — multi-page narrative comprehension:
> "How many times do people get in the head in Nyoka and the Witch Doctor's Madness?"

**engineering_drawing** — technical diagram reading:
> "How many matrix assy are used in the fixed memory module assembly?"

**infographics** — visual data lookup, multi-step reasoning:
> "Assuming you get 15 USD/hour in Angola what's the number of coffees you could buy for the price of an ipod?"

**maps** — spatial navigation, legend reading:
> "If I'm standing at the Pantheon and looking toward the Colosseum, which hill will I see most clearly to the right of it?"

**science_paper** — table/figure extraction, citation tracking:
> "How much data (in Millions) do they use to ablate the Perception Encoder?"

**science_poster** — chart reading, visual comparison:
> "What is the percentage score improvement from Baseline to TexTok in rFID-50k for the ImageNet 512x512 case with 128 tokens?"

**slide** — presentation data extraction:
> "Assuming Q3 revenue is distributed by customer in the same way as Q3 backlog, and costumers are equally distributed between them, what would be the Q3 revenue of the second largest customer?"

## Dataset Schema

Each row in the dataset:

| Field | Type | Description |
|---|---|---|
| `doc_id` | str | e.g., `comics_2` |
| `doc_category` | str | One of 8 categories |
| `preview` | Image | Thumbnail/preview |
| `document` | list[Image] | All pages as PIL Images |
| `questions` | dict | `{question_id: list[str], question: list[str]}` |
| `answers` | dict | `{question_id: list[str], answer: list[str]}` — absent in test |

## Key Observations

1. **Category balance**: Exactly 10 questions per category in val, 20 per category in test — evenly distributed despite varying doc counts.
2. **Page count skew**: Business reports (avg 121 pg) and comics (avg 54 pg) dominate page count; maps, infographics, and posters are single-page.
3. **Test images are much larger**: avg 8.5M px vs 1.4M px in val — some test images exceed 246M pixels.
4. **OCR reliability varies**: Maps and posters often yield zero OCR text. Comics OCR includes speech bubbles but with errors.
5. **Multi-hop questions**: ~20% require combining information from multiple pages or performing arithmetic.
6. **Doc overlap**: Some doc IDs appear in both splits (e.g., `comics_1`, `engineering_drawing_1`) with different questions — not a strict partition by document.

## Single page documents

| infographics | infographics_1 | 1 | 
| infographics | infographics_2 | 1 | 
| maps | maps_1 | 1 | 
| maps | maps_2 | 1 | 
| maps | maps_3 | 1 | 
| science_poster | science_poster_1 | 1 |
| science_poster | science_poster_2 | 1 |
| engineering_drawing | engineering_drawing_2 | 1 |