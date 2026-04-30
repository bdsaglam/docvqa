# Run Comparison Report

## Overall

| Run                                 |    Score |   Qs | business_r |     comics | engineerin | infographi |       maps | science_pa | science_po |      slide |
|-------------------------------------|----------|------|------------|------------|------------|------------|------------|------------|------------|------------|
| t06-precise-local-c3                | 36/80    |   80 |        60% |        60% |        50% |        70% |         0% |        30% |        70% |        20% |
| gen-think-remote-c4                 | 34/80    |   80 |        50% |        50% |        50% |        60% |         0% |        30% |        50% |        50% |
| qwen-general-precise-c8             | 34/80    |   80 |        40% |        50% |        50% |        50% |        10% |        50% |        50% |        40% |
| flat-pf15-local-c3                  | 34/80    |   80 |        30% |        20% |        60% |        80% |        20% |        30% |        50% |        50% |
| flat-solo-val-local-c4              | 33/80    |   80 |        50% |        30% |        40% |        60% |        20% |        30% |        60% |        40% |
| flat-lm10-vlm03-remote-c4           | 33/80    |   80 |        30% |        50% |        60% |        60% |        10% |        30% |        40% |        50% |
| qwen-general-precise-c8-t2          | 32/80    |   80 |        40% |        30% |        50% |        70% |        10% |        40% |        40% |        40% |
| t06-precise-local-c3-t2             | 32/80    |   80 |        40% |        20% |        60% |        80% |         0% |        20% |        70% |        30% |

\* = partial (not all docs complete)

## Per-Doc Comparison

| Doc                        | Cat            | t06-precise-local- | gen-think-remote-c | qwen-general-preci | flat-pf15-local-c3 | flat-solo-val-loca | flat-lm10-vlm03-re | qwen-general-preci | t06-precise-local- |
|----------------------------|----------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| business_report_1          | business_report |                50% |                50% |                 0% |                 0% |                50% |                 0% |                50% |                50% | ***
| business_report_2          | business_report |                 0% |               100% |                 0% |                 0% |                 0% |                 0% |                 0% |                 0% | ***
| business_report_3          | business_report |               100% |               100% |               100% |               100% |               100% |               100% |               100% |               100% |
| business_report_4          | business_report |                60% |                20% |                40% |                20% |                40% |                20% |                20% |                20% | ***
| comics_1                   | comics         |                 0% |                 0% |               100% |                 0% |                 0% |                 0% |                 0% |                 0% | ***
| comics_2                   | comics         |                75% |                50% |                50% |                 0% |                50% |                50% |                50% |                25% | ***
| comics_3                   | comics         |                67% |                67% |                33% |                33% |                33% |                67% |                33% |                33% | ***
| comics_4                   | comics         |                50% |                50% |                50% |                50% |                 0% |                50% |                 0% |                 0% | ***
| engineering_drawing_1      | engineering_drawing |                67% |                33% |                33% |                33% |                 0% |                67% |                33% |                67% | ***
| engineering_drawing_2      | engineering_drawing |                50% |               100% |               100% |               100% |               100% |               100% |               100% |               100% | ***
| engineering_drawing_3      | engineering_drawing |                33% |                33% |                33% |                67% |                33% |                33% |                33% |                33% | ***
| engineering_drawing_4      | engineering_drawing |                50% |                50% |                50% |                50% |                50% |                50% |                50% |                50% |
| infographics_1             | infographics   |                50% |                50% |                50% |                50% |               100% |                50% |               100% |               100% | ***
| infographics_2             | infographics   |                75% |                62% |                50% |                88% |                50% |                62% |                62% |                75% | ***
| maps_1                     | maps           |                 0% |                 0% |                 0% |                50% |                 0% |                 0% |                 0% |                 0% | ***
| maps_2                     | maps           |                 0% |                 0% |                20% |                20% |                40% |                20% |                20% |                 0% | ***
| maps_3                     | maps           |                 0% |                 0% |                 0% |                 0% |                 0% |                 0% |                 0% |                 0% |
| science_paper_1            | science_paper  |                29% |                43% |                57% |                43% |                29% |                29% |                57% |                29% | ***
| science_paper_2            | science_paper  |                 0% |                 0% |                 0% |                 0% |                 0% |                 0% |                 0% |                 0% |
| science_paper_3            | science_paper  |                50% |                 0% |                50% |                 0% |                50% |                50% |                 0% |                 0% | ***
| science_poster_1           | science_poster |               100% |                20% |                40% |                40% |                60% |                 0% |                20% |                80% | ***
| science_poster_2           | science_poster |                40% |                80% |                60% |                60% |                60% |                80% |                60% |                60% | ***
| slide_1                    | slide          |                50% |                50% |                50% |                50% |                50% |                50% |                50% |                50% |
| slide_2                    | slide          |                 0% |                33% |                33% |                33% |                33% |                 0% |                33% |                 0% | ***
| slide_3                    | slide          |                20% |                60% |                40% |                60% |                40% |                80% |                40% |                40% | ***
