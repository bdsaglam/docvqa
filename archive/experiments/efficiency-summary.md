# Efficiency summary — turns per question across cells

Cross-cell view of agent turn-counts (`len(trajectory)` per question)
to answer the headline question: *do the search() / cropping / tips
ablations change the number of agent turns, or only the success
rate?*

All numbers are pooled across the cell's trials (typically 8 trials
× 80 val questions = 640 questions). Source: `output/runs/<run_id>/
tasks/*/result.json`. Computed via `python3 /tmp/efficiency_full.py`
(read-only over the run dirs).

`flat_solo m=5` and `leanest-ocr-off` ran on a different host and are
**not measured locally** — they are absent from this aggregation.

## Master table (Qwen 3.5 27B, val 80q, 8 trials each)

| Cell | acc % | turns mean ± std | median | p90 | max | turns_correct | turns_wrong | wrong/correct |
|---|---|---|---|---|---|---|---|---|
| no_loop baseline | 17–21% | **0** | 0 | 0 | 0 | 0 | 0 | n/a |
| flat_solo m=10 | 41.4 | 9.92 ± 3.79 | 10 | 16 | 20 | 9.00 | 10.57 | 1.17 |
| flat_solo m=20 | 40.9 | 12.42 ± 6.29 | 11 | 20 | 30 | 10.87 | 13.49 | 1.24 |
| **flat_solo m=30 (baseline)** | **44.7** | **13.19 ± 7.85** | **11** | **26** | **40** | **11.92** | **14.22** | **1.19** |
| flat_solo m=40 | 40.8 | 13.55 ± 9.01 | 10 | 25 | 50 | 11.42 | 15.01 | 1.31 |
| flat_solo no-tips | 38.8 | 12.78 ± 7.98 | 10 | 25 | 40 | 10.50 | 14.21 | 1.35 |
| flat_solo no-cropping | 36.9 | 10.90 ± 7.32 | 8 | 21 | 40 | 8.82 | 12.14 | 1.38 |
| flat_solo no-search | 42.5 | 13.44 ± 7.81 | 11 | 25 | 40 | 11.33 | 14.99 | 1.32 |
| leanest m=25 (default) | 40.5 | 12.82 ± 7.18 | 11 | 24 | 35 | 11.54 | 13.69 | 1.19 |
| leanest m=30 | 42.5 | 13.29 ± 7.65 | 11 | 25 | 40 | 11.51 | 14.61 | 1.27 |
| leanest m=40 (peak) | 43.8 | 13.97 ± 9.08 | 11 | 27 | 50 | 11.89 | 15.59 | 1.31 |
| leanest m=50 | 41.4 | 14.21 ± 9.81 | 11 | 28 | 60 | 12.11 | 15.73 | 1.30 |

(`max` consistently lands at `m + 10` across all cells. That's not
the iteration cap leaking — `flat_solo` adds a `page_bonus` to
`max_iterations` that scales with document length, capped at +10.
Long docs (engineering_drawing, science_paper) hit that ceiling.
The mean is well below `m` for m ≥ 30, indicating that the budget
is mostly slack — agents finish before the cap on most questions.)

## Headline questions

### 1. Does search() reduce agent turns?

**No.** flat_solo m=30 = 13.19 turns; flat_solo no-search = 13.44
turns. +0.25 mean, well within noise. The agent compensates for the
missing `search()` tool by manually grepping `page_texts` with
`re.search()`, which costs the same number of turns. So search() is
not a turn-saver — it slightly stabilises wrong-trajectory length
(thrash ratio 1.19 → 1.32 without it) but doesn't shorten the loop.

### 2. Does cropping change agent turns?

**Yes — and the change is misleading.** flat_solo no-cropping =
10.90 turns vs 13.19 baseline (−2.3 mean, the largest turn drop
of any ablation). But accuracy *also* drops 7.8pp, so this is "fewer
turns, worse outcome" — when the page-only VLM can't be cropped,
the agent simply has fewer productive moves on hard visual-reasoning
questions and stops earlier. Turn-count alone is not a quality
signal; cropping enables the agent to spend more time profitably,
not less.

### 3. Do tips change agent turns?

**Net no, but they shift work to correct trajectories.** flat_solo
no-tips = 12.78 turns vs 13.19 baseline (−0.4 mean, n.s.). The
interesting cut is correct vs wrong: without tips, correct
trajectories are 1.4 turns shorter (10.50 vs 11.92) while wrong
trajectories are unchanged (14.21 vs 14.22). Thrash ratio rises
1.19 → 1.35. Read: tips appear to encourage longer, more careful
verification on the questions the agent gets right.

### 4. Is the agent budget-bound, or does the cap mostly matter for the long tail?

**Budget-bound at m=10; slack past m=30.**

| m | mean turns | p90 | hit-cap signal |
|---|---|---|---|
| 10 | 9.92 | 16 | median = 10 = m; p90 close to 2m → saturation |
| 20 | 12.42 | 20 | p90 = m → still hitting cap on long tail |
| 30 | 13.19 | 26 | mostly slack |
| 40 | 13.55 | 25 | slack — extra 10 turns barely used |

Going m=30 → m=40 only adds +0.36 turns to the mean — the extra
budget is mostly unused, but accuracy *drops* 3.9pp anyway. The
long-tail trajectories that the bigger budget enables are net
harmful (drift, context dilution). The tightest read of "budget
sweet spot" matches the accuracy peak: m=30 for flat_solo, m=40 for
leanest, both close to where the median saturates the cap.

### 5. Leanest vs flat_solo on the same budget — do they differ in turn-count?

**No, they're nearly identical.** At m=30, flat_solo = 13.19 and
leanest = 13.29 turns/question. The OCR channel doesn't save the
agent turns — it just shifts work from `look()` calls to BM25 +
`page_texts` lookups. Same volume of work in different channels.
Confirms that OCR's contribution is variance reduction (std 2.81 vs
~5.0pp) and small accuracy lift, not efficiency lift.

### 6. Wrong-answer thrash factor (turns_wrong / turns_correct)

**Wrong > correct everywhere.** Across every measured cell, wrong
answers consume more turns than correct ones — the rule, not the
exception. The ratio sits between 1.17 and 1.38 with notable cuts:

| Cell | thrash ratio | reading |
|---|---|---|
| flat_solo m=10 | 1.17 | tightest — budget-bound, no room to thrash |
| flat_solo m=30 (baseline) | 1.19 | low; tips + full tools keep wrong loops short |
| leanest m=25 | 1.19 | matches flat_solo on same shape |
| flat_solo m=20 | 1.24 | rising as cap loosens |
| leanest m=30 | 1.27 | rising as cap loosens |
| leanest m=40/50 | 1.31 / 1.30 | flat near peak |
| flat_solo m=40 | 1.31 | rises with budget |
| flat_solo no-search | 1.32 | manual grep loops widen wrong |
| flat_solo no-tips | 1.35 | tips suppress wrong-loop length |
| flat_solo no-cropping | 1.38 | widest — agent stuck without `look()` crops |

Two patterns:

- **More budget → more thrash on wrong answers**, hardly any change on
  correct ones. Bigger budgets mostly buy longer wrong trajectories.
- **Removing scaffold components (tips, cropping, search) widens the
  thrash ratio.** Each scaffold piece keeps wrong-answer loops
  shorter relative to correct ones.

This thrash gap (~3 turns × ~50% wrong rate × 80 questions × 8
trials) is also where most wall-clock time bleeds — the agent is
spending its budget hardest on the questions it ends up getting
wrong.

## Stories the turn-count tells that the accuracy didn't

- **No-cropping looks "efficient" by turn count (−2.3 turns vs
  baseline) but is the worst single ablation for accuracy (−7.8pp).**
  The shortened trajectories are truncated work, not faster work.
- **No-search is statistically tied with baseline on accuracy
  (−2.2pp, n.s.) and on turn-count (+0.25 turns).** Confirms the
  paper framing that BM25 is largely redundant — neither dimension
  shows a meaningful gap.
- **m=40 vs m=30 in flat_solo: +0.36 turns mean, −3.9pp accuracy.**
  More budget doesn't get used much *and* hurts accuracy. The
  pathological-trajectory story (drift past the peak) is consistent
  on both axes.
- **Leanest matches flat_solo on turns at every comparable budget**
  (13.29 vs 13.19 at m=30, etc.) — the OCR channel does the same
  volume of work, just in different tools. OCR's contribution is
  variance reduction, not efficiency.

## Source / replication

```bash
python3 /tmp/efficiency_full.py    # writes /tmp/efficiency_full.json
```

Each per-experiment file in this directory now carries an
"Efficiency (turns per question)" section with the per-cell numbers
inline alongside that experiment's accuracy story.
