# rvlm — Reasoner × Perceiver 3×3 matrix (val, Qwen 3.5 {4B, 9B, 27B})

> **STATUS: LIVE.** Completed cells are filled; `TBD`/`—` remain for the
> report-writing agent to fill (or mark not-run) from the run_ids in the
> *Cell data sources* table below as trials land. Keep values in sync with the
> per-cell by-model files.

Full factorial of the **proposed method `rvlm`** (OCR-free RLM + recursive VLM
`batch_look`) across its two model axes, holding everything else fixed
(DocVQA-2026 val, 25 docs / 80 Q, `enable_thinking=false`, current code):

- **Reasoner (agent LLM)** — the RLM that writes Python and drives the REPL /
  `batch_look` loop. **Rows.**
- **Perceiver (VLM)** — the model that answers each `batch_look` image query.
  **Columns.**

Each axis ∈ {Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B}. Diagonal = homogeneous
(same model both roles). Cell value = `rvlm` val score, **mean ± std (n)**.

## Matrix — mean ± std (n)

| Reasoner ↓ \ Perceiver → | 4B-VLM | 9B-VLM | 27B-VLM |
|---|---|---|---|
| **4B-LM**  | 14.22 ± 3.83 (n=8)ᵐ | — not run | 21.09 ± 3.16 (n=8)ᵐ |
| **9B-LM**  | — not run | 18.91 ± 3.81 (n=8)ᵐ | 25.31 ± 4.16 (n=8)ᵐ |
| **27B-LM** | 32.81 ± 3.13 (n=4) | *in progress (n=4)* | **41.88 ± 5.79 (n=8)** |

ᵐ = `rvlm-minimal` prompt-era variant (see caveat ¹ below); unmarked cells use
current `rvlm`. Bold = headline. Reading (where filled): both axes rise
monotonically — across row 27B (perception axis) 32.81 → [27B/9B] → 41.88, and
down column 27B-VLM (reasoning axis) 21.09 → 25.31 → 41.88.

## How to read the axes

- **Across a row** (fix Reasoner, scale Perceiver 4B→27B) = the **perception
  axis**: how much a better VLM lifts the same reasoner. A steep rise ⇒
  perception-budget-bound (D-006, prediction 1).
- **Down a column** (fix Perceiver, scale Reasoner 4B→27B) = the **reasoning
  axis**: how much a better reasoner helps at fixed perception.
- **Diagonal** = homogeneous cells (the v1 ladder). **Below/left of diagonal** =
  strong-reasoner / weak-perceiver corner; **above/right** = weak-reasoner /
  strong-perceiver corner (the perception-budget-lift corner).

## Cell data sources & status

Run-id globs the agent should read for each cell, with current status. **Score
each cell from its own runs; `avg@1` = mean ± std over trials at full 80-Q
coverage** (drop incomplete trials or resume to fill).

| Cell (LM / VLM) | run_id glob | n target | solver variant | status |
|---|---|---|---|---|
| 4B / 4B   | `rvlm-minimal-3_5-4b-val-t*`            | 8 | rvlm-minimal¹ | have |
| 4B / 9B   | *(none)*                                | — | —             | **not run** |
| 4B / 27B  | `rvlm-minimal-4b-llm-27b-vlm-val-t*`    | 8 | rvlm-minimal¹ | have |
| 9B / 4B   | *(none)*                                | — | —             | **not run** |
| 9B / 9B   | `rvlm-minimal-3_5-9b-val-t*`            | 8 | rvlm-minimal¹ | have |
| 9B / 27B  | `rvlm-minimal-9b-llm-27b-vlm-val-t*`    | 8 | rvlm-minimal¹ | have |
| 27B / 4B  | `rvlm-27b-llm-4b-vlm-val-t*`            | 4 | rvlm (current) | have |
| 27B / 9B  | `rvlm-27b-llm-9b-vlm-val-t*`            | 4 | rvlm (current) | **in progress** |
| 27B / 27B | headline `rvlm` re-run (`rvlm-cmp-val-t*`) | 8 | rvlm (current) | have (41.88 ± 5.79) |

¹ **Solver-variant caveat (must be stated in any writeup).** The small-model
cells were run under the `rvlm-minimal` run-id era — an **earlier prompt-scrub
variant** of the same `rvlm_solver` (`docvqa.solvers.rvlm_solver`), not the
current canonical prompt. The 27B-row cells (27B/4B, 27B/9B, 27B/27B) use the
**current `rvlm`**. So the matrix mixes two prompt eras; the 27B-anchored ladders
(row 27B, column 27B) are internally consistent (current `rvlm`), while the
4B/9B-reasoner cells are minimal-variant. Flag this wherever the matrix is cited;
re-running the minimal cells under current `rvlm` would make it fully uniform.

## Notes

- **n is not uniform** (27B-row cells are n=4, others n=8). State per-cell n.
- **Known degenerate-loop drop:** `science_paper_1` (and sometimes `maps_2`,
  `comics_2`) can fail under `rvlm` (no exec-timeout), worse with a weak VLM →
  some cells score over 24/25 docs. Note the doc count per cell.
- Per-cell detail lives in the by-model files: `qwen-3_5-4b.md`,
  `qwen-3_5-9b.md`, and the 27B per-solver file `rvlm-qwen-3_5-27b.md`. This
  matrix is the cross-axis synthesis view; numbers here must match those.
