# Improvement Ideas

## BM25 Stemming Toggle
**Status**: Not started
**Priority**: Medium
**Idea**: Make BM25 stemming configurable (enable/disable English stemmer). Currently always uses `Stemmer.Stemmer("english")`. For some document types (engineering drawings, maps with proper nouns), stemming may hurt precision by collapsing distinct terms. Try both and compare per-category accuracy.

**Note**: BM25 already builds on-the-fly via `get_or_build_index()` in rlm_solver.py — no pre-build step needed. Just need to thread the stemming flag through.

**Implementation**:
- Add `stemming: bool = True` param to `build_index()`, `get_or_build_index()` in `src/docvqa/search.py`
- When `stemming=False`, pass `stemmer=None` to `bm25s.tokenize()`
- Also update search tool's query tokenization to match
- Add config option in `configs/solver/rlm.yaml`
- Use separate cache subdirs (e.g. `bm25-stemmed/`, `bm25-unstemmed/`) or skip caching since build is fast
- Test on categories with proper nouns (maps, engineering drawings) vs text-heavy docs

## Per-Question Timeout (Save Partial Results)
**Status**: Not started
**Priority**: High
**Idea**: Currently timeout is per-document (3600s). If a doc with 5 questions times out, ALL 5 get "Unknown". Instead, implement per-question timeout so completed questions are saved even if later ones time out.
**Impact**: Would have saved ~5-8 questions in Pro+Flash eval where entire docs timed out.

## Comics: Pre-index Speech Bubbles via VLM
**Status**: Not started
**Priority**: Medium
**Idea**: For comics, OCR captures image descriptions instead of speech bubble text. Pre-scan all pages with VLM to extract dialogue/narration and build a text index. This makes BM25 search work for comics.
**Impact**: 4/7 wrong comics answers were "Unknown" due to missing text in OCR.

## Maps: Atomic Waypoint Verification
**Status**: Not started
**Priority**: Medium
**Idea**: For path-following map questions, the agent should: 1) Build a graph of connections from cropped regions (not ask VLM to trace entire routes), 2) Use inspect_region for each waypoint, 3) Traverse graph in code.
Current approach: asking VLM "trace the route from X to Y" which hallucinates.
**Impact**: 5/8 wrong map answers involved path-tracing failures.

## Reduce VLM Call Overhead
**Status**: Partially addressed
**Idea**: VLM max_tokens was 4096, causing truncation. Increased to 8192 (Qwen) / 16384 (Flash). Also consider reducing unnecessary VLM calls — agent sometimes does 6 iterations of verification when 2-3 would suffice.

## Batch Question Answering
**Status**: Prototype built, needs redesign
**Priority**: High
**Idea**: Answer all questions for a document in a single RLM session so exploration is shared.

**What we tried** (`src/docvqa/solvers/batch_rlm_solver.py`):
- Pass all questions as JSON list, output a JSON dict of answers
- Higher iteration/call limits (10/50 vs 6/30)
- Result: 0/5 vs 3/5 on science_poster_1 — agent got confused juggling multiple questions simultaneously, computed wrong values, took longer (2345s vs 1408s)

**Why it failed**: The agent tries to answer all questions in every iteration, context gets cluttered, and it makes mistakes on questions it would have gotten right individually.

**Better approaches to try**:
1. **Explore-then-answer**: Two-phase approach — Phase 1: agent explores the document freely, building a structured summary (page contents, tables, figures). Phase 2: answer each question using the pre-built summary as context (no VLM calls needed, just reasoning over extracted data).
2. **Shared context, separate answers**: Run single-question RLM but pass a "document summary" built from a quick initial exploration pass. Each question still gets its own RLM session but starts with richer context.
3. **Question clustering**: Group related questions (e.g., all about the same table/figure), batch those, answer unrelated questions individually.

## Prompt Engineering Lessons
**Status**: Documented
**Finding**: Adding more prescriptive instructions (comics guidance, map strategy, interpretation rules, VLM hallucination warnings) to RLM_TASK_INSTRUCTIONS caused a REGRESSION vs the original prompt. The agent became slower and less accurate.

**Takeaway**: The original prompt works well. Don't over-specify strategy — Pro LLM already reasons well. Future prompt changes should be minimal and A/B tested on a sample before full eval.