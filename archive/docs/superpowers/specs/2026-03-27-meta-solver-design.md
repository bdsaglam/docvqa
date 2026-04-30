# Meta-Solver Design Document

**Date:** 2026-03-27
**Author:** Claude
**Status:** Draft

## Overview

A new solver `MetaSolver` that acts as an intelligent orchestrator. It first explores the document using RLM with code execution, then dynamically delegates question-solving to sub-agent solver instances (default: FlatBatchProgram). The main agent controls what to delegate, what context to provide, and accumulates knowledge from sub-agent results.

## Motivation

Current solvers have limitations:
- **rlm_solver**: Sequential, one question at a time, slow
- **flat_batch_solver**: All questions in one RLM session, no delegation, can timeout on complex docs
- **skilled_solver**: Like flat_batch but with skills, still no delegation

Meta-solver enables:
1. **Exploration first** — Main agent understands the document before delegating
2. **Dynamic delegation** — Delegate only when beneficial (complex visual tasks, grouped questions)
3. **Knowledge accumulation** — Learn from sub-agents, reuse for subsequent questions
4. **Flexible sub-agents** — Use any solver as sub-agent (default: flat_batch)

## Architecture

```
MetaSolver
├── Main RLM Agent (explorer + orchestrator)
│   ├── Tools:
│   │   ├── look() / batch_look() — direct VLM calls
│   │   ├── search() — BM25 search
│   │   └── subagent(pages, questions, knowledge, max_iterations) → {answers, knowledge}
│   └── Code execution sandbox
│
└── Sub-Agent Solver Factory
    └── Creates solver instances (default: FlatBatchProgram)
        ├── Receives filtered document (page subset)
        ├── Receives questions + optional knowledge
        ├── Returns answers + extracted knowledge
        └── Configurable budget (max_iterations)
```

## Key Components

### 1. Subagent Tool

```python
def subagent(
    pages: list[int] | None = None,  # None = all pages
    questions: list[str] | str,      # question(s) to solve
    knowledge: dict | None = None,   # context/hints from main agent
    max_iterations: int | None = None,  # budget override
) -> dict:
    """
    Solve questions using a sub-agent solver.

    Returns: {
        "answers": {"q1": "answer1", ...},
        "knowledge": {"extracted_table": {...}, "labels": [...], ...}
    }
    """
```

### 2. Main Agent Prompt

```python
TASK_INSTRUCTIONS = """
You are a Document Visual Question Answering orchestrator. You explore documents,
understand their structure, then delegate question-solving to sub-agents when beneficial.

## DATA
- `questions`: JSON list of {question_id, question} dicts. You must answer ALL of them.
- `page_texts`: OCR-extracted text per page. May be inaccurate — verify critical values visually.
- `pages`: list of page images (PIL Images) (0-indexed). Pass to `look()`.

## TOOLS
- search(query, k=5) -> list[dict]: BM25 search over OCR text.
- look(image, query) -> str: Send any PIL Image to the VLM with a query.
- batch_look(requests) -> list[str]: Parallel VLM calls.
- subagent(pages, questions, knowledge=None, max_iterations=None) -> dict:
  Delegate solving to a sub-agent. Returns {answers: {...}, knowledge: {...}}.

## APPROACH
1. EXPLORE: Understand the document. Read page_texts, use look() to survey pages.
2. PLAN: Group questions by page/data dependencies.
3. DELEGATE: Use subagent() for heavy visual tasks or grouped questions.
4. ACCUMULATE: Store knowledge from sub-agents, reuse for subsequent calls.
5. SUBMIT: Final answer dict with all question_ids.

## OUTPUT FORMAT
- SUBMIT a dict mapping each question_id to its answer string.
"""
```

### 3. Sub-Agent Integration

**Wrapper Layer**: Sub-agents are wrapped to handle page filtering and knowledge injection.

```python
def _run_sub_agent(
    document: Document,
    page_filter: list[int] | None,
    question_ids: list[str] | None,
    knowledge: dict | None,
    max_iterations: int | None,
) -> dict:
    """
    Run a sub-agent with page filtering, question filtering, and knowledge injection.
    Returns {"answers": {...}, "knowledge": {...}}.
    """
    # 1. Filter document pages and questions
    if page_filter is not None:
        filtered_doc = _filter_document(document, page_filter, question_ids)
    else:
        filtered_doc = document

    # 2. Build sub-agent with specified budget
    sub_solver = _create_sub_solver(
        solver_type="flat_batch",
        vlm_lm=self.vlm_lm,
        max_iterations=max_iterations or self.sub_max_iterations,
    )

    # 3. Inject knowledge as category_tips (FlatBatchProgram supports this via parameter)
    # FlatBatchProgram._run_rlm() accepts category_tips as a string argument
    knowledge_str = _format_knowledge(knowledge) if knowledge else ""

    # 4. Run sub-agent with knowledge injection
    # Note: FlatBatchProgram.solve_document() doesn't accept category_tips directly
    # We need to call the internal _run_rlm method or modify the signature
    # For now, we'll use solve_document and enhance it to accept knowledge parameter
    predictions, trajectories = sub_solver.solve_document(filtered_doc)

    # 5. Extract knowledge from results
    extracted_knowledge = _extract_knowledge_from_trajectory(trajectories, predictions)

    return {
        "answers": predictions,
        "knowledge": extracted_knowledge,
    }
```

**Configuration**:
- `sub_solver_type`: flat_batch (default), skilled_batch, rlm, etc.
- Knowledge is formatted as hints and injected via `category_tips` parameter
- Page filtering creates a new Document with only specified pages

**Knowledge Format**:
```python
knowledge = {
    "page_2": "Sales table with regions: North, South, East, West",
    "extracted_tables": {"page_2_table": {...}},
    "previous_answers": {"q1": "$1.2M", "q2": "15%"},
}
# Converted to hints string:
hints = """
KNOWLEDGE FROM MAIN AGENT:
- Page 2: Sales table with regions: North, South, East, West
- Previous answers: q1=$1.2M, q2=15%
"""
```

### 4. Knowledge Flow

```
Main Agent
  |
  | 1. Explore (look, search)
  v
 Extracts: "Page 2 has sales table"
  |
  | 2. Delegate with knowledge
  v
subagent(
  pages=[2, 3],
  questions=["What is total sales?"],
  knowledge={"page_2": "sales table with regional data"}
)
  |
  v
Sub-Agent Returns: {
  "answers": {"q1": "$1.2M"},
  "knowledge": {"extracted_table": {...}}  ← Additional extracted data
}
  |
  v
Main Agent accumulates → reuses for next delegation
```

## Implementation Details

### Helper Functions

```python
def _filter_document(
    document: Document,
    page_indices: list[int],
    question_ids: list[str] | None = None,
) -> Document:
    """Create a new Document with only specified pages and optionally filtered questions.

    Args:
        document: Original document
        page_indices: List of page indices to include (0-indexed)
        question_ids: Optional list of question IDs to include. If None, includes all questions.

    Returns:
        Filtered Document with subset of pages and optionally subset of questions.
    """
    # Filter pages with bounds checking
    filtered_pages = [i for i in page_indices if 0 <= i < len(document.images)]
    if not filtered_pages:
        raise ValueError(f"No valid pages in page_indices={page_indices}")

    # Filter questions if specified
    if question_ids is not None:
        question_id_set = set(question_ids)
        filtered_questions = [q for q in document.questions if q.question_id in question_id_set]
        if not filtered_questions:
            raise ValueError(f"No matching questions in question_ids={question_ids}")
    else:
        filtered_questions = document.questions

    return Document(
        doc_id=document.doc_id,
        doc_category=document.doc_category,
        images=[document.images[i] for i in filtered_pages],
        page_texts=[document.page_texts[i] for i in filtered_pages] if document.page_texts else None,
        questions=filtered_questions,
    )

def _format_knowledge(knowledge: dict | None) -> str:
    """Convert knowledge dict to hints string for sub-agent."""
    if not knowledge:
        return ""
    lines = ["KNOWLEDGE FROM MAIN AGENT:"]
    for key, value in knowledge.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)

def _extract_knowledge_from_trajectory(
    trajectories: dict[str, list[dict]],
    predictions: dict[str, str],
) -> dict:
    """Extract structured knowledge from sub-agent execution.

    Knowledge includes:
    - previous_answers: All answers from this run (reusable for later questions)
    - vlm_observations: VLM look() results with page references
    - extracted_data: Any structured data mentioned in trajectories
    """
    knowledge: dict = {
        "previous_answers": predictions,
    }

    # Extract VLM observations from trajectories
    # (look() calls return useful findings that can be reused)
    for qid, traj in trajectories.items():
        if not traj:
            continue
        for step in traj:
            # Look for look() tool results (observations)
            if "observation" in step:
                obs = step["observation"]
                key = f"{qid}_obs_{step.get('iteration', len(knowledge))}"
                knowledge[key] = str(obs)[:1000]  # Store VLM findings
            # Look for code outputs with structured data
            if "code_output" in step:
                knowledge[f"{qid}_output_{step.get('iteration', 0)}"] = str(step["code_output"])[:1000]

    return knowledge

def _create_sub_solver(
    solver_type: str,
    vlm_lm: dspy.LM,
    max_iterations: int | None = None,
    base_iterations: int = 2,
    iterations_per_question: int = 2,
) -> Any:
    """Factory function to create sub-agent solver instances with configurable budget."""
    if solver_type == "flat_batch":
        from docvqa.solvers.flat_batch_solver import FlatBatchProgram
        return FlatBatchProgram(
            vlm_lm=vlm_lm,
            iterations_per_question=iterations_per_question,
            base_iterations=base_iterations,
        )
    elif solver_type == "skilled_batch":
        from docvqa.solvers.skilled_solver import SkilledBatchProgram
        return SkilledBatchProgram(
            vlm_lm=vlm_lm,
            iterations_per_question=iterations_per_question,
            base_iterations=base_iterations,
        )
    else:
        raise ValueError(f"Unknown sub_solver_type: {solver_type}")
```

### Timeout Handling

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def _run_sub_agent_with_timeout(
    sub_solver, document, timeout_seconds=300
) -> dict:
    """Run sub-agent with timeout protection."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(sub_solver.solve_document, document)
        try:
            predictions, trajectories = future.result(timeout=timeout_seconds)
            return {"answers": predictions, "knowledge": _extract_knowledge_from_trajectory(trajectories)}
        except TimeoutError:
            return {"answers": {}, "knowledge": {}, "error": f"Timeout after {timeout_seconds}s"}
```

### File: `src/docvqa/solvers/meta_solver.py`

```python
class MetaSolver:
    """Meta-solver with sub-agent delegation."""

    def __init__(
        self,
        vlm_lm: dspy.LM,
        sub_solver_type: str = "flat_batch",
        sub_max_iterations: int = 5,
        sub_base_iterations: int = 2,
        sub_iterations_per_question: int = 2,
        iterations_per_question: int = 3,
        base_iterations: int = 8,
    ):
        # Main agent RLM (similar to flat_batch structure)
        # Sub-agent factory (creates solver instances on demand)
        # VLM for direct look() calls

    def solve_document(self, document: Document) -> tuple[dict[str, str], dict[str, list[dict]]]:
        # Save pages to temp dir
        # Build search index
        # Run main RLM with subagent tool
        # Parse answers, return predictions + trajectories
```

### Config: `configs/solver/meta.yaml`

```yaml
_target: docvqa.solvers.meta_solver.create_meta_program

# Main agent settings
iterations_per_question: 3
base_iterations: 8
max_llm_calls: 50

# Sub-agent settings
sub_solver_type: flat_batch
sub_max_iterations: 5
sub_base_iterations: 2
sub_iterations_per_question: 2

# VLM config (shared)
vlm: ${vlm}
```

## Error Handling

- Sub-agent failures return `{"answers": {}, "knowledge": {}, "error": "..."}`
- Main agent can inspect error and retry with different parameters
- Timeout per delegation (configurable, default 300s)

## Testing Strategy

### Quick Test (2-3 fast docs)
- Documents: `engineering_drawing_2`, `business_report_3`
- Success criteria:
  - At least 1 `subagent()` call happens
  - Knowledge dict grows across calls (check logs)
  - No timeout or hanging issues
  - Answers parse correctly

### A/B Test
- Compare meta_solver vs flat_batch on same 5-8 docs
- Metrics: accuracy, time, number of sub-agent calls
- Analyze when subagent is used (complex visual vs simple text)

### Knowledge Accumulation Examples
```python
# First sub-agent call
result1 = subagent(
    pages=[2],
    questions=["What are the regional sales figures?"],
)
# result1["knowledge"] = {"page_2": "Sales table: North=$500K, South=$300K, East=$400K, West=$200K"}

# Second call reuses knowledge
result2 = subagent(
    pages=[2, 3],
    questions=["What is the total sales?", "Which region has highest growth?"],
    knowledge=result1["knowledge"],  # Reuse accumulated knowledge
)
```

## Success Criteria

- [x] Design approved
- [ ] Implementation complete
- [ ] Quick test passes (subagent tool works)
- [ ] Full eval on val set completes
- [ ] Accuracy comparable or better than flat_batch

## Implementation Decisions

### Knowledge Injection Approach (DECIDED)

**Option A chosen**: Add `knowledge: dict | None` parameter to `FlatBatchProgram.solve_document()`.

This is the cleanest approach. The knowledge dict will be:
1. Converted to a hints string via `_format_knowledge()`
2. Appended to the `category_tips` in TASK_INSTRUCTIONS
3. Passed to the sub-agent as additional context

```python
# In FlatBatchProgram.solve_document()
def solve_document(
    self,
    document: Document,
    knowledge: dict | None = None,  # NEW parameter
) -> tuple[dict[str, str], dict[str, list[dict]]]:
    # ... existing code ...

    # Build category tips with knowledge
    tips = get_category_tips(document.doc_category)
    if knowledge:
        knowledge_hints = _format_knowledge(knowledge)
        tips = f"{tips}\n\n{knowledge_hints}"

    # Use in RLM
    result = rlm(
        questions=questions_json,
        doc_info=doc_info,
        page_texts=page_texts,
        category_tips=tips,  # Includes knowledge hints
    )
```

### Tool Registration

The `subagent` tool will be registered with the main agent's RLM:

```python
# In MetaSolver.solve_document()
def _subagent_impl(
    pages: list[int] | None,
    questions: list[str],
    knowledge: dict | None = None,
    max_iterations: int | None = None,
) -> dict:
    """Tool implementation bound to MetaSolver instance."""
    return self._run_sub_agent(
        page_filter=pages,
        question_ids=questions,  # Will be mapped to actual question IDs
        knowledge=knowledge,
        max_iterations=max_iterations,
    )

# Register tools with RLM
tools = [_look_impl, _batch_look_impl, search, _subagent_impl]
rlm = RLM(signature=..., tools=tools, ...)
```

## Open Questions

1. **VLM Resource Sharing**: Main agent and sub-agents both use the same VLM (localhost:8927 Qwen). vLLM has internal queuing which we'll rely on. If issues arise, add explicit serialization with a threading lock in sandbox code.

2. **Sandbox Code Duplication**: Pages are loaded twice (main sandbox + sub-agent sandbox). Acceptable for now (fast local I/O). Could optimize by sharing temp directory between sandboxes if needed.
