"""Orchestrator solver — pure delegation, no direct search.

The main agent is a text-only orchestrator that NEVER touches documents directly.
It delegates ALL exploration and extraction to RVLM sub-agents via batch_subagent().
Sub-agents can see images, search OCR text, and run code. The main agent only
reasons, plans, and synthesizes.

Key difference from meta_solver: main agent has NO search() tool and NO page_texts.
Only tool is batch_subagent(prompts) for parallel sub-agent delegation.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import nullcontext
from typing import Any

import dspy
import logfire
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from docvqa.data import Document
from docvqa.metrics import evaluate_prediction
from docvqa.prompts import ANSWER_FORMATTING_RULES
from docvqa.lean_rlm import LeanRLM
from docvqa.code_rlm import CodeRLM
from docvqa.thinking_rlm import ThinkingRLM
from docvqa.rlm import RLM
from docvqa.rvlm import RVLM
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main agent instructions — pure orchestrator, no direct document access
# ---------------------------------------------------------------------------

ORCHESTRATOR_INSTRUCTIONS = (
    "You are a Document Visual Question Answering orchestrator. You answer questions about documents "
    "by delegating to sub-agents. You cannot see the document yourself.\n\n"

    "## DATA (pre-loaded — do NOT reassign these variables)\n"
    "- `questions`: JSON list of {question_id, question} dicts. You must answer ALL of them.\n"
    "- `doc_info`: Document metadata (category, page count).\n\n"

    "## TOOL (pre-loaded — no need to import)\n"
    "- batch_subagent(prompts) -> list[str]: Run sub-agents IN PARALLEL. "
    "Each prompt spawns a sub-agent that can see and analyze the document. "
    "Returns a list of text results in the same order as prompts.\n\n"

    "## HOW TO USE batch_subagent\n"
    "Always pass a list of prompts — even for a single task, use `batch_subagent(['my task'])`.\n"
    "Sub-agents work best with focused tasks on 1-3 pages. Split large tasks into parallel pieces.\n\n"
    "EXAMPLES:\n"
    "```python\n"
    "# Survey a large document in parallel chunks\n"
    "summaries = batch_subagent([\n"
    "    'Summarize what is on pages 0-4: titles, tables, figures, key content.',\n"
    "    'Summarize what is on pages 5-9: titles, tables, figures, key content.',\n"
    "    'Summarize what is on pages 10-14: titles, tables, figures, key content.',\n"
    "])\n"
    "```\n"
    "```python\n"
    "# Solve independent questions in parallel\n"
    "results = batch_subagent([\n"
    "    'On page 2, what is the total revenue shown in the bar chart?',\n"
    "    'On page 7, read the table and list all product names and their prices.',\n"
    "    'On page 3, what color is the largest segment in the pie chart?',\n"
    "])\n"
    "```\n"
    "```python\n"
    "# Cross-verify a critical value with two independent reads\n"
    "checks = batch_subagent([\n"
    "    'On page 5, read the GDP value for 2023 from the table. Return just the number.',\n"
    "    'On page 5, look at the bar chart for 2023 GDP. What value does the bar reach?',\n"
    "])\n"
    "# Compare checks[0] and checks[1] — if they agree, confident answer\n"
    "```\n"
    "```python\n"
    "# Count across pages: each sub-agent counts its section, you sum\n"
    "counts = batch_subagent([\n"
    "    'Count how many times character X appears in pages 0-3. Return the count.',\n"
    "    'Count how many times character X appears in pages 4-7. Return the count.',\n"
    "])\n"
    "total = int(counts[0]) + int(counts[1])\n"
    "```\n\n"

    "## APPROACH\n"
    "1. EXPLORE: Survey the document. Use `batch_subagent()` to survey page ranges in parallel.\n"
    "2. PLAN: Map questions to relevant pages. Group independent questions for parallel solving.\n"
    "3. SOLVE: Use `batch_subagent()` to answer questions in parallel. "
    "For dependent questions, pass prior findings in the prompt.\n"
    "4. VERIFY: For uncertain answers, use `batch_subagent()` to cross-check via two different approaches.\n"
    "5. SUBMIT: All answers as a dict.\n\n"

    "## GUIDELINES\n"
    "- Sub-agents work best with focused tasks on 1-3 pages. Split large tasks into parallel pieces.\n"
    "- For counting or superlatives: ask sub-agents to list ALL candidates, then counting in Python. Don't ask a subagent to do the counting in multiple pages. Instead, ask different sub-agents to count on single pages and then combine the results.\n"
    "- Extract raw values via sub-agents, compute (sums, differences, percentages) in Python yourself.\n"
    "- Answer 'Unknown' only when multiple sub-agents confirm the info doesn't exist in the document.\n"
    "- NEVER use outside/world knowledge. ALL answers MUST come from the document.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a dict mapping each question_id to its answer string.\n"
    '- Example: SUBMIT(answers={"q1": "42", "q2": "Tokyo"})\n'
    "- Each answer must follow these formatting rules:\n\n"

    + ANSWER_FORMATTING_RULES
)

# ---------------------------------------------------------------------------
# Category-specific tips for meta solver (main agent — delegation-focused)
# ---------------------------------------------------------------------------

MAIN_CATEGORY_TIPS: dict[str, str] = {
    "engineering_drawing": (
        "## CATEGORY TIPS (engineering_drawing)\n"
        "- BOM tables are dense — tell sub-agents to crop the specific row/column at full resolution.\n"
        "- Questions distinguish ITEM NUMBER from PART/IDENTIFYING NUMBER. "
        "Tell sub-agents exactly which column to read.\n"
        "- For counting parts, delegate: 'Crop the BOM table, read ALL rows with their QTY column, return as list'. "
        "Then sum in your own code.\n"
        "- Dimensions: 'Width' = shorter cross-section (Section view), not overall length. "
        "Tell sub-agents which view to display and what dimension to measure.\n"
    ),
    "business_report": (
        "## CATEGORY TIPS (business_report)\n"
        "- Multiple tables may look similar — tell sub-agents which table by position "
        "('the table at the top of page 0', 'the table titled Revenue').\n"
        "- For YoY calculations: delegate extraction of raw values, compute in Python yourself.\n"
        "- 'Broken down into' = immediate sub-categories only.\n"
        "- Pictograms: delegate 'crop each icon and describe what it shows' rather than yes/no filtering.\n"
    ),
    "comics": (
        "## CATEGORY TIPS (comics)\n"
        "- OCR is useless for speech bubbles — ALL extraction must go through sub-agents visually.\n"
        "- STORY MAP FIRST: Before answering, build a story index by delegating: "
        "'Scan pages X-Y, identify story titles and page ranges'. Then match question to story.\n"
        "- For counting events: give sub-agents strict inclusion criteria in the prompt "
        "('Is someone physically hit in the head? Exclude mentions, near-misses, aftermath'). "
        "Then count matches in code.\n"
        "- LITERAL VS FIGURATIVE: 'in reality', 'actually', 'truly' means the answer contradicts "
        "the surface description. Tell sub-agents to look for the distinction.\n"
    ),
    "maps": (
        "## CATEGORY TIPS (maps)\n"
        "- Spatial reasoning is hard for VLMs — break into small, precise sub-agent tasks.\n"
        "- Delegate grid reads: 'Display the map, crop to grid cell C4, list everything visible'.\n"
        "- For road types: delegate 'Crop the legend and the road segment at HIGH resolution, compare line styles'.\n"
        "- For distances/directions: collect landmark coordinates from sub-agents, compute with Python.\n"
    ),
    "science_paper": (
        "## CATEGORY TIPS (science_paper)\n"
        "- Papers are long — use search() and page_texts to locate relevant sections before delegating.\n"
        "- Citation questions: find [N] patterns in page_texts with regex, delegate only ambiguous ones to sub-agents.\n"
        "- Ablation studies: verify the section matches the specific component asked about "
        "before delegating extraction.\n"
    ),
    "science_poster": (
        "## CATEGORY TIPS (science_poster)\n"
        "- Single dense page — sub-agents don't need page navigation, just precise cropping.\n"
        "- Chart values: tell sub-agents to read annotation labels on bars/lines, not estimate from heights.\n"
        "- 'Percentage improvement' = absolute difference in percentage points (80% - 50% = 30%).\n"
        "- Color-coded values: delegate 'crop this table at MAX resolution, read all red/blue numbers'.\n"
    ),
    "infographics": (
        "## CATEGORY TIPS (infographics)\n"
        "- OCR describes images rather than reading text — delegate all text extraction to sub-agents.\n"
        "- N/A in charts: if a comparison shows N/A for an item, answer 'Unknown' regardless of narrative text.\n"
        "- Stylized icons: delegate 'describe each icon individually at high resolution' "
        "rather than asking about shapes in bulk.\n"
    ),
    "slide": (
        "## CATEGORY TIPS (slide)\n"
        "- Many pages — use search() to find the right slide first, then delegate visual extraction.\n"
        "- Off-by-one page errors are common — verify page indices with sub-agents before trusting them.\n"
        "- Tables in slides are small — tell sub-agents to crop at full resolution before reading.\n"
        "- Exact entity matching: if a column/variable doesn't exist after thorough search, answer 'Unknown'.\n"
    ),
}

SUB_CATEGORY_TIPS: dict[str, str] = {
    "engineering_drawing": (
        "## CATEGORY TIPS (engineering_drawing)\n"
        "- Part numbers are almost always numeric (0-9 plus dashes). If you see I, O, l where digits "
        "1, 0 would be expected, re-display at higher zoom. Common: I↔1, O↔0, l↔1.\n"
        "- Crop BOM table rows one section at a time. Read the QTY column as numbers.\n"
        "- Leader lines on drawings: display the label AND the part it connects to separately.\n"
    ),
    "business_report": (
        "## CATEGORY TIPS (business_report)\n"
        "- Multiple similar tables may exist — verify you're reading the correct one by checking headers/titles.\n"
        "- Read chart values from annotations/tick labels, not by estimating bar heights.\n"
        "- Qualitative descriptions (adjectives) may be in footnotes or paragraphs, not tables.\n"
    ),
    "comics": (
        "## CATEGORY TIPS (comics)\n"
        "- Read speech bubbles and panel actions visually — OCR is unreliable for comics.\n"
        "- Describe each panel's action explicitly: 'what happens in this panel?'\n"
        "- Character names: use the exact term from speech bubbles.\n"
    ),
    "maps": (
        "## CATEGORY TIPS (maps)\n"
        "- Start coarse (full page), then zoom to ~800px crops, then ~400px for small text.\n"
        "- For road types: crop the legend AND the road segment at high resolution. Compare line styles.\n"
        "- Grid cells: crop the specific cell and list everything visible.\n"
    ),
    "science_paper": (
        "## CATEGORY TIPS (science_paper)\n"
        "- Citation numbers: look for [N] patterns in text. Distinguish body text from table headers/figure captions.\n"
        "- If a specific entity (layer number, model variant) isn't visible after thorough search, say so.\n"
    ),
    "science_poster": (
        "## CATEGORY TIPS (science_poster)\n"
        "- Dense single page — crop specific sections precisely.\n"
        "- Read chart annotations directly on bars/lines rather than estimating from heights.\n"
        "- Color-coded values: crop at maximum resolution, describe individual cell colors.\n"
    ),
    "infographics": (
        "## CATEGORY TIPS (infographics)\n"
        "- OCR describes images — read actual text by displaying the cropped region.\n"
        "- Examine icons individually at high resolution for identification.\n"
        "- If a chart shows N/A for an item, report N/A — don't substitute from narrative text.\n"
    ),
    "slide": (
        "## CATEGORY TIPS (slide)\n"
        "- Tables in slides are small — always crop at full resolution before reading values.\n"
        "- Verify you're on the correct slide by checking the title/header.\n"
    ),
}


def _get_main_tips(category: str | None) -> str:
    if not category or category not in MAIN_CATEGORY_TIPS:
        return ""
    return "\n" + MAIN_CATEGORY_TIPS[category]


def _get_sub_tips(category: str | None) -> str:
    if not category or category not in SUB_CATEGORY_TIPS:
        return ""
    return "\n" + SUB_CATEGORY_TIPS[category]


# ---------------------------------------------------------------------------
# Signature builders
# ---------------------------------------------------------------------------

def _build_orchestrator_signature(category: str | None = None, tips: bool = True) -> dspy.Signature:
    instructions = ORCHESTRATOR_INSTRUCTIONS + (_get_main_tips(category) if tips else "")
    fields: dict = {
        "questions": (
            str,
            dspy.InputField(desc="JSON list of {question_id, question} dicts — answer ALL of them"),
        ),
        "doc_info": (str, dspy.InputField(desc="Document metadata: category and page count")),
        "answers": (
            str,
            dspy.OutputField(desc="Dict mapping question_id to answer string. Must include ALL question_ids."),
        ),
    }
    return dspy.Signature(fields, instructions)


SUB_AGENT_INSTRUCTIONS = (
    "You are a document analysis sub-agent. Complete the task in `prompt` and return a clear, structured result.\n\n"

    "## DATA (pre-loaded — do NOT reassign these variables)\n"
    "- `prompt`: Your instructions from the main agent.\n"
    "- `pages`: PRE-LOADED list of PIL Images (0-indexed).\n\n"

    "## TOOLS\n"
    "- display(image) — Show a PIL Image inline. You will SEE the image in the next step. "
    "Use it to view pages, crops, or any processed image. "
    "Example: `display(pages[0])` or `display(pages[0].crop((l,t,r,b)))`.\n"
    "- print() — ALWAYS print to see text results.\n\n"

    "## APPROACH\n"
    "- After `display()`, LOOK at the image and describe what you see in your reasoning.\n"
    "- Full-page `display()` gives overview. For fine details, crop first: "
    "`display(pages[i].crop((l,t,r,b)))`. Use `pages[i].size` for dimensions.\n"
    "- **One fact per display()**: Ask yourself one specific question before each display. "
    "E.g., 'What is the number in the 3rd column?' not 'Tell me everything'.\n"
    "- CONFLICT RESOLUTION: If you read conflicting values, crop TIGHTER and display again.\n"
    "- SUPERLATIVES: Enumerate ALL candidates first, then select programmatically.\n\n"

    "## EXTRACTION STRATEGIES\n"
    "- **Tables**: Display in overlapping horizontal strips so no row falls on a boundary. "
    "Read each strip, stitch rows in code.\n"
    "- **Dense text/small print**: Crop to the exact region and display at high zoom. "
    "Read characters one at a time if needed — a wrong digit changes the answer.\n"
    "- **Technical drawings**: Trace leader lines carefully. Display the label region and "
    "the part it points to separately to confirm connections.\n"
    "- **Charts/graphs**: Read axis labels first (display the axis area cropped), then read data points. "
    "Don't estimate from bars — check the tick labels.\n"
    "- **Spatial questions**: Display the full page first to orient, then crop to each region of interest.\n"

    "## GUIDELINES\n"
    "- Focus on the prompt. Return exactly what was asked for, WITH EVIDENCE.\n"
    "- Always include proof: exact values read, crop coordinates used.\n"
    "- Extract raw facts visually, then count/compare/compute in Python.\n"
    "- **VERIFY your reads**: Don't trust a single display(). For critical values, "
    "display the same region at different scales — full page for context, then tighter crop "
    "for precision. If two reads disagree, crop even tighter to break the tie.\n"
    "- OCR text (`page_texts`) is unreliable — use it only to locate relevant regions, not as final answers.\n"
    "- Be concise — the main agent will interpret your result.\n"
)


def _build_sub_signature(category: str | None = None, tips: bool = True) -> dspy.Signature:
    instructions = SUB_AGENT_INSTRUCTIONS + (_get_sub_tips(category) if tips else "")
    fields: dict = {
        "prompt": (str, dspy.InputField(desc="Instructions from the main agent")),
        "result": (
            str,
            dspy.OutputField(desc="Concise structured result with evidence."),
        ),
    }
    return dspy.Signature(fields, instructions)

# ---------------------------------------------------------------------------
# Sandbox code builder (shared by main and sub agents)
# ---------------------------------------------------------------------------

def _build_main_sandbox_code(tmpdir: str, num_pages: int) -> str:
    """Build sandbox code for the main agent (search only, no vision tools)."""
    return f'''
import os

def search(query, k=5):
    """BM25 search over OCR text. Returns list of {{page, score, text}} dicts."""
    return _search(query, k)
'''


def _build_sub_sandbox_code(tmpdir: str, num_pages: int) -> str:
    """Build sandbox code for RVLM sub-agents — pages only, display() built into RVLM."""
    return f'''
import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000
pages = []
for i in range({num_pages}):
    path = os.path.join({tmpdir!r}, f"page_{{i}}.png")
    assert os.path.exists(path), f"Page image not found: {{path}}"
    pages.append(Image.open(path))
assert len(pages) == {num_pages}, f"Expected {num_pages} pages, got {{len(pages)}}"
'''


# ---------------------------------------------------------------------------
# OrchestratorSolver
# ---------------------------------------------------------------------------

class OrchestratorSolver:
    """Pure orchestrator — main agent has NO direct document access, delegates everything."""

    def __init__(
        self,
        sub_max_iterations: int = 8,
        iterations_per_question: int = 3,
        base_iterations: int = 8,
        sub_images_for_last_n: int = 1,
        sub_max_image_pixels: int = 1_000_000,
        main_tips: bool = True,
        sub_tips: bool = True,
        vlm_lm: dspy.LM | None = None,
        rlm_type: str = "standard",
    ):
        self.sub_max_iterations = sub_max_iterations
        self.iterations_per_question = iterations_per_question
        self.base_iterations = base_iterations
        self.sub_images_for_last_n = sub_images_for_last_n
        self.sub_max_image_pixels = sub_max_image_pixels
        self.main_tips = main_tips
        self.sub_tips = sub_tips
        self.vlm_lm = vlm_lm
        self.rlm_type = rlm_type

    def solve_document(self, document: Document) -> tuple[dict[str, str], dict[str, list[dict]]]:
        """Solve all questions for a document using pure orchestrator approach."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save pages for sub-agents
            for i, img in enumerate(document.images):
                img.save(os.path.join(tmpdir, f"page_{i}.png"), format="PNG")

            # Build search index for sub-agents
            search_index = None
            if document.page_texts:
                from docvqa.search import get_or_build_index
                search_index = get_or_build_index(document.doc_id, document.page_texts)

            doc_info = f"Category: {document.doc_category}, Pages: {len(document.images)}"
            page_texts = [t.strip() or "[No text]" for t in (document.page_texts or [])]

            num_q = len(document.questions)
            max_iter = self.base_iterations + self.iterations_per_question * num_q

            # Sub-agent sandbox code (has pages, display, search via host)
            sub_sandbox = _build_sub_sandbox_code(tmpdir, len(document.images))

            # ------ Search impl for sub-agents ------
            def _search(query: str, k: int = 5) -> list[dict]:
                k = min(k, 5)
                if search_index is None:
                    return [{"error": "No search index available"}]
                with logfire.span("search", query=query, k=k) as span:
                    import bm25s
                    import Stemmer
                    chunks = search_index._chunk_meta
                    query_tokens = bm25s.tokenize([query], stemmer=Stemmer.Stemmer("english"))
                    n = min(k, len(chunks))
                    results, scores = search_index.retrieve(query_tokens, k=n)
                    records = []
                    for idx, score in zip(results[0], scores[0]):
                        if score <= 0:
                            continue
                        chunk = chunks[idx]
                        records.append({"page": chunk["page"], "score": round(float(score), 2), "text": chunk["text"]})
                    span.set_attribute("num_results", len(records))
                    return records

            # ------ Sub-agent (RVLM with search + page_texts) ------
            def _subagent_impl(prompt: str) -> str:
                with logfire.span("subagent", prompt=prompt[:500]) as span:
                    # Sub-agent gets search as a tool + page_texts in sandbox
                    sub_sandbox_with_search = sub_sandbox + f'''
page_texts = {page_texts!r}

def search(query, k=5):
    """BM25 search over OCR text. Returns list of {{page, score, text}} dicts."""
    return _search(query, k)
'''
                    sub_rvlm = RVLM(
                        signature=_build_sub_signature(
                            category=document.doc_category, tips=self.sub_tips
                        ),
                        max_iterations=self.sub_max_iterations,
                        max_llm_calls=self.sub_max_iterations * 3,
                        tools=[_search],
                        verbose=True,
                        sandbox_code=sub_sandbox_with_search,
                        images_for_last_n=self.sub_images_for_last_n,
                        max_image_pixels=self.sub_max_image_pixels,
                    )
                    try:
                        ctx = dspy.context(lm=self.vlm_lm) if self.vlm_lm else nullcontext()
                        with ctx:
                            sub_result = sub_rvlm(prompt=prompt)
                        response = sub_result.result or ""
                        span.set_attribute("response", response[:2000])
                        return response
                    except Exception as e:
                        logger.warning("Sub-agent failed: %s", e)
                        span.set_attribute("error", str(e)[:500])
                        return f"[Sub-agent error: {e}]"

            def _subagent_with_timeout(prompt: str, timeout_seconds: int = 900) -> str:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_subagent_impl, prompt)
                    try:
                        return future.result(timeout=timeout_seconds)
                    except TimeoutError:
                        logger.warning("Sub-agent timeout after %ds", timeout_seconds)
                        return f"[Sub-agent timeout after {timeout_seconds}s]"

            def batch_subagent(prompts: list) -> list[str]:
                """Run multiple sub-agents in parallel."""
                from concurrent.futures import as_completed
                if not prompts:
                    return []
                results: list[str] = [""] * len(prompts)

                def _run(idx: int, prompt: str) -> tuple[int, str]:
                    return idx, _subagent_with_timeout(prompt)

                with logfire.span("batch_subagent", num_tasks=len(prompts)):
                    with ThreadPoolExecutor(max_workers=min(len(prompts), 4)) as pool:
                        futures = {pool.submit(_run, i, p): i for i, p in enumerate(prompts)}
                        for future in as_completed(futures):
                            idx, result = future.result()
                            results[idx] = result
                return results

            # Main agent: NO search, NO page_texts — only batch_subagent tool
            tools = [batch_subagent]
            RLMClass = {"code": CodeRLM, "lean": LeanRLM, "thinking": ThinkingRLM}.get(self.rlm_type, RLM)
            rlm = RLMClass(
                signature=_build_orchestrator_signature(
                    category=document.doc_category, tips=self.main_tips
                ),
                max_iterations=max_iter,
                max_llm_calls=max_iter * 3,
                tools=tools,
                verbose=True,
                sandbox_code="",  # No sandbox setup needed — main agent has no data
            )

            # Use short IDs
            short_to_full = {}
            questions_list = []
            for i, q in enumerate(document.questions):
                short_id = f"q{i + 1}"
                short_to_full[short_id] = q.question_id
                questions_list.append({"question_id": short_id, "question": q.question})
            questions_json = json.dumps(questions_list)
            short_ids = set(short_to_full.keys())

            # Rate limit retry
            def _is_rate_limit(e: BaseException) -> bool:
                return "429" in str(e) or "RateLimit" in type(e).__name__ or "RESOURCE_EXHAUSTED" in str(e)

            @retry(
                retry=retry_if_exception(_is_rate_limit),
                stop=stop_after_attempt(4),
                wait=wait_exponential(multiplier=30, min=30, max=120),
                before_sleep=lambda rs: logger.warning(
                    "Rate limit, retry %d in %.0fs", rs.attempt_number, rs.next_action.sleep
                ),
                reraise=True,
            )
            def _solve_batch():
                return rlm(
                    questions=questions_json,
                    doc_info=doc_info,
                )

            with logfire.span(
                "solve_orchestrator",
                doc_id=document.doc_id,
                doc_category=document.doc_category,
                num_questions=len(questions_list),
                num_pages=len(document.images),
            ) as batch_span:
                try:
                    result = _solve_batch()
                    raw_answers = result.answers
                    trajectory = result.trajectory
                except Exception as e:
                    logger.warning("Orchestrator failed for doc '%s': %s", document.doc_id, e)
                    raw_answers = "{}"
                    trajectory = []

                short_answers = _parse_answers(raw_answers, short_ids)
                answers_dict = {
                    short_to_full[k]: v for k, v in short_answers.items() if k in short_to_full
                }
                batch_span.set_attribute("num_iterations", len(trajectory))
                batch_span.set_attribute("num_parsed_answers", len(answers_dict))

                predictions = {}
                trajectories = {}

                for q in document.questions:
                    answer = answers_dict.get(q.question_id, "Unknown")
                    if not answer or answer.strip() == "":
                        answer = "Unknown"
                    predictions[q.question_id] = answer
                    trajectories[q.question_id] = trajectory

                    if q.answer is not None:
                        is_correct, extracted = evaluate_prediction(answer, q.answer)
                        with logfire.span(
                            "orchestrator_question_result",
                            question_id=q.question_id,
                            question=q.question[:200],
                            is_correct=is_correct,
                            prediction=answer[:200],
                            ground_truth=q.answer[:200],
                            extracted_answer=extracted[:200],
                        ):
                            pass
                        logger.info(
                            "Orch Q %s: %s (GT=%s, PRED=%s)",
                            q.question_id,
                            "CORRECT" if is_correct else "WRONG",
                            q.answer[:40],
                            extracted[:40],
                        )

            return predictions, trajectories


# ---------------------------------------------------------------------------
# Factory for hydra instantiation
# ---------------------------------------------------------------------------

def create_orchestrator_program(
    sub_max_iterations: int = 8,
    iterations_per_question: int = 3,
    base_iterations: int = 8,
    sub_images_for_last_n: int = 1,
    sub_max_image_pixels: int = 1_000_000,
    main_tips: bool = True,
    sub_tips: bool = True,
    vlm: dict[str, Any] | None = None,
    rlm_type: str = "standard",
) -> OrchestratorSolver:
    # Create VLM LM for sub-agents — MUST be separate from main LLM
    from docvqa.types import LMConfig
    vlm_lm = None
    if vlm and vlm.get("model"):
        vlm_config = LMConfig(
            model=vlm["model"],
            api_base=vlm.get("api_base"),
            api_key=vlm.get("api_key"),
            max_tokens=vlm.get("max_tokens", 65536),
            temperature=vlm.get("temperature", 1.0),
            vertex_location=vlm.get("vertex_location"),
        )
        vlm_lm = vlm_config.to_dspy_lm()
        logger.info("Orchestrator: sub-agents will use VLM %s (not main LLM)", vlm["model"])
    else:
        logger.warning("Orchestrator: no VLM config — sub-agents will inherit main LLM!")

    return OrchestratorSolver(
        sub_max_iterations=sub_max_iterations,
        iterations_per_question=iterations_per_question,
        base_iterations=base_iterations,
        sub_images_for_last_n=sub_images_for_last_n,
        sub_max_image_pixels=sub_max_image_pixels,
        main_tips=main_tips,
        sub_tips=sub_tips,
        vlm_lm=vlm_lm,
        rlm_type=rlm_type,
    )
