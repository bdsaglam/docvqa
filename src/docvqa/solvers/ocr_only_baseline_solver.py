"""OCR-only baseline — the text-perception control for the OCR-free claim.

Clean fork of :mod:`docvqa.solvers.rvlm_ocr_ablation_solver` with the
**vision** channel removed: the agent keeps the LeanRLM REPL scaffold and
an OCR perception channel (``page_texts`` + BM25 ``search``) but has NO
``batch_look``/``look`` and NO image access at all.

Role in the comparison matrix (D-006). The proposed method ``rvlm`` is
OCR-*free* recursive *visual* perception. The skeptic's question is:
*could a cheap OCR-text pipeline match it?* This solver answers exactly
that by holding the scaffold constant (same LeanRLM REPL, same minimal
prompt, same ``search`` tool) and swapping the perception **modality**
from visual (``batch_look``) to textual (OCR ``page_texts`` + ``search``).

- vs ``rvlm`` (vision, no OCR): isolates perception modality, scaffold held constant.
- vs ``repl_only_baseline`` (REPL, NO perception): adds back the OCR text channel only.
- vs ``rvlm_ocr_ablation`` (vision + OCR): this is that solver minus the vision sub-call.

If ``rvlm`` >> this, visual perception is doing real work that OCR text
cannot replace (supports the OCR-free framing). If they tie, the visual
story weakens. Either way it is the load-bearing control.

Prompt parity (D-007). MINIMAL prompt matching ``rvlm_minimal`` /
``rvlm_ocr_ablation``: a category-agnostic ``_TASK_BODY`` plus the
profile's answer-formatting rules, no per-category overlay or tips.
``ANSWER_FORMATTING_RULES`` is read from ``profile.answer_formatting_rules``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import dspy
import logfire

from docvqa.data import Document, Question
from docvqa.datasets.profile import DatasetProfile, get_profile
from docvqa.rlm import LeanRLM, CodeRLM, ThinkingRLM, RLM
from docvqa.search import get_or_build_index

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt body (formatting rules substituted from the profile)
# ---------------------------------------------------------------------------

_TASK_BODY = (
    "You are a Document Visual Question Answering agent operating under an "
    "OCR-only configuration. You answer a question about a document by writing "
    "Python code and reasoning programmatically over OCR-extracted text. You "
    "have NO access to the page images — only their OCR text.\n\n"

    "## DATA\n"
    "- `question`: The question you must answer.\n"
    "- `doc_info`: Document metadata (category and page count).\n"
    "- `page_texts`: OCR-extracted text per page (list of strings, 0-indexed). "
    "This is your ONLY view of the document content. OCR may be inaccurate, "
    "misordered, or missing text in figures/tables/charts — reason carefully.\n\n"

    "## TOOLS\n"
    "- search(query, k=5) -> list[dict]: BM25 search over the OCR text. Returns "
    "[{page, score, text}]. Use it on multi-page documents to locate relevant "
    "pages cheaply. For single-page docs, read `page_texts` directly.\n"
    "- Python REPL: the standard library is available for string manipulation, "
    "arithmetic, counting, sorting, and comparison.\n"
    "- There are NO vision tools (no `look`, no `batch_look`, no image access).\n\n"

    "## APPROACH\n"
    "1. EXPLORE: Read `page_texts` and/or use `search()` to understand the "
    "document structure and locate candidate pages.\n"
    "2. LOCATE: Find the specific span(s) of OCR text relevant to the question. "
    "Use `search()` on multi-page documents to narrow the candidate set.\n"
    "3. EXTRACT: Pull the exact value(s) from the OCR text with Python string "
    "operations.\n"
    "4. COMPUTE/VERIFY: count/compare/compute in Python; cross-check ambiguous "
    "spans against other pages.\n"
    "5. SUBMIT: Once you have the answer, SUBMIT it.\n\n"

    "## GUIDELINES\n"
    "- OCR text may be wrong, reordered, or absent for content rendered as "
    "figures, charts, handwriting, or dense tables. If the answer is not present "
    "in the OCR text, you cannot see it — answer based only on what the text "
    "supports, and use the dataset's unanswerable convention when warranted.\n"
    "- SUPERLATIVES: For 'largest', 'first', 'last', 'only' questions — enumerate "
    "ALL candidates from the text first, then select programmatically. Do NOT "
    "stop at the first match.\n"
    "- NEVER use outside/world knowledge. ALL answers MUST come from the document text.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a single answer string.\n"
    '- Example: SUBMIT(answer="42")\n'
    "- The answer must follow these formatting rules:\n\n"
)

def _build_task_instructions(profile: DatasetProfile) -> str:
    return _TASK_BODY + profile.answer_formatting_rules

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    num_pages: int
    search_index: Any = None
    page_texts: list[str] | None = None

def _format_page_texts(page_texts: list[str]) -> list[str]:
    return [t.strip() or "[No text extracted for this page]" for t in page_texts]

def _build_signature(instructions: str) -> dspy.Signature:
    fields: dict = {
        "question": (
            str,
            dspy.InputField(desc="The question to answer about the document"),
        ),
        "doc_info": (str, dspy.InputField(desc="Document metadata: category and page count")),
        "page_texts": (
            list,
            dspy.InputField(desc="OCR-extracted text per page. List of strings, one per page (0-indexed)."),
        ),
        "answer": (
            str,
            dspy.OutputField(desc="The answer string for the question."),
        ),
    }
    return dspy.Signature(fields, instructions)

def _create_tools(ctx: RunContext) -> list:
    def _search(query: str, k: int = 5) -> list[dict]:
        """Search document OCR text using BM25. Returns list of {page, score, text} records."""
        if ctx.search_index is None:
            return [{"error": "No search index available"}]
        with logfire.span("search", query=query, k=k) as span:
            import bm25s
            import Stemmer
            chunks = ctx.search_index._chunk_meta
            query_tokens = bm25s.tokenize([query], stemmer=Stemmer.Stemmer("english"))
            n = min(k, len(chunks))
            results, scores = ctx.search_index.retrieve(query_tokens, k=n)
            records = []
            for idx, score in zip(results[0], scores[0]):
                if score <= 0:
                    continue
                chunk = chunks[idx]
                records.append({"page": chunk["page"], "score": round(float(score), 2), "text": chunk["text"]})
            span.set_attribute("num_results", len(records))
            return records

    return [_search]

def _build_sandbox_code() -> str:
    """Sandbox code: defines `search()` only. No image loading, no `batch_look`."""
    return '''
def search(query, k=5):
    """BM25 search over OCR text. Returns list of {page, score, text} dicts."""
    return _search(query, k)
'''

# ---------------------------------------------------------------------------
# OcrOnlyBaselineProgram
# ---------------------------------------------------------------------------

class OcrOnlyBaselineProgram:
    """OCR-only baseline — each question solved independently.

    Tool surface: ``search()`` (BM25 over OCR) + ``page_texts`` input. No
    vision: no ``look``/``batch_look``, no image access. Same LeanRLM REPL
    scaffold as ``rvlm`` — only the perception modality differs.
    """

    def __init__(
        self,
        profile: DatasetProfile,
        max_iterations: int = 20,
        rlm_type: str = "lean",
        page_factor: float = 1.5,
        question_concurrency: int = 1,
    ):
        self.profile = profile
        self.max_iterations = max_iterations
        self.rlm_type = rlm_type
        self.page_factor = page_factor
        self.question_concurrency = question_concurrency

    def _per_question_prefix(self, q: Question) -> str:
        if self.profile.question_format_hint_fn is None:
            return ""
        hint = self.profile.question_format_hint_fn(q)
        return f"\n{hint}\n" if hint else ""

    def solve_document(self, document: Document) -> tuple[dict[str, str], dict[str, list[dict]]]:
        """Solve all questions for a document, one question at a time."""
        search_index = None
        if document.page_texts:
            search_index = get_or_build_index(
                document.doc_id,
                document.page_texts,
                bm25_dir=document.bm25_dir,
            )

        ctx = RunContext(
            num_pages=len(document.images),
            search_index=search_index,
            page_texts=document.page_texts,
        )

        doc_info = f"Category: {document.doc_category}, Pages: {len(document.images)}"
        page_texts = _format_page_texts(document.page_texts) if document.page_texts else None
        if page_texts is None:
            page_texts = ["[No OCR text available]"]

        num_pages = len(document.images)
        page_bonus = min(10, self.page_factor * math.ceil(math.sqrt(max(0, num_pages - 9))))
        max_iter = self.max_iterations + int(page_bonus)

        instructions = _build_task_instructions(self.profile)
        tools = _create_tools(ctx)
        sandbox_code = _build_sandbox_code()

        def _solve_question(q: Question):
            """Solve a single question. Returns (question_id, answer, trajectory)."""
            with logfire.span(
                "solve_ocr_only",
                doc_id=document.doc_id,
                question_id=q.question_id,
                question=q.question[:200],
                profile=self.profile.name,
            ) as q_span:
                question_text = q.question + self._per_question_prefix(q)
                RLMClass = {"code": CodeRLM, "lean": LeanRLM, "thinking": ThinkingRLM}.get(self.rlm_type, RLM)
                rlm = RLMClass(
                    signature=_build_signature(instructions),
                    max_iterations=max_iter,
                    max_llm_calls=max_iter * 3,
                    tools=tools,
                    verbose=True,
                    sandbox_code=sandbox_code,
                )
                logger.info(
                    "OCR-only [%s] (%s) Q %s: max_iterations=%d (page_bonus=%d)",
                    self.profile.name, self.rlm_type, q.question_id, max_iter, int(page_bonus),
                )

                result = rlm(
                    question=question_text,
                    doc_info=doc_info,
                    page_texts=page_texts,
                )
                answer = str(result.answer or "").strip()
                trajectory = result.trajectory

                if not answer:
                    answer = "Unknown"

                q_span.set_attribute("num_iterations", len(trajectory))
                q_span.set_attribute("prediction", answer[:200])

                if q.answer is not None:
                    is_correct, extracted = self.profile.score_fn(answer, q.answer, q)
                    q_span.set_attribute("is_correct", is_correct)
                    q_span.set_attribute("ground_truth", q.answer[:200])
                    q_span.set_attribute("extracted_answer", extracted[:200])
                    logger.info(
                        "OCR-only[%s] Q %s: %s (GT=%s, PRED=%s)",
                        self.profile.name,
                        q.question_id,
                        "CORRECT" if is_correct else "WRONG",
                        q.answer[:40],
                        extracted[:40],
                    )

                return q.question_id, answer, trajectory

        predictions: dict[str, str] = {}
        trajectories: dict[str, list[dict]] = {}

        if self.question_concurrency <= 1:
            for q in document.questions:
                qid, answer, trajectory = _solve_question(q)
                predictions[qid] = answer
                trajectories[qid] = trajectory
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_w = min(self.question_concurrency, len(document.questions))
            logger.info("OCR-only: running %d questions with concurrency=%d", len(document.questions), max_w)
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                futures = {pool.submit(_solve_question, q): q for q in document.questions}
                for future in as_completed(futures):
                    qid, answer, trajectory = future.result()
                    predictions[qid] = answer
                    trajectories[qid] = trajectory

        correct = 0
        scored = 0
        for q in document.questions:
            if q.answer is not None:
                scored += 1
                if self.profile.score_fn(predictions[q.question_id], q.answer, q)[0]:
                    correct += 1
        if scored > 0:
            logger.info(
                "OCR-only [%s] doc %s: %d/%d = %.1f%%",
                self.profile.name, document.doc_id, correct, scored,
                100 * correct / scored,
            )

        return predictions, trajectories

# ---------------------------------------------------------------------------
# Factory for hydra instantiation
# ---------------------------------------------------------------------------

def create_ocr_only_baseline_program(
    profile_name: str | None = None,
    dataset: str | None = None,
    max_iterations: int = 20,
    rlm_type: str = "lean",
    page_factor: float = 1.5,
    question_concurrency: int = 4,
) -> OcrOnlyBaselineProgram:
    """Hydra factory. See ``rvlm_ocr_ablation_solver.create_rvlm_ocr_ablation_program``."""
    from docvqa.datasets.profile import _PROFILES  # type: ignore[attr-defined]

    if profile_name is not None:
        for p in _PROFILES.values():
            if p.name == profile_name:
                profile = p
                break
        else:
            profile = get_profile(profile_name)
    elif dataset is not None:
        profile = get_profile(dataset)
    else:
        profile = get_profile("VLR-CVC/DocVQA-2026")

    return OcrOnlyBaselineProgram(
        profile=profile,
        max_iterations=max_iterations,
        rlm_type=rlm_type,
        page_factor=page_factor,
        question_concurrency=question_concurrency,
    )
