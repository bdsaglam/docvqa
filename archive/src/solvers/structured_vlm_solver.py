"""Structured VLM solver — batch_look with optional Pydantic schema support.

Cloned from parallel_vlm_solver.py. Key addition: batch_look accepts
(image, query, PydanticModel) tuples. When a schema is provided, the VLM
returns structured JSON matching the schema, parsed back into a dict that
the sandbox can reconstruct into the Pydantic model.

IPC protocol:
  - Sandbox converts PydanticModel → JSON schema dict via model_json_schema()
  - Host receives schema, calls litellm with response_format={type: json_schema, ...}
  - Host returns parsed dict matching the schema
  - Sandbox reconstructs: Model(**result_dict)
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any

import dspy
import litellm
import logfire
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from docvqa.data import Document
from docvqa.metrics import evaluate_prediction
from docvqa.prompts import ANSWER_FORMATTING_RULES, get_category_tips
from docvqa.rlm import RLM
from docvqa.search import get_or_build_index
from docvqa.types import LMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

TASK_INSTRUCTIONS = (
    "You are a Document Visual Question Answering agent. You answer questions about documents by "
    "writing Python code, calling vision tools iteratively, and reasoning programmatically.\n\n"

    "## DATA\n"
    "- `questions`: JSON list of {question_id, question} dicts. You must answer ALL of them.\n"
    "- `page_texts`: OCR-extracted text per page. May be inaccurate — verify critical values visually.\n"
    "- `pages`: list of page images (PIL Images) (0-indexed). Crop with `pages[i].crop((l,t,r,b))`.\n\n"

    "## TOOLS\n"
    "- search(query, k=5) -> list[dict]: BM25 search over OCR text. Returns [{page, score, text}]. "
    "Useful for multi-page documents to locate relevant pages. For single-page docs, read `page_texts` directly.\n"
    "- batch_look(requests) -> list[str | dict]: Send multiple images to the VLM IN PARALLEL. "
    "Input: list of (image, query) or (image, query, PydanticModel) tuples. Returns: list of answers in same order. "
    "When a PydanticModel is provided, the VLM returns structured data matching the schema (as a dict). "
    "Without a model, returns a plain string as before.\n\n"

    "## STRUCTURED OUTPUT\n"
    "For reliable data extraction, define Pydantic models and pass them to batch_look:\n"
    "```python\n"
    "from pydantic import BaseModel\n\n"
    "class NumberReading(BaseModel):\n"
    "    explanation: str  # VLM reasoning — always include\n"
    "    value: float\n\n"
    "class TableRow(BaseModel):\n"
    "    item: str\n"
    "    quantity: int\n\n"
    "class TableReading(BaseModel):\n"
    "    explanation: str\n"
    "    rows: list[TableRow]\n\n"
    "# Structured results — no parsing needed\n"
    "results = batch_look([\n"
    "    (crop, 'Read the number shown', NumberReading),\n"
    "    (crop2, 'Extract all table rows', TableReading),\n"
    "])\n"
    "number = results[0].value  # float, directly usable\n"
    "rows = results[1].rows     # list[TableRow]\n"
    "```\n"
    "Always include an `explanation` field in your models — it improves VLM accuracy and lets you "
    "inspect reasoning when results are ambiguous.\n\n"
    "**Majority voting with structured output:**\n"
    "```python\n"
    "readings = batch_look([(crop, 'Read the number', NumberReading)] * 3)\n"
    "values = [r.value for r in readings]\n"
    "from collections import Counter\n"
    "answer = Counter(values).most_common(1)[0][0]\n"
    "```\n\n"

    "## KEY PRINCIPLE: PARALLEL READS FOR RELIABILITY\n"
    "The VLM is fast but noisy — the same query on the same crop can return different answers. "
    "Instead of re-querying sequentially, exploit parallelism. Think of `batch_look` as a sensor array "
    "you can point at the document however you want — all queries run simultaneously.\n\n"

    "**Strategies** (combine freely):\n"
    "- **Overlapping sweep**: To read a table, split it into overlapping horizontal strips and read each strip "
    "in parallel. Stitch results in code. Overlap ensures no row falls on a boundary.\n"
    "- **Grid scan**: To find something on a page, split into a grid (e.g. 3x3) and batch_look all 9 crops "
    "with 'what labels/landmarks are here?'. Build a spatial map in code.\n"
    "- **Multi-scale**: Same query at different crop sizes — full page for context, tight crop for precision.\n"
    "- **Multi-phrasing**: Same region, different question wordings — catches VLM blind spots.\n"
    "- **Redundant reads**: For critical values, send 2-3 identical pairs and majority-vote.\n"
    "- **Cross-region verification**: Read a value from the table AND from a chart that shows the same data.\n"
    "After receiving results, analyze in Python: `Counter(results).most_common(1)`, detect outliers, "
    "stitch strips, or cross-validate. Trust consensus, not any single read.\n\n"

    "## APPROACH\n"
    "1. EXPLORE: Read `page_texts`, then batch_look full pages to understand layout.\n"
    "2. PLAN: Group questions by region. Design batch queries that serve multiple questions at once.\n"
    "3. SOLVE: For each question, design a batch of parallel queries — sweeps, multi-scale, redundant reads. "
    "Send a LARGE batch at once (8-16 queries is fine). Analyze results in Python — find consensus, "
    "flag disagreements. Only do another batch if results are truly ambiguous.\n"
    "4. SUBMIT: Once all questions are answered, SUBMIT all answers together.\n\n"

    "## GUIDELINES\n"
    "- Full-page batch_look gives a broad overview. For fine details, CROP FIRST: `pages[i].crop((l,t,r,b))`.\n"
    "- Use `pages[i].size` to get dimensions for cropping.\n"
    "- Ask the VLM ONE simple factual question per query. Extract raw facts, then compute in Python.\n"
    "- A single VLM read is UNRELIABLE. Always design for redundancy — send 2-3 reads per critical value.\n"
    "- CONFLICT RESOLUTION: When batch reads disagree, give more weight to the TIGHTER crop. "
    "Never blindly adopt a number — compare across reads and take the consensus.\n"
    "- For tables: sweep overlapping strips from top to bottom, read each strip's rows.\n"
    "- For spatial questions: grid-scan to locate landmarks, then compute relationships in code.\n"
    "- SUPERLATIVES: For 'largest', 'first', 'last', 'only' — enumerate ALL candidates, don't stop at first match.\n"
    "- UNKNOWN RULES: Redundancy helps find answers, but do NOT let it prevent you from answering 'Unknown'. "
    "If a specific entity genuinely does not exist, more reads won't find it. Answer 'Unknown' when:\n"
    "  (a) A specific named entity (column name, layer number, variable) does not exist after thorough search.\n"
    "  (b) A chart/table explicitly shows N/A or missing data for the requested item.\n"
    "  (c) Multiple independent reads all fail to find the requested information — that IS the signal.\n"
    "  Do NOT substitute a similar-sounding entity or extrapolate from nearby data.\n"
    "  Do NOT use narrative/descriptive text when a chart explicitly shows N/A.\n"
    "- COMPUTATION: When a question says 'total' or 'considering X and Y', it may require arithmetic. "
    "Extract all referenced values and compute explicitly in Python.\n"
    "- Be efficient. Reuse observations. Aim for large, well-designed batches over many small ones.\n"
    "- NEVER use outside/world knowledge. ALL answers MUST come from the document.\n\n"

    "## OUTPUT FORMAT\n"
    "- SUBMIT a dict mapping each question_id to its answer string.\n"
    '- Example: SUBMIT(answers={"q1": "42", "q2": "Tokyo"})\n'
    "- Each answer must follow these formatting rules:\n\n"

    + ANSWER_FORMATTING_RULES
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_page_texts(page_texts: list[str]) -> list[str]:
    return [t.strip() or "[No text extracted - use batch_look() for visual content]" for t in page_texts]


def _build_signature(instructions: str = TASK_INSTRUCTIONS) -> dspy.Signature:
    fields: dict = {
        "questions": (
            str,
            dspy.InputField(desc="JSON list of {question_id, question} dicts — answer ALL of them"),
        ),
        "doc_info": (str, dspy.InputField(desc="Document metadata: category and page count")),
        "page_texts": (
            list,
            dspy.InputField(desc="OCR-extracted text per page. List of strings, one per page (0-indexed)."),
        ),
        "answers": (
            str,
            dspy.OutputField(desc="Dict mapping question_id to answer string. Must include ALL question_ids."),
        ),
    }
    return dspy.Signature(fields, instructions)


@dataclass
class RunContext:
    page_dir: str
    num_pages: int
    search_index: Any = None
    page_texts: list[str] | None = None


def _create_tools(vlm_lm: dspy.LM, ctx: RunContext) -> list:
    from PIL import Image as PILImage

    # Extract litellm-compatible model/kwargs from dspy.LM
    vlm_model = vlm_lm.model
    vlm_api_base = vlm_lm.kwargs.get("api_base")
    vlm_api_key = vlm_lm.kwargs.get("api_key", "dummy")
    vlm_max_tokens = vlm_lm.kwargs.get("max_tokens", 65536)
    vlm_extra_body = vlm_lm.kwargs.get("extra_body")

    # Pre-build dspy.Predict for unstructured calls (reused across invocations)
    _vlm_predict = dspy.Predict(
        dspy.Signature(
            {
                "image": (dspy.Image, dspy.InputField(desc="Image")),
                "query": (str, dspy.InputField(desc="Query")),
                "answer": (str, dspy.OutputField(desc="Concise response")),
            },
            "Analyze the image content strictly to answer the query. "
            "Transcribe numbers and characters exactly. "
            "For technical drawings, trace leader lines and arrows to connect labels to their specific parts. "
            "Output ONLY the concise answer. If the information is missing, output 'Unknown'.",
        )
    )

    def _look_impl(image_path: str, query: str, schema_json: str | None = None) -> str | dict:
        """Internal: load image from path and send to VLM. Optionally with structured output."""
        import base64

        with logfire.span("look", image_path=image_path, query=query, structured=schema_json is not None) as span:
            if schema_json:
                # Structured output mode — use litellm directly with response_format
                with open(image_path, "rb") as f:
                    img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                schema = json.loads(schema_json)
                schema_name = schema.get("title", "structured_output")
                clean_schema = _clean_schema_for_vllm(schema)

                call_kwargs: dict[str, Any] = {
                    "model": vlm_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Analyze the image content strictly to answer the query. "
                                "Transcribe numbers and characters exactly. "
                                "For technical drawings, trace leader lines and arrows to connect labels to their specific parts. "
                                "If the information is missing, say so in the explanation field. "
                                "Respond with JSON only."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": query},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                                },
                            ],
                        }
                    ],
                    "max_tokens": vlm_max_tokens,
                    "timeout": 300,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": clean_schema,
                        },
                    },
                }
                if vlm_api_base:
                    call_kwargs["api_base"] = vlm_api_base
                if vlm_api_key:
                    call_kwargs["api_key"] = vlm_api_key
                if vlm_extra_body:
                    call_kwargs["extra_body"] = vlm_extra_body

                resp = litellm.completion(**call_kwargs)
                text = resp.choices[0].message.content or "{}"
                try:
                    result = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning("Structured VLM response not valid JSON: %s", text[:200])
                    result = {"error": text}
                span.set_attribute("answer", str(result)[:2000])
                return result
            else:
                # Plain text mode (backward compatible)
                img = PILImage.open(image_path)
                with dspy.context(lm=vlm_lm):
                    result = _vlm_predict(image=dspy.Image(img), query=query)
                    answer = result.answer or ""
                    span.set_attribute("answer", answer[:2000])
                    return answer

    def _batch_look_impl(requests_json: str) -> list[str | dict]:
        """Internal: batch VLM calls in parallel. Input is JSON list of {path, query, schema?}."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        requests = json.loads(requests_json)
        if not requests:
            return []
        results: list[str | dict] = [""] * len(requests)
        num_structured = sum(1 for r in requests if r.get("schema"))
        logger.info("batch_look: %d/%d requests have schema (keys: %s)", num_structured, len(requests),
                     [list(r.keys()) for r in requests[:3]])

        def _do(idx: int, path: str, query: str, schema_json: str | None) -> tuple[int, str | dict]:
            return idx, _look_impl(path, query, schema_json)

        is_vertex = "vertex_ai" in (vlm_lm.model if hasattr(vlm_lm, 'model') else str(vlm_lm))
        max_w = min(len(requests), 2 if is_vertex else 8)
        with logfire.span("batch_look", num_requests=len(requests)):
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                futures = {
                    pool.submit(_do, i, r["path"], r["query"], r.get("schema")): i
                    for i, r in enumerate(requests)
                }
                for future in as_completed(futures):
                    idx, answer = future.result()
                    results[idx] = answer
        return results

    def _search(query: str, k: int = 5) -> list[dict]:
        """Search document text using BM25. Returns list of {page, score, text} records."""
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

    return [_batch_look_impl, _search]


def _clean_schema_for_vllm(schema: dict) -> dict:
    """Clean a Pydantic JSON schema for vLLM strict mode.

    vLLM strict mode requires additionalProperties: false on all objects
    and doesn't support 'title' or 'default' fields inside properties.
    """
    schema = dict(schema)  # shallow copy

    # Remove top-level fields vLLM doesn't want
    schema.pop("title", None)

    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        # Ensure all properties are required
        if "properties" in schema and "required" not in schema:
            schema["required"] = list(schema["properties"].keys())

    # Recursively clean properties
    if "properties" in schema:
        cleaned_props = {}
        for key, prop in schema["properties"].items():
            cleaned_props[key] = _clean_schema_for_vllm(prop)
        schema["properties"] = cleaned_props

    # Clean items (for arrays)
    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = _clean_schema_for_vllm(schema["items"])

    # Clean $defs
    if "$defs" in schema:
        cleaned_defs = {}
        for key, defn in schema["$defs"].items():
            cleaned_defs[key] = _clean_schema_for_vllm(defn)
        schema["$defs"] = cleaned_defs

    # Remove unsupported fields
    for field in ("title", "default"):
        schema.pop(field, None)

    return schema


def _build_sandbox_code(page_dir: str, num_pages: int) -> str:
    """Build sandbox code — batch_look with optional structured output."""
    return f'''
import os
import json as _json
import tempfile
from PIL import Image
from pydantic import BaseModel
from collections import Counter

# Load all pages as PIL Images
Image.MAX_IMAGE_PIXELS = 500_000_000
pages = []
for i in range({num_pages}):
    path = os.path.join({page_dir!r}, f"page_{{i}}.png")
    assert os.path.exists(path), f"Page image not found: {{path}}"
    pages.append(Image.open(path))
assert len(pages) == {num_pages}, f"Expected {{num_pages}} pages, got {{len(pages)}}"

def search(query, k=5):
    """BM25 search over OCR text. Returns list of {{page, score, text}} dicts."""
    return _search(query, k)

def batch_look(requests):
    """Send multiple images to the VLM in parallel. Supports structured output.

    Input: list of tuples, either:
      - (image, query)           -> returns str
      - (image, query, Model)    -> returns Model instance (structured output)

    When a Pydantic BaseModel subclass is provided as the third element,
    the VLM returns structured JSON matching the model's schema.
    The result is automatically parsed into a Model instance.

    Example — structured number reading with majority vote:
        class NumberReading(BaseModel):
            explanation: str
            value: float

        readings = batch_look([(crop, "Read the number", NumberReading)] * 3)
        values = [r.value for r in readings]
        answer = Counter(values).most_common(1)[0][0]

    Example — structured table extraction:
        class TableRow(BaseModel):
            item: str
            quantity: int

        class TableReading(BaseModel):
            explanation: str
            rows: list[TableRow]

        result = batch_look([(crop, "Extract all table rows", TableReading)])[0]
        total = sum(r.quantity for r in result.rows)

    Example — plain text (backward compatible):
        results = batch_look([(crop, "Describe what you see")])
        # results[0] is a plain string
    """
    paths = []
    schemas = []  # Track which requests have schemas and the model class
    for item in requests:
        if len(item) == 3:
            image, query, model_cls = item
            schema = model_cls.model_json_schema()
            schema_str = _json.dumps(schema)
        elif len(item) == 2:
            image, query = item
            model_cls = None
            schema_str = None
        else:
            raise ValueError(f"batch_look: expected (image, query) or (image, query, Model), got {{len(item)}} elements")

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp, format="PNG")
        tmp.close()

        entry = {{"path": tmp.name, "query": query}}
        if schema_str is not None:
            entry["schema"] = schema_str
        paths.append(entry)
        schemas.append(model_cls)

    raw_results = _batch_look_impl(_json.dumps(paths))

    # Reconstruct Pydantic models for structured results
    final_results = []
    for i, (raw, model_cls) in enumerate(zip(raw_results, schemas)):
        if model_cls is not None and isinstance(raw, dict):
            try:
                final_results.append(model_cls(**raw))
            except Exception as e:
                # If model construction fails, return raw dict
                final_results.append(raw)
        else:
            final_results.append(raw)

    return final_results
'''


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def _parse_answers(raw: str, expected_ids: set[str]) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}

    raw = str(raw).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except (json.JSONDecodeError, ValueError):
        pass

    json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError):
            pass

    result = {}
    for qid in expected_ids:
        pattern = re.compile(rf'["\']?{re.escape(qid)}["\']?\s*[:=]\s*["\']([^"\']+)["\']', re.IGNORECASE)
        m = pattern.search(raw)
        if m:
            result[qid] = m.group(1).strip()

    if result:
        return result

    if len(expected_ids) == 1:
        qid = next(iter(expected_ids))
        return {qid: raw}

    logger.warning("Could not parse batch answers: %s", raw[:200])
    return {}


# ---------------------------------------------------------------------------
# StructuredVLMProgram
# ---------------------------------------------------------------------------

class StructuredVLMProgram:
    """Structured VLM solver — batch_look with optional Pydantic schema support."""

    def __init__(
        self,
        vlm_lm: dspy.LM,
        iterations_per_question: int = 5,
        base_iterations: int = 10,
    ):
        self.vlm_lm = vlm_lm
        self.iterations_per_question = iterations_per_question
        self.base_iterations = base_iterations

    def solve_document(self, document: Document) -> tuple[dict[str, str], dict[str, list[dict]]]:
        """Solve all questions for a document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, img in enumerate(document.images):
                img.save(os.path.join(tmpdir, f"page_{i}.png"), format="PNG")

            search_index = None
            if document.page_texts:
                search_index = get_or_build_index(document.doc_id, document.page_texts)

            ctx = RunContext(
                page_dir=tmpdir,
                num_pages=len(document.images),
                search_index=search_index,
                page_texts=document.page_texts,
            )

            doc_info = f"Category: {document.doc_category}, Pages: {len(document.images)}"
            page_texts = _format_page_texts(document.page_texts) if document.page_texts else None

            num_q = len(document.questions)
            max_iter = self.base_iterations + self.iterations_per_question * num_q

            tips = get_category_tips(document.doc_category)
            instructions = TASK_INSTRUCTIONS + ("\n" + tips if tips else "")
            tools = _create_tools(self.vlm_lm, ctx)

            rlm = RLM(
                signature=_build_signature(instructions),
                max_iterations=max_iter,
                max_llm_calls=max_iter * 3,
                tools=tools,
                verbose=True,
                sandbox_code=_build_sandbox_code(tmpdir, len(document.images)),
            )
            logger.info("Structured VLM limits for %d questions: max_iterations=%d", num_q, max_iter)

            short_to_full = {}
            questions_list = []
            for i, q in enumerate(document.questions):
                short_id = f"q{i + 1}"
                short_to_full[short_id] = q.question_id
                questions_list.append({"question_id": short_id, "question": q.question})
            questions_json = json.dumps(questions_list)
            short_ids = set(short_to_full.keys())

            if page_texts is None:
                page_texts = ["[No OCR text available]"]

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
                    page_texts=page_texts,
                )

            with logfire.span(
                "solve_structured_vlm",
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
                    logger.warning("Structured VLM RLM failed for doc '%s': %s", document.doc_id, e)
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
                correct_count = 0
                scored_count = 0

                for q in document.questions:
                    answer = answers_dict.get(q.question_id, "Unknown")
                    if not answer or answer.strip() == "":
                        answer = "Unknown"
                    predictions[q.question_id] = answer
                    trajectories[q.question_id] = trajectory

                    if q.answer is not None:
                        scored_count += 1
                        is_correct, extracted = evaluate_prediction(answer, q.answer)
                        if is_correct:
                            correct_count += 1
                        with logfire.span(
                            "structured_vlm_question_result",
                            question_id=q.question_id,
                            question=q.question[:200],
                            is_correct=is_correct,
                            prediction=answer[:200],
                            ground_truth=q.answer[:200],
                            extracted_answer=extracted[:200],
                        ):
                            pass
                        logger.info(
                            "StrVLM Q %s: %s (GT=%s, PRED=%s)",
                            q.question_id,
                            "CORRECT" if is_correct else "WRONG",
                            q.answer[:40],
                            extracted[:40],
                        )

                if scored_count > 0:
                    batch_span.set_attribute("accuracy", correct_count / scored_count)
                    batch_span.set_attribute("correct", correct_count)
                    batch_span.set_attribute("scored_questions", scored_count)

            return predictions, trajectories


# ---------------------------------------------------------------------------
# Factory for hydra instantiation
# ---------------------------------------------------------------------------

def create_structured_vlm_program(
    iterations_per_question: int = 5,
    base_iterations: int = 10,
    vlm: dict[str, Any] | None = None,
) -> StructuredVLMProgram:
    vlm_config = LMConfig(
        model=vlm["model"],
        api_base=vlm.get("api_base"),
        api_key=vlm.get("api_key"),
        max_tokens=vlm.get("max_tokens", 65536),
        temperature=vlm.get("temperature", 1.0),
        vertex_location=vlm.get("vertex_location"),
    ) if vlm and vlm.get("model") else LMConfig()

    vlm_lm = vlm_config.to_dspy_lm()

    return StructuredVLMProgram(
        vlm_lm=vlm_lm,
        iterations_per_question=iterations_per_question,
        base_iterations=base_iterations,
    )
