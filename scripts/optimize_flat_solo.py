"""GEPA optimization for flat_solo DocVQA solver via optimize_anything.

Optimizes 9 prompt components (Scope A):
- task_instructions: main RLM agent prompt
- tip_<8 categories>: per-category tips

Student: local Qwen 3.5 27B (vLLM at localhost:8927) for both LLM and VLM roles.
Reflection: Gemini 3.1 Pro (vertex_ai/gemini-3_1-pro-preview).

Each example = one document (pre-computed: pages saved as PNGs, BM25 index built).
ASI = per-question correct/wrong + last reasoning snippet for failures + per-component focused feedback.

Train: 17 val docs (~61 questions). Held-out val: 8 docs (1/category, ~19 questions).
After optimization, top-3 candidates should be re-evaluated on full 25-doc val
via `evals.py solver=flat_solo_gepa solver.candidate_path=...`.

Usage:
    uv run python scripts/optimize_flat_solo.py
    uv run python scripts/optimize_flat_solo.py --max-metric-calls 60   # short smoke
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

assert load_dotenv(), "Failed to load .env file"

from docvqa.obs import setup_observability

setup_observability()

import dspy
import gepa.optimize_anything as oa
import litellm
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    TrackingConfig,
    optimize_anything,
)

from docvqa.adapters import RetryJSONAdapter
from docvqa.data import Document, load_documents
from docvqa.metrics import evaluate_prediction, get_anls
from docvqa.search import get_or_build_index
from docvqa.solvers.flat_solo_gepa_solver import (
    FlatSoloGepaProgram,
    _format_page_texts,
    build_seed_candidate,
    candidate_to_overrides,
)
from docvqa.types import LMConfig

litellm.drop_params = True
litellm.request_timeout = 300

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Held-out val docs — 1 per category, deterministic
# Picked to favor smaller documents (fewer pages) so val eval is faster
# while keeping meaningful question count.
# ---------------------------------------------------------------------------
HELDOUT_DOC_IDS: set[str] = {
    "business_report_3",       # 89p, 2q
    "comics_4",                # 69p, 2q
    "engineering_drawing_2",   # 1p, 2q
    "infographics_1",          # 1p, 2q
    "maps_3",                  # 1p, 3q
    "science_paper_2",         # 19p, 1q
    "science_poster_1",        # 1p, 5q
    "slide_1",                 # 36p, 2q
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Student LLM (the RLM agent). Matches configs/lm/qwen-3_5-27b-vllm-local.yaml
# but with enable_thinking=False (proven best for flat-solo).
STUDENT_LM_CONFIG = LMConfig(
    model="hosted_vllm/Qwen/Qwen3.5-27B",
    api_base="http://localhost:8927/v1",
    api_key="dummy",
    max_tokens=None,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    enable_thinking=False,
)

VLM_CONFIG = LMConfig(
    model="hosted_vllm/Qwen/Qwen3.5-27B",
    api_base="http://localhost:8927/v1",
    api_key="dummy",
    max_tokens=16384,
    temperature=0.3,
    top_k=20,
    enable_thinking=False,
)

REFLECTION_MODEL = "vertex_ai/gemini-3.1-pro-preview"


def make_reflection_lm(model_name: str = REFLECTION_MODEL):
    """Custom LM callable for GEPA reflection that passes Vertex-specific kwargs.

    GEPA's default `make_litellm_lm` only passes (model, messages) and skips
    vertex_location, reasoning_effort, etc. — which causes 400 errors on
    "global" Vertex endpoints. This wrapper passes the full set.
    """
    import litellm as _litellm

    def _lm(prompt):
        messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        )
        completion = _litellm.completion(
            model=model_name,
            messages=messages,
            vertex_location="global",
            reasoning_effort="high",
            temperature=1.0,
        )
        return completion.choices[0].message.content

    return _lm


# ---------------------------------------------------------------------------
# Dataset prep
# ---------------------------------------------------------------------------


def prepare_document_workspace(doc: Document, workspace_root: Path) -> dict[str, Any]:
    """Save page images and build BM25 index once per doc.

    Uses absolute paths so the subprocess REPL (which may have a different
    cwd than the optimization script) can find the images.

    Returns the example dict consumed by the evaluator.
    """
    page_dir = (workspace_root / doc.doc_id).resolve()
    page_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(doc.images):
        out = page_dir / f"page_{i}.png"
        if not out.exists():
            img.save(out, format="PNG")

    search_index = None
    if doc.page_texts:
        search_index = get_or_build_index(doc.doc_id, doc.page_texts)

    page_texts_formatted = (
        _format_page_texts(doc.page_texts) if doc.page_texts else ["[No OCR text available]"]
    )

    return {
        "doc_id": doc.doc_id,
        "document": doc,
        "page_dir": str(page_dir),
        "search_index": search_index,
        "page_texts_formatted": page_texts_formatted,
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _last_reasoning_snippet(trajectory: list[dict] | None, max_chars: int = 300) -> str:
    """Return last step's reasoning, truncated."""
    if not trajectory:
        return ""
    last = trajectory[-1]
    reasoning = (last.get("reasoning") or "").strip()
    if not reasoning:
        # Fall back to second-to-last if the last had no reasoning
        if len(trajectory) >= 2:
            reasoning = (trajectory[-2].get("reasoning") or "").strip()
    if not reasoning:
        return ""
    if len(reasoning) > max_chars:
        reasoning = reasoning[:max_chars].rstrip() + "..."
    return reasoning


def _classify_failure(extracted: str, anls: float) -> str:
    e = extracted.lower().strip()
    if e in ("unknown", "unkown", ""):
        return "gave up (answer exists in document)"
    if anls > 0.8:
        return f"close (ANLS={anls:.2f}, likely format/case mismatch)"
    if anls > 0.5:
        return f"close but imprecise (ANLS={anls:.2f})"
    return f"wrong answer (ANLS={anls:.2f})"


# Per-document timeout (seconds). Hard upper bound to prevent a single
# stuck VLM call from blocking the entire optimization run.
PER_DOC_TIMEOUT_SECONDS = 1800  # 30 min


def evaluate(candidate: dict[str, str], example: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Evaluate candidate on a single document. Returns (accuracy, side_info)."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

    doc: Document = example["document"]
    task_instructions, tips_overrides = candidate_to_overrides(candidate)

    vlm_lm = VLM_CONFIG.to_dspy_lm()

    program = FlatSoloGepaProgram(
        vlm_lm=vlm_lm,
        task_instructions=task_instructions,
        tips_overrides=tips_overrides,
        max_iterations=30,
        rlm_type="lean",
        page_factor=1.5,
        question_concurrency=2,
    )

    def _solve():
        return program.solve_document(
            doc,
            precomputed={
                "page_dir": example["page_dir"],
                "search_index": example["search_index"],
                "page_texts_formatted": example["page_texts_formatted"],
            },
        )

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"solve-{doc.doc_id}")
    future = pool.submit(_solve)
    try:
        predictions, trajectories = future.result(timeout=PER_DOC_TIMEOUT_SECONDS)
        pool.shutdown(wait=False)
    except FutureTimeoutError:
        # Don't wait for the hung worker thread — leak it. GEPA continues; the
        # leaked thread will eventually die when its hung HTTP call timeouts.
        pool.shutdown(wait=False)
        oa.log(
            f"TIMEOUT: Solver exceeded {PER_DOC_TIMEOUT_SECONDS}s on {doc.doc_id}; "
            f"likely a hung VLM call. Returning 0.0."
        )
        return 0.0, {
            "scores": {"accuracy": 0.0},
            "Error": f"per-doc timeout ({PER_DOC_TIMEOUT_SECONDS}s)",
            "Feedback": (
                f"Solver timed out on {doc.doc_id} (category: {doc.doc_category}). "
                "Likely a hung VLM call under heavy concurrency. "
                "Not necessarily a candidate flaw."
            ),
        }
    except Exception as e:
        pool.shutdown(wait=False)
        oa.log(f"ERROR: Solver failed for {doc.doc_id}: {e}")
        return 0.0, {
            "scores": {"accuracy": 0.0},
            "Error": f"{type(e).__name__}: {e}",
            "Feedback": (
                f"Solver crashed on {doc.doc_id} (category: {doc.doc_category}). "
                "The candidate may have caused the agent to fail (broken syntax, "
                "missing SUBMIT instruction, incompatible format, etc.)."
            ),
        }

    correct_count = 0
    total_scored = 0
    question_lines: list[str] = []
    answered_count = 0

    for q in doc.questions:
        if q.answer is None:
            continue
        total_scored += 1
        raw = predictions.get(q.question_id, "Unknown")
        if not raw or not raw.strip():
            raw = "Unknown"
        else:
            answered_count += 1

        # evaluate_prediction expects "FINAL ANSWER:" prefix or raw text — handle both
        candidate_pred = raw if raw.startswith("FINAL ANSWER:") else f"FINAL ANSWER: {raw}"

        is_correct, extracted = evaluate_prediction(candidate_pred, q.answer)
        anls = get_anls(extracted.lower().strip(), q.answer.lower().strip())

        if is_correct:
            correct_count += 1
            question_lines.append(
                f"  CORRECT: Q='{q.question[:120]}' -> '{extracted[:80]}'"
            )
        else:
            issue = _classify_failure(extracted, anls)
            line = (
                f"  WRONG: Q='{q.question[:120]}' "
                f"predicted='{extracted[:80]}' expected='{q.answer[:80]}' — {issue}"
            )
            question_lines.append(line)
            snippet = _last_reasoning_snippet(trajectories.get(q.question_id))
            if snippet:
                question_lines.append(f"    Last reasoning: \"{snippet}\"")

    score = correct_count / total_scored if total_scored > 0 else 0.0

    feedback_header = [
        f"Document: {doc.doc_id} (category: {doc.doc_category}, "
        f"{len(doc.images)} pages, {len(doc.questions)} questions)",
        f"Score: {correct_count}/{total_scored} ({score * 100:.0f}%)",
        f"Answered: {answered_count}/{total_scored}",
        "",
    ]
    feedback_str = "\n".join(feedback_header + question_lines)

    side_info: dict[str, Any] = {
        "scores": {"accuracy": score},
        "Feedback": feedback_str,
    }

    # Per-component focused feedback for the matching tip_<category>
    tip_key = f"tip_{doc.doc_category}"
    side_info[f"{tip_key}_specific_info"] = {
        "Feedback": (
            f"Category: {doc.doc_category}\n"
            f"Score with current tips: {correct_count}/{total_scored} ({score * 100:.0f}%)\n\n"
            + "\n".join(question_lines)
        ),
    }

    oa.log(f"Doc {doc.doc_id}: {correct_count}/{total_scored} = {score * 100:.0f}%")
    for line in question_lines:
        oa.log(line)

    return score, side_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="GEPA optimization for flat-solo DocVQA")
    parser.add_argument("--max-metric-calls", type=int, default=150)
    parser.add_argument("--run-dir", type=str, default="output/gepa-flat-solo")
    parser.add_argument("--run-name", type=str, default="run-1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--reflection-minibatch-size", type=int, default=3)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 5 train docs, 3 val docs, 10 metric calls",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Configure global student LM (RLM agent default). VLM is set per-call via
    # dspy.context(lm=vlm_lm) inside the solver's look()/batch_look() tools.
    student_lm = STUDENT_LM_CONFIG.to_dspy_lm()
    dspy.configure(lm=student_lm, adapter=RetryJSONAdapter(max_retries=3))
    logger.info("Configured student LM: %s", student_lm.model)

    run_dir = Path(args.run_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace_root = run_dir / "doc_workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    # Load val with ground truth
    logger.info("Loading val dataset...")
    documents = load_documents("VLR-CVC/DocVQA-2026", "val")
    documents = [d for d in documents if any(q.answer is not None for q in d.questions)]
    logger.info("Loaded %d documents with ground truth", len(documents))

    # Pre-compute workspaces
    logger.info("Preparing document workspaces...")
    all_examples: list[dict[str, Any]] = []
    for doc in documents:
        all_examples.append(prepare_document_workspace(doc, workspace_root))

    # Train / val split
    train_examples = [ex for ex in all_examples if ex["doc_id"] not in HELDOUT_DOC_IDS]
    val_examples = [ex for ex in all_examples if ex["doc_id"] in HELDOUT_DOC_IDS]
    logger.info(
        "Train: %d docs / %d questions; Val: %d docs / %d questions",
        len(train_examples),
        sum(len(ex["document"].questions) for ex in train_examples),
        len(val_examples),
        sum(len(ex["document"].questions) for ex in val_examples),
    )

    if args.smoke:
        train_examples = train_examples[:5]
        val_examples = val_examples[:3]
        logger.warning("Smoke mode: train=%d val=%d", len(train_examples), len(val_examples))

    for ex in val_examples:
        d = ex["document"]
        logger.info(
            "  VAL: %s (%s, %d pages, %d questions)",
            d.doc_id, d.doc_category, len(d.images), len(d.questions),
        )

    # Seed candidate
    seed_candidate = build_seed_candidate()
    logger.info("Seed candidate components (%d):", len(seed_candidate))
    for name, text in seed_candidate.items():
        logger.info("  %s: %d chars", name, len(text))

    # GEPA config
    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=args.max_metric_calls,
            run_dir=str(run_dir),
            seed=args.seed,
            parallel=True,
            max_workers=args.max_workers,
            display_progress_bar=True,
            use_cloudpickle=True,
            cache_evaluation=True,
            raise_on_exception=False,
        ),
        reflection=ReflectionConfig(
            reflection_lm=make_reflection_lm(),
            reflection_minibatch_size=args.reflection_minibatch_size,
            module_selector="round_robin",
        ),
        tracking=TrackingConfig(
            use_wandb=True,
            wandb_init_kwargs={
                "project": "docvqa",
                "entity": "bdsaglam",
                "name": f"gepa-flat-solo-{args.run_name}",
            },
        ),
    )

    objective = (
        "Optimize the prompts driving a Document Visual Question Answering agent. "
        "The agent (an LLM in a Python REPL) writes code that calls vision tools "
        "(look, batch_look, search) to answer questions about documents in 8 categories: "
        "business reports, comics, engineering drawings, infographics, maps, science papers, "
        "science posters, presentation slides. Maximize answer accuracy (ANLS-graded). "
        "Answers must follow strict format rules (YYYY-MM-DD dates, % attached to numbers, "
        "no thousands separators, etc.) — those are documented in the prompt's "
        "ANSWER FORMATTING RULES section and should be preserved verbatim."
    )

    background = (
        "Optimizable components (round-robin reflection):\n"
        "- task_instructions: main RLM prompt — DATA / TOOLS / APPROACH / GUIDELINES sections.\n"
        "  Must keep: SUBMIT(answer=\"...\") syntax (one positional name=value pair), the\n"
        "  ANSWER FORMATTING RULES block at the end, the tool descriptions for look/batch_look/search.\n"
        "- tip_<category>: short list of category-specific tips appended after task_instructions\n"
        "  when the document's category matches. 8 categories total.\n\n"
        "Constraints:\n"
        "- Answers must match strict format (dates as YYYY-MM-DD, numbers without commas, % with no space).\n"
        "- Agent has limited iterations (30 base + page bonus); efficiency matters.\n"
        "- VLM calls are not free — crop tightly before querying for fine details.\n"
        "- OCR text is available but may be inaccurate; visual verification is important.\n"
        "- 'Unknown' is the correct answer ONLY when the information genuinely doesn't exist in the document; "
        "  about 7.5% of questions are intentionally unanswerable.\n"
        "- The agent is a local Qwen 3.5 27B (no thinking) — keep instructions clear and concrete; "
        "  it follows specific imperatives better than abstract advice."
    )

    logger.info("Starting GEPA optimization (max_metric_calls=%d)...", args.max_metric_calls)
    result = optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=evaluate,
        dataset=train_examples,
        valset=val_examples,
        objective=objective,
        background=background,
        config=config,
    )

    logger.info("GEPA optimization complete.")
    best_score = result.val_aggregate_scores[result.best_idx]
    logger.info(
        "Best val score: %.3f (candidate %d of %d)",
        best_score, result.best_idx, len(result.candidates),
    )

    out_path = run_dir / "best_candidate.json"
    out_path.write_text(json.dumps(result.best_candidate, indent=2))
    logger.info("Saved best candidate to %s", out_path)

    # Save top-3 by val score for downstream evaluation
    scored = sorted(
        enumerate(result.val_aggregate_scores), key=lambda kv: kv[1], reverse=True
    )[:3]
    top3_path = run_dir / "top3_candidates.json"
    top3 = [
        {"idx": idx, "score": score, "candidate": result.candidates[idx]}
        for idx, score in scored
    ]
    top3_path.write_text(json.dumps(top3, indent=2))
    logger.info("Saved top-3 candidates to %s (scores: %s)", top3_path, [round(s, 3) for _, s in scored])

    print("\n" + "=" * 60)
    print("OPTIMIZED COMPONENTS (best)")
    print("=" * 60)
    for name, text in result.best_candidate.items():
        print(f"\n--- {name} ({len(text)} chars) ---")
        print(text[:300])
        if len(text) > 300:
            print(f"... ({len(text)} chars total)")


if __name__ == "__main__":
    main()
