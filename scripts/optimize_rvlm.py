"""GEPA optimization for the rvlm_unified scaffold.

Optimizes ONE prompt component: ``task_instructions`` (= TASK_BODY +
UNIFIED_TIPS merged, see ``rvlm_gepa_solver.SEED_TASK_INSTRUCTIONS``).
The dataset-specific answer-formatting rules are NOT optimized — they
remain profile-injected at runtime so GEPA cannot break answer parsing.

Train/val split: DocVQA-2026 val set, deterministic ~50/50 by
category. The val set has only **25 docs** (2-4 per category) so the
split lands at ~13 train / 12 val docs. Per-category breakdown:

- categories with 4 docs (comics, engineering_drawing, business_report)
  → 2 train / 2 val
- categories with 3 docs (maps, science_paper, slide) → 2 train / 1 val
- categories with 2 docs (infographics, science_poster) → 1 train / 1 val

That's small for prompt optimization — flagged to the user; if results
are noisy, the obvious next step is to mix in MP-DocVQA + MMLongBench
training samples.

Student LM: Qwen 3.5 27B at localhost:8927 (the model being optimized
for).

Reflection LM: ``vertex_ai/gemini-3-flash-preview`` (Gemini 3 Flash).
Stronger reflection model than the student is the canonical GEPA
recipe; Flash is the cheapest frontier-grade option we have available
and matches the archived ``optimize_flat_batch.py`` precedent of using
a Vertex Gemini for reflection.

Usage:
    uv run python scripts/optimize_rvlm.py
    uv run python scripts/optimize_rvlm.py --max-metric-calls 30
    uv run python scripts/optimize_rvlm.py --run-dir output/optim/rvlm-gepa-2026-05-28

Estimated wall: each candidate eval ≈ 40 docs × ~30s/doc on Qwen 27B
with c=8 ≈ 20min. 30 candidate evals ≈ 10h. Adjust --max-metric-calls
to your time budget.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
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

from docvqa.data import Document, load_documents
from docvqa.datasets.profile import get_profile
from docvqa.solvers.rvlm_gepa_solver import (
    SEED_TASK_INSTRUCTIONS,
    RvlmGepaProgram,
)
from docvqa.types import LMConfig

litellm.drop_params = True
litellm.request_timeout = 300

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET = "VLR-CVC/DocVQA-2026"
SPLIT = "val"
SPLIT_SEED = 42  # deterministic train/val split seed
# Per-category 50/50 with ceiling on train side (so 3-doc categories
# give 2/1 train/val rather than 1/2). 4-doc → 2/2, 3-doc → 2/1,
# 2-doc → 1/1. See module docstring for the resulting size.
TRAIN_FRACTION = 0.5

# Student LM = Qwen 3.5 27B at the same vllm the unified-tips chain
# uses, so the optimizer must wait for that chain to finish before
# launching (else vllm gets oversubscribed).
QWEN_27B_CONFIG = LMConfig(
    model="hosted_vllm/Qwen/Qwen3.5-27B",
    api_base="http://localhost:8927/v1",
    api_key="dummy",
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    enable_thinking=False,
    max_tokens=None,
)

# Reflection LM = Gemini 3 Flash via Vertex (cheapest frontier-grade
# option; matches archived ``optimize_flat_batch.py`` precedent of
# using a Vertex Gemini for reflection).
REFLECTION_LM_MODEL = "vertex_ai/gemini-3-flash-preview"

# ASI string formatter cap (per-question feedback lines).
MAX_FEEDBACK_QUESTIONS = 6


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------


def split_by_category(
    documents: list[Document],
    train_fraction: float = TRAIN_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[list[Document], list[Document]]:
    """Deterministic per-category split.

    For each category, sort docs by doc_id (reproducible order), shuffle
    with the seed, then take ``ceil(n_cat * train_fraction)`` into train
    and the remainder into val. Ceiling-on-train ensures every category
    is represented in train even when ``n_cat`` is odd.
    """
    import math

    rng = random.Random(seed)
    by_cat: dict[str, list[Document]] = defaultdict(list)
    for d in documents:
        by_cat[d.doc_category].append(d)

    train: list[Document] = []
    val: list[Document] = []
    for cat in sorted(by_cat):
        docs = sorted(by_cat[cat], key=lambda d: d.doc_id)
        rng.shuffle(docs)
        n_train = max(1, math.ceil(len(docs) * train_fraction))
        train.extend(docs[:n_train])
        val.extend(docs[n_train:])
    return train, val


# ---------------------------------------------------------------------------
# Evaluator (called by GEPA on each (candidate, example) pair)
# ---------------------------------------------------------------------------


def _format_feedback(doc: Document, predictions: dict[str, str], profile_score_fn) -> tuple[float, str]:
    """Score a doc + build per-question feedback string for ASI."""
    lines: list[str] = []
    correct = 0
    scored = 0
    detailed_lines: list[str] = []
    for q in doc.questions:
        if q.answer is None:
            continue
        scored += 1
        pred = predictions.get(q.question_id, "Unknown") or "Unknown"
        is_correct, extracted = profile_score_fn(pred, q.answer, q)
        if is_correct:
            correct += 1
            detailed_lines.append(
                f"  CORRECT: Q={q.question[:120]!r} -> {extracted[:80]!r}"
            )
        else:
            detailed_lines.append(
                f"  WRONG: Q={q.question[:120]!r} predicted={extracted[:80]!r} expected={q.answer[:80]!r}"
            )

    score = correct / scored if scored > 0 else 0.0
    lines.append(
        f"Document: {doc.doc_id} (category: {doc.doc_category}, {len(doc.images)} pages, {scored} scored q's)"
    )
    lines.append(f"Score: {correct}/{scored} ({score*100:.0f}%)")
    lines.append("")
    # Cap feedback lines to keep reflection prompt small on Qwen.
    if len(detailed_lines) > MAX_FEEDBACK_QUESTIONS:
        kept = detailed_lines[:MAX_FEEDBACK_QUESTIONS]
        kept.append(f"  ... ({len(detailed_lines) - MAX_FEEDBACK_QUESTIONS} more questions omitted)")
        lines.extend(kept)
    else:
        lines.extend(detailed_lines)
    return score, "\n".join(lines)


def make_evaluator(profile_name: str):
    """Return an evaluator closure suitable for ``optimize_anything``.

    Each example is a ``Document``. The evaluator builds an RvlmGepaProgram
    with the candidate's ``task_instructions``, runs ``solve_document``,
    scores answers with the dataset profile's ``score_fn``, and returns
    ``(score, side_info)`` where ``side_info["Feedback"]`` is the per-doc
    feedback string GEPA's reflection LM will see.
    """
    profile = get_profile(profile_name)
    vlm_lm = QWEN_27B_CONFIG.to_dspy_lm()
    student_lm = QWEN_27B_CONFIG.to_dspy_lm()

    def evaluate(candidate: dict[str, str], example: Document) -> tuple[float, dict[str, Any]]:
        program = RvlmGepaProgram(
            vlm_lm=vlm_lm,
            profile=profile,
            max_iterations=25,
            rlm_type="lean",
        )
        program.apply_candidate(candidate)

        try:
            with dspy.context(lm=student_lm):
                predictions, _trajectories = program.solve_document(example)
        except Exception as e:
            oa.log(f"ERROR: Solver failed for {example.doc_id}: {e}")
            return 0.0, {
                "Error": str(e),
                "Feedback": (
                    f"Solver crashed on {example.doc_id} ({example.doc_category}). "
                    "The task_instructions may have caused the agent to fail. "
                    "Check that instructions still describe tools (batch_look) and "
                    "the SUBMIT(answer=...) call format."
                ),
            }

        score, feedback = _format_feedback(example, predictions, profile.score_fn)
        oa.log(f"Doc {example.doc_id}: score={score:.2f}")
        return score, {"Feedback": feedback, "scores": {"accuracy": score}}

    return evaluate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GEPA optimization for rvlm_unified")
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=30,
        help="Total candidate evaluations across all proposed candidates",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Output dir; defaults to output/optim/rvlm-gepa-<UTC-ts>",
    )
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Parallel evaluator workers. Keep low (1-2) to avoid saturating Qwen vllm.",
    )
    parser.add_argument(
        "--reflection-minibatch-size",
        type=int,
        default=3,
        help="How many examples GEPA samples per reflection step.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    run_dir = args.run_dir or os.path.join(
        "output", "optim",
        f"rvlm-gepa-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    # Load docs and split.
    logger.info("Loading %s [%s]...", DATASET, SPLIT)
    documents = load_documents(DATASET, SPLIT)
    documents = [d for d in documents if any(q.answer is not None for q in d.questions)]
    logger.info("Loaded %d docs with GT", len(documents))

    train, val = split_by_category(documents, seed=args.seed)
    logger.info("Train: %d docs, Val: %d docs", len(train), len(val))
    # Sanity-check per-category counts so the split is paper-quotable.
    def _count(docs: list[Document]) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for d in docs:
            c[d.doc_category] += 1
        return dict(c)
    logger.info("Train per-cat: %s", _count(train))
    logger.info("Val   per-cat: %s", _count(val))

    # Persist the split so re-runs and writeups can quote exact doc_ids.
    split_path = os.path.join(run_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "train": [d.doc_id for d in train],
                "val": [d.doc_id for d in val],
            },
            f,
            indent=2,
        )

    # Seed candidate (one component).
    seed_candidate: dict[str, str] = {"task_instructions": SEED_TASK_INSTRUCTIONS}
    logger.info(
        "Seed candidate: task_instructions = %d chars",
        len(seed_candidate["task_instructions"]),
    )

    evaluate = make_evaluator(DATASET)

    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=args.max_metric_calls,
            run_dir=run_dir,
            seed=args.seed,
            parallel=True,
            max_workers=args.max_workers,
            display_progress_bar=True,
            use_cloudpickle=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=REFLECTION_LM_MODEL,
            reflection_minibatch_size=args.reflection_minibatch_size,
            module_selector="round_robin",
        ),
        tracking=TrackingConfig(
            use_wandb=not args.no_wandb,
            wandb_init_kwargs={"project": "docvqa", "entity": "bdsaglam"},
        ),
    )

    objective = (
        "Optimize the agent instructions for a Document Visual Question "
        "Answering scaffold. The agent reasons by writing Python code in a "
        "REPL with one tool: batch_look(requests) which sends "
        "(PIL_image, query) pairs to a VLM and returns answer strings. "
        "Pages are pre-loaded into a list `pages`. The agent must arrive "
        "at an answer and call SUBMIT(answer=...). Maximize per-document "
        "answer accuracy on DocVQA-2026."
    )
    background = (
        "There is exactly ONE optimizable component:\n"
        "- task_instructions: the agent's system prompt, which currently "
        "merges (a) generic 'how to use batch_look + how to reason' "
        "guidance and (b) per-category tips for all 8 DocVQA-2026 "
        "categories (business_report, comics, engineering_drawing, "
        "infographics, maps, science_paper, science_poster, slide).\n\n"
        "Hard constraints:\n"
        "- The agent's only visual tool is batch_look. Instructions MUST "
        "tell the agent it exists and how to call it.\n"
        "- The final answer is delivered by calling SUBMIT(answer=...). "
        "Instructions MUST preserve this convention.\n"
        "- DO NOT include answer-formatting rules (dates, units, etc.) "
        "in the optimized text — those are appended by the dataset "
        "profile at runtime and modifying them here has no effect.\n"
        "- 'Unknown' is the convention for 'information truly not in "
        "document'. Do not encourage giving up too easily — most failures "
        "are missed evidence, not absent evidence.\n"
        "- The agent has a turn budget. Encourage focused exploration "
        "(survey → locate → crop → verify) over exhaustive enumeration."
    )

    logger.info(
        "Starting GEPA optimization (max_metric_calls=%d, max_workers=%d)",
        args.max_metric_calls, args.max_workers,
    )
    result = optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=evaluate,
        dataset=train,
        valset=val,
        objective=objective,
        background=background,
        config=config,
    )

    logger.info("GEPA optimization complete.")
    best_score = result.val_aggregate_scores[result.best_idx]
    logger.info(
        "Best score: %.3f (candidate %d of %d)",
        best_score, result.best_idx, len(result.candidates),
    )

    out_path = os.path.join(run_dir, "best_candidate.json")
    with open(out_path, "w") as f:
        json.dump(result.best_candidate, f, indent=2)
    logger.info("Saved best candidate to %s", out_path)

    # Also dump a 1-line summary for quick scan.
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "best_idx": result.best_idx,
                "best_val_score": best_score,
                "num_candidates": len(result.candidates),
                "seed": args.seed,
                "train_size": len(train),
                "val_size": len(val),
                "max_metric_calls": args.max_metric_calls,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("OPTIMIZED task_instructions (first 400 chars)")
    print("=" * 60)
    best = result.best_candidate
    text = best.get("task_instructions", "") if isinstance(best, dict) else ""
    print(text[:400])
    if len(text) > 400:
        print(f"... ({len(text)} chars total; see {out_path})")


if __name__ == "__main__":
    main()
