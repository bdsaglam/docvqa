"""GEPA optimization for the rvlm_unified scaffold.

Optimizes ONE prompt component: ``task_instructions`` (= TASK_BODY +
UNIFIED_TIPS merged, see ``rvlm_gepa_solver.SEED_TASK_INSTRUCTIONS``).
The dataset-specific answer-formatting rules are NOT optimized — they
remain profile-injected at runtime so GEPA cannot break answer parsing.

Training data: **MP-DocVQA + MMLongBench-Doc val samples** (the
pre-built 200q stratified subsamples shipped in
``data/{mp-docvqa,mmlongbench-doc}/val/sample_200q_doc_ids.txt``).
~61 docs / ~412 questions total. NO DocVQA-2026 docs in train —
generalization to DocVQA-2026 is the held-out signal.

Validation: full DocVQA-2026 val (25 docs / 80 questions). GEPA only
commits a new candidate if it improves the DocVQA-2026 val score,
which protects against the optimizer compressing away DocVQA-2026
category content that doesn't help on MP-DocVQA / MMLongBench.

Per-example scoring: each train doc is tagged with its dataset's
profile (ANLS for MP-DocVQA, Qwen judge for MMLongBench-Doc, ANLS for
DocVQA-2026 val). The evaluator dispatches scoring per-example.

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
from dataclasses import dataclass
from datetime import datetime, timezone
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

VAL_DATASET = "VLR-CVC/DocVQA-2026"
VAL_SPLIT = "val"
# Default training datasets (CLI: --train-datasets overrides).
DEFAULT_TRAIN_DATASETS = (
    "lmms-lab/MP-DocVQA",
    "yubo2333/MMLongBench-Doc",
)
SPLIT_SEED = 42  # deterministic ordering when datasets are sampled / shuffled

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


@dataclass
class TaggedDoc:
    """A document paired with the HF dataset slug it came from.

    The evaluator uses ``dataset`` to look up the profile (and therefore
    the score function and per-question hint) at scoring time. This lets
    a single GEPA run train on cross-dataset examples while keeping
    correct per-example scoring semantics.
    """
    document: Document
    dataset: str  # HF slug; key into get_profile()


def load_training_docs(dataset_slugs: list[str], seed: int = SPLIT_SEED) -> list[TaggedDoc]:
    """Load training docs from the listed datasets.

    Each dataset is loaded via :func:`load_documents`, which automatically
    honors the ``data/<dataset>/val/sample_200q_doc_ids.txt`` sample lists
    when present. Order: dataset-by-dataset (sorted by slug for
    reproducibility), then by doc_id within each dataset, then shuffled
    with the seed so GEPA's batch sampling sees a balanced mix.
    """
    rng = random.Random(seed)
    examples: list[TaggedDoc] = []
    for slug in sorted(dataset_slugs):
        logger.info("Loading training docs from %s ...", slug)
        docs = load_documents(slug, "val")
        docs = [d for d in docs if any(q.answer is not None for q in d.questions)]
        # Deterministic order within a dataset before global shuffle.
        docs.sort(key=lambda d: d.doc_id)
        logger.info("  %s: %d docs, %d Qs", slug, len(docs), sum(len(d.questions) for d in docs))
        examples.extend(TaggedDoc(document=d, dataset=slug) for d in docs)
    rng.shuffle(examples)
    return examples


def load_val_docs(dataset: str = VAL_DATASET, split: str = VAL_SPLIT) -> list[TaggedDoc]:
    """Load the held-out validation docs. Always DocVQA-2026 val."""
    docs = load_documents(dataset, split)
    docs = [d for d in docs if any(q.answer is not None for q in d.questions)]
    docs.sort(key=lambda d: d.doc_id)
    return [TaggedDoc(document=d, dataset=dataset) for d in docs]


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


def make_evaluator():
    """Return an evaluator closure suitable for ``optimize_anything``.

    Each example is a :class:`TaggedDoc`. The evaluator looks up the
    profile for ``example.dataset`` at call time, builds an
    ``RvlmGepaProgram`` with the candidate's ``task_instructions`` and
    that profile, runs ``solve_document``, scores answers with the
    profile's ``score_fn``, and returns ``(score, side_info)`` where
    ``side_info["Feedback"]`` is the per-doc feedback string GEPA's
    reflection LM will see.
    """
    vlm_lm = QWEN_27B_CONFIG.to_dspy_lm()
    student_lm = QWEN_27B_CONFIG.to_dspy_lm()

    def evaluate(candidate: dict[str, str], example: TaggedDoc) -> tuple[float, dict[str, Any]]:
        doc = example.document
        profile = get_profile(example.dataset)
        program = RvlmGepaProgram(
            vlm_lm=vlm_lm,
            profile=profile,
            max_iterations=25,
            rlm_type="lean",
        )
        program.apply_candidate(candidate)

        try:
            with dspy.context(lm=student_lm):
                predictions, _trajectories = program.solve_document(doc)
        except Exception as e:
            oa.log(f"ERROR: Solver failed for {doc.doc_id} ({example.dataset}): {e}")
            return 0.0, {
                "Error": str(e),
                "Feedback": (
                    f"Solver crashed on {doc.doc_id} (dataset={example.dataset}, "
                    f"category={doc.doc_category}). The task_instructions may "
                    f"have caused the agent to fail. Check that instructions "
                    f"still describe the batch_look tool and the "
                    f"SUBMIT(answer=...) call format."
                ),
            }

        score, feedback = _format_feedback(doc, predictions, profile.score_fn)
        feedback = f"[dataset={example.dataset}] " + feedback
        oa.log(f"Doc {doc.doc_id} (ds={example.dataset}): score={score:.2f}")
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
    parser.add_argument(
        "--train-datasets",
        type=str,
        default=",".join(DEFAULT_TRAIN_DATASETS),
        help="Comma-separated HF dataset slugs to use for training. "
             "Defaults to lmms-lab/MP-DocVQA + yubo2333/MMLongBench-Doc.",
    )
    args = parser.parse_args()
    train_datasets = [s.strip() for s in args.train_datasets.split(",") if s.strip()]

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

    # Load training docs from non-DocVQA datasets.
    logger.info("Loading training docs from: %s", train_datasets)
    train = load_training_docs(train_datasets, seed=args.seed)
    train_per_ds: dict[str, int] = defaultdict(int)
    train_q_per_ds: dict[str, int] = defaultdict(int)
    for ex in train:
        train_per_ds[ex.dataset] += 1
        train_q_per_ds[ex.dataset] += len(ex.document.questions)
    logger.info(
        "Train: %d docs / %d Qs total",
        len(train), sum(train_q_per_ds.values()),
    )
    for ds, n in train_per_ds.items():
        logger.info("  %s: %d docs, %d Qs", ds, n, train_q_per_ds[ds])

    # Validation: full DocVQA-2026 val (held-out from training).
    logger.info("Loading validation docs: %s [%s]", VAL_DATASET, VAL_SPLIT)
    val = load_val_docs()
    logger.info(
        "Val: %d docs / %d Qs",
        len(val), sum(len(ex.document.questions) for ex in val),
    )

    # Persist the split so re-runs and writeups can quote exact doc_ids.
    split_path = os.path.join(run_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "train_datasets": train_datasets,
                "train": [
                    {"dataset": ex.dataset, "doc_id": ex.document.doc_id}
                    for ex in train
                ],
                "val_dataset": VAL_DATASET,
                "val": [ex.document.doc_id for ex in val],
            },
            f,
            indent=2,
        )
    logger.info("Wrote split manifest to %s", split_path)

    # Seed candidate (one component).
    seed_candidate: dict[str, str] = {"task_instructions": SEED_TASK_INSTRUCTIONS}
    logger.info(
        "Seed candidate: task_instructions = %d chars",
        len(seed_candidate["task_instructions"]),
    )

    evaluate = make_evaluator()

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
        "at an answer and call SUBMIT(answer=...). The prompt is trained "
        "on MP-DocVQA + MMLongBench-Doc samples (multi-document and "
        "long-document benchmarks with no overlap with the val set) and "
        "validated on DocVQA-2026 val (single-doc, 8-category). The "
        "objective is to maximize answer accuracy on the held-out "
        "DocVQA-2026 val — i.e., produce a prompt that generalizes from "
        "the training distributions to DocVQA-2026."
    )
    background = (
        "There is exactly ONE optimizable component:\n"
        "- task_instructions: the agent's system prompt. The seed merges "
        "(a) generic 'how to use batch_look + how to reason' guidance and "
        "(b) per-category tips for all 8 DocVQA-2026 categories "
        "(business_report, comics, engineering_drawing, infographics, "
        "maps, science_paper, science_poster, slide). Training docs come "
        "from MP-DocVQA (mostly business / financial pages, up to ~20 "
        "pages) and MMLongBench-Doc (research papers, reports, decks; up "
        "to 80 pages). MP-DocVQA and MMLongBench docs do not have those "
        "8 categories — the agent should still benefit from any general "
        "guidance in the prompt.\n\n"
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
        "(survey → locate → crop → verify) over exhaustive enumeration.\n"
        "- Keep DocVQA-2026 category tips in the prompt even if they "
        "appear unused on the MP-DocVQA / MMLongBench training docs. The "
        "validation set IS DocVQA-2026 and those tips help there."
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
