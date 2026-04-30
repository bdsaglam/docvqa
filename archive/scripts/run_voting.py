"""Run ensemble voting with multiple solver copies.

Usage:
    uv run python scripts/run_voting.py --doc_ids comics_1 --num_votes 3
"""

import argparse

import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = 500_000_000

from dotenv import load_dotenv

assert load_dotenv()

from docvqa.obs import setup_observability

setup_observability()

import dspy
from dspy.adapters import JSONAdapter

from docvqa.data import load_documents
from docvqa.metrics import evaluate_prediction
from docvqa.solvers.ensemble_solver import EnsembleSolver
from docvqa.solvers.flat_solo_solver import create_flat_solo_program
from docvqa.types import LMConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_ids", nargs="+", required=True)
    parser.add_argument("--num_votes", type=int, default=3)
    parser.add_argument("--lm", default="qwen", choices=["flash", "pro", "qwen"])
    parser.add_argument("--vlm", default="qwen", choices=["flash", "qwen"])
    parser.add_argument("--max_parallel", type=int, default=5)
    args = parser.parse_args()

    # Load LM configs
    lm_configs = {
        "flash": {
            "model": "vertex_ai/gemini-3-flash-preview",
            "vertex_location": "global",
        },
        "pro": {"model": "vertex_ai/gemini-3-pro-preview", "vertex_location": "global"},
        "qwen": {
            "model": "hosted_vllm/Qwen/Qwen3.5-27B",
            "api_base": "http://localhost:8927/v1",
            "api_key": "dummy",
        },
    }
    vlm_configs = {
        "flash": {"model": "vertex_ai/gemini-3-flash-preview", "max_tokens": 16384},
        "qwen": {
            "model": "hosted_vllm/Qwen/Qwen3.5-27B",
            "api_base": "http://localhost:8927/v1",
            "api_key": "dummy",
            "max_tokens": 8192,
        },
    }

    lm_cfg = LMConfig(**{k: v for k, v in lm_configs[args.lm].items()})
    lm = lm_cfg.to_dspy_lm()
    dspy.configure(lm=lm, adapter=JSONAdapter())
    print(f"LLM: {args.lm}, VLM: {args.vlm}, Votes: {args.num_votes}")

    # Create N identical solvers
    solvers = []
    for _ in range(args.num_votes):
        solver = create_flat_solo_program(vlm=vlm_configs[args.vlm], rlm_type="lean")
        solvers.append(solver)

    ensemble = EnsembleSolver(solvers=solvers, max_parallel=args.max_parallel)

    # Load docs
    documents = load_documents("VLR-CVC/DocVQA-2026", "val", doc_ids=args.doc_ids)
    print(f"Loaded {len(documents)} documents")

    for doc in documents:
        print(
            f"\nSolving {doc.doc_id} ({doc.doc_category}, {len(doc.questions)} questions)"
        )
        preds, trajs = ensemble.solve_document(doc)

        correct = 0
        total = 0
        for q in doc.questions:
            if q.answer is not None:
                total += 1
                is_correct, extracted = evaluate_prediction(
                    preds[q.question_id], q.answer
                )
                if is_correct:
                    correct += 1
                mark = "OK" if is_correct else "XX"
                print(
                    f"  {mark} pred={preds[q.question_id][:60]:60s} gt={q.answer[:40]}"
                )

        if total > 0:
            print(f"  {doc.doc_id} -> {correct / total * 100:.0f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
