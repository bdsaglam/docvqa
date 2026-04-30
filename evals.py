"""DocVQA 2026 evaluation pipeline.

Usage:
    uv run python evals.py                           # default: 5 val samples
    uv run python evals.py data.split="val[:10]"     # 10 val samples
    uv run python evals.py lm.model=vertex_ai/gemini-3-flash-preview # different model
    uv run python evals.py run_id=2026-03-14_12-00-00  # resume a run
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

from dotenv import load_dotenv

assert load_dotenv(), "Failed to load .env file"

from docvqa.obs import setup_observability

setup_observability()

import dspy
import hydra
import litellm
from omegaconf import DictConfig, OmegaConf

from docvqa.data import load_documents
from docvqa.runner import evaluate
from docvqa.types import LMConfig

litellm.drop_params = True
litellm.request_timeout = 300  # 5min timeout for all LLM calls


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Determine output directory
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    if cfg.get("run_id"):
        run_dir = Path("output/runs") / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Save config for reproducibility
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    # Configure LLM
    from docvqa.adapters import RetryJSONAdapter
    lm_cfg = LMConfig(**{k: v for k, v in OmegaConf.to_container(cfg.lm, resolve=True).items() if v is not None})
    lm = lm_cfg.to_dspy_lm()
    dspy.configure(lm=lm, adapter=RetryJSONAdapter(max_retries=3))
    print(f"LLM: {lm.model}")

    # Instantiate solver
    solver_cfg = OmegaConf.to_container(cfg.solver, resolve=True)
    target = solver_cfg.pop("_target_")
    # Import and instantiate
    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)

    # Copy solver source for reproducibility
    if module.__file__:
        shutil.copy2(module.__file__, run_dir / Path(module.__file__).name)
    solver_class = getattr(module, class_name)
    solver = solver_class(**solver_cfg)
    print(f"Solver: {target}")

    # Load data
    doc_ids = cfg.data.get("doc_ids")
    if doc_ids is not None:
        doc_ids = list(doc_ids)
    categories = cfg.data.get("categories")
    if categories is not None:
        categories = list(categories)
    print(f"Loading dataset: {cfg.data.dataset} [{cfg.data.split}] (doc_ids={doc_ids}, categories={categories})")
    documents = load_documents(cfg.data.dataset, cfg.data.split, doc_ids=doc_ids)
    if categories:
        documents = [d for d in documents if d.doc_category in categories]
    print(f"Loaded {len(documents)} documents with {sum(len(d.questions) for d in documents)} questions")

    # Run evaluation
    max_concurrency = cfg.get("max_concurrency", 1)
    task_timeout = cfg.get("task_timeout_seconds", 600)
    summary = evaluate(solver, documents, run_dir, max_concurrency=max_concurrency, task_timeout_seconds=task_timeout)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    if summary.get("overall_accuracy") is not None:
        print(f"Overall accuracy: {summary['overall_accuracy']:.1%} ({summary['correct']}/{summary['scored_questions']})")
        if "by_category" in summary:
            print("\nPer category:")
            for cat, stats in summary["by_category"].items():
                print(f"  {cat:20s}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    else:
        print("No scored questions (test set without ground truth?)")
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
