"""Sweep iteration budgets on full val set with parallel configs.

Usage:
    uv run python scripts/sweep_iterations.py --lm qwen --vlm qwen --max_concurrency 16 --parallel_configs 4
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_config(base: int, per_q: int, lm: str, vlm: str, max_concurrency: int, doc_ids: list[str] | None = None) -> dict:
    """Run a single config as a subprocess."""
    run_id = f"sweep-{lm}-b{base}-pq{per_q}"
    log_file = f"/tmp/sweep-{run_id}.log"

    cmd = [
        "uv", "run", "python", "evals.py",
        f"solver=flat_batch",
        f"solver.iterations_per_question={per_q}",
        f"solver.base_iterations={base}",
        f"lm={lm}",
        f"vlm={vlm}",
        f"data.num_samples=null",
        f"max_concurrency={max_concurrency}",
        f"run_id={run_id}",
    ]
    if doc_ids:
        cmd.append(f"data.doc_ids=[{','.join(doc_ids)}]")

    print(f"[START] base={base} per_q={per_q} -> {run_id}")
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=7200)

    # Parse results
    results_file = Path(f"output/runs/{run_id}/results.json")
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        summary = data.get("summary", {})
        accuracy = summary.get("overall_accuracy", 0)
        correct = summary.get("correct", 0)
        total = summary.get("scored_questions", 0)
        by_cat = summary.get("by_category", {})
        print(f"[DONE]  base={base} per_q={per_q} -> {accuracy*100:.1f}% ({correct}/{total})")
        return {
            "base": base,
            "per_q": per_q,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "by_category": by_cat,
            "run_id": run_id,
            "exit_code": result.returncode,
        }
    else:
        print(f"[FAIL]  base={base} per_q={per_q} -> no results (exit={result.returncode})")
        return {
            "base": base,
            "per_q": per_q,
            "accuracy": 0,
            "correct": 0,
            "total": 0,
            "run_id": run_id,
            "exit_code": result.returncode,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", default="qwen")
    parser.add_argument("--vlm", default="qwen")
    parser.add_argument("--base_iterations", nargs="+", type=int, default=[1, 2, 4, 6])
    parser.add_argument("--per_question", nargs="+", type=int, default=[2, 3, 5])
    parser.add_argument("--doc_ids", nargs="+", default=None, help="Specific doc IDs (default: full val)")
    parser.add_argument("--max_concurrency", type=int, default=16, help="Doc concurrency per config")
    parser.add_argument("--parallel_configs", type=int, default=3, help="How many configs to run in parallel")
    args = parser.parse_args()

    configs = [(b, pq) for b in args.base_iterations for pq in args.per_question]
    print(f"LM: {args.lm}, VLM: {args.vlm}")
    print(f"Configs: {len(configs)} (base × per_q = {args.base_iterations} × {args.per_question})")
    print(f"Doc concurrency: {args.max_concurrency}, Parallel configs: {args.parallel_configs}")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=args.parallel_configs) as pool:
        futures = {
            pool.submit(run_config, base, per_q, args.lm, args.vlm, args.max_concurrency, args.doc_ids): (base, per_q)
            for base, per_q in configs
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                base, per_q = futures[future]
                print(f"[ERROR] base={base} per_q={per_q}: {e}")
                results.append({"base": base, "per_q": per_q, "accuracy": 0, "correct": 0, "total": 0})

    # Summary
    print("\n" + "=" * 60)
    print("SWEEP RESULTS")
    print("=" * 60)
    print(f"{'base':>5s} {'per_q':>5s} {'max_iter(2Q)':>12s} {'accuracy':>8s} {'score':>8s}")
    print("-" * 45)
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        max_iter_2q = r["base"] + r["per_q"] * 2
        print(f"{r['base']:5d} {r['per_q']:5d} {max_iter_2q:12d} {r['accuracy']*100:7.1f}% {r['correct']:3d}/{r['total']}")

    # Save results
    out_path = Path("output/sweep_iterations.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
