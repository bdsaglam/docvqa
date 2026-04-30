"""Generate an experiment results report from a list of run IDs.

Usage:
    python scripts/report.py run_id1 run_id2 ...
    python scripts/report.py flat-full-pro-flash-v2 full-val-pro
    # Or discover all runs:
    python scripts/report.py --all
    # Filter by minimum questions and recency:
    python scripts/report.py --all --min-questions 80 --recent 7
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

RUNS_DIR = Path(__file__).resolve().parent.parent / "output" / "runs"

BASELINES = {
    "Gemini 3 Pro": 0.375,
    "GPT-5.2": 0.350,
    "Gemini 3 Flash": 0.3375,
    "GPT-5 Mini": 0.225,
}

MODEL_SHORT = {
    "vertex_ai/gemini-3-pro-preview": "Pro",
    "vertex_ai/gemini-3.1-pro-preview": "Pro 3.1",
    "vertex_ai/gemini-3-flash-preview": "Flash",
    "hosted_vllm/Qwen/Qwen3.5-27B": "Qwen 27B",
}

SOLVER_SHORT = {
    "docvqa.solvers.flat_solo_solver.create_flat_solo_program": "Flat Solo",
    "docvqa.solvers.leanest_solo_solver.create_leanest_solo_program": "Leanest Solo",
    "docvqa.solvers.lean_solo_solver.create_lean_solo_program": "Lean Solo",
    "docvqa.solvers.flat_batch_solver.create_flat_batch_program": "Flat Batch",
    "docvqa.solvers.routing_solver.create_routing_solver": "Routing",
    "docvqa.solvers.rvlm_solver.create_rvlm_program": "RVLM",
}

MODEL_SHORT["gemini/gemma-4-31b-it"] = "Gemma4 31B"


def short_model(model: str) -> str:
    return MODEL_SHORT.get(model, model)


def short_solver(target: str) -> str:
    return SOLVER_SHORT.get(target, target.split(".")[-1])


def resolve_vlm(config: dict) -> str:
    """Resolve VLM model from config, handling ${vlm} / ${page_agent.vlm} references."""
    solver = config.get("solver", {})

    # Get the raw solver.vlm value
    solver_vlm = solver.get("vlm", {})

    if isinstance(solver_vlm, str):
        if solver_vlm == "${vlm}":
            return (config.get("vlm") or {}).get("model", "?")
        if solver_vlm == "${page_agent.vlm}":
            pa = config.get("page_agent", {})
            return (pa.get("vlm") or {}).get("model", "?") if isinstance(pa, dict) else "?"

    if isinstance(solver_vlm, dict):
        return solver_vlm.get("model", "?")

    # Fallback: top-level vlm, then page_agent.vlm
    top = config.get("vlm")
    if isinstance(top, dict) and "model" in top:
        return top["model"]

    pa = config.get("page_agent", {})
    if isinstance(pa, dict):
        pa_vlm = pa.get("vlm", {})
        if isinstance(pa_vlm, dict) and "model" in pa_vlm:
            return pa_vlm["model"]

    return "?"


def load_run(run_id: str) -> dict | None:
    run_dir = RUNS_DIR / run_id
    rfile = run_dir / "results.json"
    cfile = run_dir / "config.yaml"

    if not rfile.exists():
        print(f"WARNING: no results.json for {run_id}", file=sys.stderr)
        return None

    results = json.loads(rfile.read_text())
    config = yaml.safe_load(cfile.read_text()) if cfile.exists() else {}

    solver_target = config.get("solver", {}).get("_target_", "?")
    lm_model = config.get("lm", config.get("solver", {}).get("lm", {})).get("model", "?")
    vlm_model = resolve_vlm(config)

    return {
        "run_id": run_id,
        "solver": short_solver(solver_target),
        "llm": short_model(lm_model),
        "vlm": short_model(vlm_model),
        "total": results["summary"]["total_questions"],
        "correct": results["summary"]["correct"],
        "accuracy": results["summary"]["overall_accuracy"],
        "by_category": results["summary"].get("by_category", {}),
    }


def discover_runs(min_questions: int = 0, recent_days: int | None = None) -> list[str]:
    """Discover all run IDs with results, optionally filtered."""
    from datetime import datetime, timedelta

    cutoff = datetime.now() - timedelta(days=recent_days) if recent_days else None
    runs = []

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        rfile = run_dir / "results.json"
        if not rfile.exists():
            continue

        if min_questions > 0 or cutoff:
            results = json.loads(rfile.read_text())
            if results["summary"]["total_questions"] < min_questions:
                continue
            if cutoff and datetime.fromtimestamp(run_dir.stat().st_mtime) < cutoff:
                continue

        runs.append(run_dir.name)

    return runs


def fmt_pct(correct: int, total: int) -> str:
    return f"{correct}/{total} = {correct / total * 100:.1f}%"


def generate_report(runs: list[dict]) -> str:
    lines: list[str] = []

    # --- Baselines ---
    lines.append("# Experiment Results — DocVQA 2026\n")
    lines.append("## Official Baselines\n")
    lines.append("| Model | Overall |")
    lines.append("|-------|---------|")
    for name, acc in BASELINES.items():
        bold = "**" if acc == max(BASELINES.values()) else ""
        lines.append(f"| {name} | {bold}{acc * 100:.2f}%{bold} |")
    lines.append("")

    # --- Model shorthand ---
    used_models = set()
    for r in runs:
        used_models.add(r["llm"])
        used_models.add(r["vlm"])

    if used_models - set(MODEL_SHORT.values()):
        lines.append("## Model Shorthand\n")
        lines.append("| Short | Full |")
        lines.append("|-------|------|")
        for full, short in MODEL_SHORT.items():
            if short in used_models:
                lines.append(f"| {short} | `{full}` |")
        lines.append("")

    # --- Summary table ---
    lines.append("## Results\n")
    lines.append("| # | Run | Solver | LLM | VLM | Score |")
    lines.append("|---|-----|--------|-----|-----|-------|")

    sorted_runs = sorted(runs, key=lambda r: (-r["accuracy"], -r["total"]))
    best = sorted_runs[0]["accuracy"] if sorted_runs else 0

    for i, r in enumerate(sorted_runs, 1):
        bold = "**" if r["accuracy"] == best else ""
        score_str = f"{bold}{fmt_pct(r['correct'], r['total'])}{bold}"
        lines.append(f"| {i} | {r['run_id']} | {r['solver']} | {r['llm']} | {r['vlm']} | {score_str} |")
    lines.append("")

    # --- Per-category table ---
    # Collect all categories across runs
    all_cats = sorted(set(c for r in runs for c in r["by_category"]))
    if not all_cats:
        return "\n".join(lines)

    lines.append("## Per-Category Breakdown\n")

    # Column headers
    header = "| Category |"
    sep = "|----------|"
    for r in sorted_runs:
        header += f" {r['run_id']} |"
        sep += ":---:|"
    lines.append(header)
    lines.append(sep)

    for cat in all_cats:
        row = f"| {cat} |"
        for r in sorted_runs:
            c = r["by_category"].get(cat, {})
            if c:
                row += f" {c['correct']}/{c['total']} |"
            else:
                row += " — |"
        lines.append(row)

    # Overall row
    overall = "| **Overall** |"
    for r in sorted_runs:
        overall += f" **{fmt_pct(r['correct'], r['total'])}** |"
    lines.append(overall)
    lines.append("")

    # --- Best per-category ---
    lines.append("### Best per-category\n")
    lines.append("| Category | Best Score | Runs |")
    lines.append("|----------|:----------:|------|")

    for cat in all_cats:
        best_acc = 0
        best_runs = []
        for r in sorted_runs:
            c = r["by_category"].get(cat, {})
            if c and c.get("accuracy", 0) > best_acc:
                best_acc = c["accuracy"]
                best_runs = [r["run_id"]]
            elif c and c.get("accuracy", 0) == best_acc and best_acc > 0:
                best_runs.append(r["run_id"])

        if best_acc > 0:
            lines.append(f"| {cat} | {best_acc * 100:.0f}% | {', '.join(best_runs)} |")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate experiment results report")
    parser.add_argument("run_ids", nargs="*", help="Run IDs to include")
    parser.add_argument("--all", action="store_true", help="Discover all runs with results")
    parser.add_argument("--min-questions", type=int, default=0, help="Minimum question count filter (with --all)")
    parser.add_argument("--recent", type=int, default=None, help="Only runs from last N days (with --all)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    if args.all:
        run_ids = discover_runs(min_questions=args.min_questions, recent_days=args.recent)
    elif args.run_ids:
        run_ids = args.run_ids
    else:
        parser.error("Provide run IDs or use --all")

    runs = [r for rid in run_ids if (r := load_run(rid)) is not None]

    if not runs:
        print("No valid runs found.", file=sys.stderr)
        sys.exit(1)

    report = generate_report(runs)

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
