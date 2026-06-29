"""Stratified-random doc_id sampling for the dataset-axis evals.

For large benchmarks we evaluate a sample rather than the full set. The sample
is **stratified** by a categorical key (so every stratum is represented in
proportion to its size) and **random within each stratum** (fixed seed for
reproducibility). Writes one doc_id per line to an output file consumable via
``data.doc_ids_file=...``.

Usage:
    uv run python scripts/stratified_sample.py mmlongbench --n-docs 20 --seed 0 \
        --out data/mmlongbench-doc/val/sample_doc_ids.txt
    uv run python scripts/stratified_sample.py mp_docvqa --n-docs 40 --seed 0 \
        --out data/mp-docvqa/val/sample_doc_ids.txt
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


def _stratified_sample(
    items: dict[str, str], n: int, seed: int
) -> list[str]:
    """``items`` maps id -> stratum key. Return ~n ids, stratified-random.

    Allocates per stratum proportional to its share (largest-remainder), then
    samples that many ids uniformly at random within the stratum.
    """
    rng = random.Random(seed)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for _id, key in items.items():
        by_stratum[str(key)].append(_id)
    for ids in by_stratum.values():
        ids.sort()  # determinism before shuffle
        rng.shuffle(ids)

    total = len(items)
    # proportional float allocation, then largest-remainder rounding to hit n
    raw = {k: (len(v) / total) * n for k, v in by_stratum.items()}
    alloc = {k: min(len(by_stratum[k]), int(v)) for k, v in raw.items()}
    short = n - sum(alloc.values())
    # distribute the remaining slots by largest fractional remainder
    rema = sorted(by_stratum, key=lambda k: raw[k] - int(raw[k]), reverse=True)
    i = 0
    while short > 0 and rema:
        k = rema[i % len(rema)]
        if alloc[k] < len(by_stratum[k]):
            alloc[k] += 1
            short -= 1
        i += 1
        if i > 10 * len(rema):  # all strata exhausted
            break

    picked: list[str] = []
    for k, m in alloc.items():
        picked.extend(by_stratum[k][:m])
    picked.sort()
    return picked


def _mmlb_items() -> dict[str, str]:
    """doc_id -> doc_type for MMLongBench-Doc (cheap: reads HF rows, no render)."""
    from datasets import load_dataset

    ds = load_dataset("yubo2333/MMLongBench-Doc", split="train")
    items: dict[str, str] = {}
    for row in ds:
        items[row["doc_id"]] = row["doc_type"]
    return items


def _mp_items() -> dict[str, str]:
    """doc_id -> page-count-bin stratum for MP-DocVQA val.

    MP-DocVQA has no native category, so we stratify by document length
    (number of pages, bucketed) — the axis-relevant property. Reads the val
    rows via streaming to avoid decoding every embedded image.
    """
    import ast

    from datasets import load_dataset

    ds = load_dataset("lmms-lab/MP-DocVQA", split="val", streaming=True)
    pages: dict[str, int] = {}
    for row in ds:
        doc_id = row["doc_id"]
        if doc_id in pages:
            continue
        try:
            pid = ast.literal_eval(row["page_ids"]) if isinstance(row["page_ids"], str) else row["page_ids"]
            pages[doc_id] = len(pid) if pid else 1
        except (ValueError, SyntaxError):
            pages[doc_id] = 1

    def _bin(p: int) -> str:
        if p <= 2:
            return "1-2pg"
        if p <= 5:
            return "3-5pg"
        if p <= 10:
            return "6-10pg"
        return "11-20pg"

    return {d: _bin(p) for d, p in pages.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["mmlongbench", "mp_docvqa"])
    ap.add_argument("--n-docs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = _mmlb_items() if args.dataset == "mmlongbench" else _mp_items()
    print(f"{args.dataset}: {len(items)} docs across {len(set(items.values()))} strata")
    by_stratum: dict[str, int] = defaultdict(int)
    for k in items.values():
        by_stratum[str(k)] += 1
    for k in sorted(by_stratum):
        print(f"  {k:20s}: {by_stratum[k]}")

    picked = _stratified_sample(items, args.n_docs, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(picked) + "\n")
    sel_strata: dict[str, int] = defaultdict(int)
    for d in picked:
        sel_strata[str(items[d])] += 1
    print(f"\nWrote {len(picked)} doc_ids -> {out}")
    for k in sorted(sel_strata):
        print(f"  {k:20s}: {sel_strata[k]}")


if __name__ == "__main__":
    main()
