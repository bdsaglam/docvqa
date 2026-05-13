"""Prepare the MP-DocVQA val 200-question sample.

Builds ``data/mp-docvqa/val/sample_200q_doc_ids.txt`` with a fixed
stratified sample, seed=42. Stratification axis is page count, sampled
proportionally to the val question distribution (multi-page mix matters
for this benchmark).

Re-running is idempotent: same seed, same sample.
"""

from __future__ import annotations

import ast
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

SEED = 42
TARGET_QS = 200
OUT = Path("data/mp-docvqa/val/sample_200q_doc_ids.txt")

# Bucket boundaries (inclusive upper bound)
BUCKETS = [
    ("1pp", 1, 1),
    ("2-5pp", 2, 5),
    ("6-10pp", 6, 10),
    ("11-20pp", 11, 20),
]


def _bucket(p: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= p <= hi:
            return name
    return "other"


def _pages_count(row) -> int:
    pages = row["page_ids"]
    if isinstance(pages, str):
        pages = ast.literal_eval(pages)
    return len(pages)


def main() -> None:
    ds = load_dataset("lmms-lab/MP-DocVQA", split="val")

    doc_pages: dict[str, int] = {}
    doc_qcount: dict[str, int] = defaultdict(int)
    for row in ds:
        d = row["doc_id"]
        doc_qcount[d] += 1
        if d not in doc_pages:
            doc_pages[d] = _pages_count(row)

    total_q = sum(doc_qcount.values())
    bucket_q: dict[str, int] = defaultdict(int)
    bucket_docs: dict[str, list[str]] = defaultdict(list)
    for d, p in doc_pages.items():
        b = _bucket(p)
        bucket_q[b] += doc_qcount[d]
        bucket_docs[b].append(d)

    # Proportional per-bucket targets
    targets = {
        b: max(1, round(TARGET_QS * bucket_q[b] / total_q)) for b in bucket_q
    }
    print("Universe:", {b: (len(bucket_docs[b]), bucket_q[b]) for b in bucket_docs})
    print("Targets (questions per bucket):", targets)

    rng = random.Random(SEED)
    picked: list[str] = []
    picked_qs = 0
    actual_per_bucket: Counter[str] = Counter()

    for b in sorted(bucket_docs.keys()):
        candidates = list(bucket_docs[b])
        rng.shuffle(candidates)
        budget = targets[b]
        for d in candidates:
            if actual_per_bucket[b] >= budget:
                break
            picked.append(d)
            actual_per_bucket[b] += doc_qcount[d]
            picked_qs += doc_qcount[d]

    print("Picked:", actual_per_bucket, "total_qs=", picked_qs, "total_docs=", len(picked))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# MP-DocVQA val sample (seed={SEED}, target={TARGET_QS} questions)",
        f"# Picked {len(picked)} docs / {picked_qs} questions",
        "# Per-bucket: " + ", ".join(f"{b}={actual_per_bucket[b]}q" for b in sorted(actual_per_bucket)),
        "",
    ]
    body = "\n".join(header + picked) + "\n"
    OUT.write_text(body)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
