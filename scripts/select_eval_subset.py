#!/usr/bin/env python
"""Select a small, representative *doc* subset of a split for fast eval-during-training.

Given the per-trial `result.json`s of a solver run (n trials of the same config),
this does a **stratified random search**: it draws many category-proportional random
doc-subsets, scores each against the full set across all trials, and keeps the one that
best matches the full set on (a) mean accuracy, (b) per-trial tracking (Pearson r over
the n trials), while staying under a page budget (eval cost). The result is a subset
that is *representative* (difficulty + category mix) and *cheap* (fewer pages), chosen
randomly-but-filtered (reproducible via --seed), not hand-picked.

The loader supports `data.doc_ids=[...]`, so the printed list is directly usable.

Example:
    python scripts/select_eval_subset.py \
        --base codeact-chat-4b-llm-27b-vlm-val \
        --target-size 13 --page-budget 450 --seed 20260613 \
        --cross-base codeact-chat-val           # optional cross-config sanity check
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import random
import statistics as st
from collections import defaultdict

RUNS = "output/runs"


def load(base: str):
    """Return (perdoc, cats, pages, trials).

    perdoc[doc][trial] = (n_correct, n_questions) or None if the doc is missing in that trial.
    """
    perdoc: dict[str, dict[str, tuple[int, int] | None]] = {}
    cats: dict[str, str] = {}
    trial_dirs = sorted(glob.glob(f"{RUNS}/{base}-t*/tasks"))
    trials = [os.path.basename(os.path.dirname(d)) for d in trial_dirs]
    for tdir, t in zip(trial_dirs, trials):
        for dd in os.listdir(tdir):
            rj = f"{tdir}/{dd}/result.json"
            if not os.path.exists(rj):
                perdoc.setdefault(dd, {})[t] = None
                continue
            r = json.load(open(rj))
            qs = r.get("questions", [])
            n_ok = sum(1 for q in qs if str(q.get("is_correct")) == "True")
            perdoc.setdefault(dd, {})[t] = (n_ok, len(qs))
            cats[dd] = r.get("doc_category", "_".join(dd.split("_")[:-1]))
    pages = {}
    if trials:
        first = f"{RUNS}/{trials[0]}/tasks"
        for d in perdoc:
            pages[d] = len(glob.glob(f"{first}/{d}/page_*.jpg"))
    return perdoc, cats, pages, trials


def trial_vec(subset, perdoc, trials):
    """Question-weighted accuracy (%) per trial over `subset`."""
    out = []
    for t in trials:
        c = sum(perdoc[d][t][0] for d in subset)
        n = sum(perdoc[d][t][1] for d in subset)
        out.append(100 * c / n if n else 0.0)
    return out


def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da * db else 0.0


def proportional_allocations(bycat, order, target):
    """Largest-remainder proportional allocation to `target` docs (min 1/category).

    Returns all variants when the fractional-remainder tie is ambiguous, so the search
    randomizes over equally-proportional splits too.
    """
    total = sum(len(bycat[c]) for c in order)
    share = {c: len(bycat[c]) * target / total for c in order}
    alloc = {c: max(1, int(share[c])) for c in order}
    rem = target - sum(alloc.values())
    if rem <= 0:
        return [alloc]
    # candidates to receive +1: highest fractional remainders
    fr = {c: share[c] - int(share[c]) for c in order}
    cutoff = sorted(fr.values(), reverse=True)[min(rem, len(order)) - 1] - 1e-9
    high = [c for c in order if fr[c] >= cutoff]
    variants = []
    for combo in itertools.combinations(high, rem):
        a = dict(alloc)
        for c in combo:
            a[c] += 1
        if sum(a.values()) == target:
            variants.append(a)
    return variants or [alloc]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="run-id prefix; trials are <base>-t1..tN")
    ap.add_argument("--target-size", type=int, default=13, help="number of docs in the subset")
    ap.add_argument("--page-budget", type=int, default=None, help="max total pages (eval cost cap)")
    ap.add_argument("--tol-mean", type=float, default=0.8, help="max |subset_mean - full_mean| in pp")
    ap.add_argument("--min-r", type=float, default=0.90, help="min Pearson r across trials")
    ap.add_argument("--draws", type=int, default=150000, help="random stratified draws")
    ap.add_argument("--seed", type=int, default=20260613)
    ap.add_argument("--cross-base", default=None, help="optional second config for a robustness check")
    ap.add_argument("--json-out", default=None, help="optional path to write the chosen subset")
    args = ap.parse_args()

    perdoc, cats, pages, trials = load(args.base)
    if not trials:
        raise SystemExit(f"No trials found for base '{args.base}' under {RUNS}/")
    docs = [d for d in sorted(perdoc) if all(perdoc[d].get(t) for t in trials)]
    bycat = defaultdict(list)
    for d in docs:
        bycat[cats[d]].append(d)
    order = sorted(bycat)
    fullv = trial_vec(docs, perdoc, trials)
    fm = st.mean(fullv)
    full_pages = sum(pages.get(d, 0) for d in docs)

    print(f"FULL [{args.base}]: {len(docs)} docs, {len(trials)} trials, {full_pages or '?'}pp, "
          f"mean {fm:.2f} std {st.stdev(fullv):.2f}")
    print(f"  per-category: { {c: len(bycat[c]) for c in order} }")

    variants = proportional_allocations(bycat, order, args.target_size)

    random.seed(args.seed)
    cand: dict[tuple, tuple] = {}
    for _ in range(args.draws):
        a = random.choice(variants)
        s = []
        for c in order:
            s += random.sample(bycat[c], a[c])
        key = tuple(sorted(s))
        if key in cand:
            continue
        v = trial_vec(key, perdoc, trials)
        md = abs(st.mean(v) - fm)
        mx = max(abs(x - y) for x, y in zip(v, fullv))
        r = pearson(v, fullv)
        pg = sum(pages.get(d, 0) for d in key)
        cand[key] = (md, mx, r, st.mean(v), pg, a)

    def eligible(v) -> bool:
        md, mx, r, m, pg, a = v
        if md > args.tol_mean or r < args.min_r:
            return False
        if args.page_budget is not None and pg > args.page_budget:
            return False
        return True

    elig = [(k, v) for k, v in cand.items() if eligible(v)]
    if not elig:
        raise SystemExit("No subset met the filters — loosen --tol-mean / --min-r / --page-budget.")
    # rank by match quality: tight mean (md), low max-dev (mx), high correlation (r)
    elig.sort(key=lambda kv: kv[1][0] * 1.0 + kv[1][1] * 0.4 + (1 - kv[1][2]) * 6.0)
    print(f"\n{len(elig)} distinct subsets pass (Δ<={args.tol_mean}pp, r>={args.min_r}"
          + (f", pages<={args.page_budget}" if args.page_budget else "") + "); picking best match.\n")

    key, (md, mx, r, m, pg, a) = elig[0]
    s = list(key)
    v = trial_vec(s, perdoc, trials)
    nq = sum(pd[1] for d in s if (pd := perdoc[d][trials[0]]))
    speed = f"{full_pages / pg:.1f}x" if pg else "n/a"
    print(f"=== SUBSET: {len(s)} docs"
          + (f", {pg}pp = {100*pg/full_pages:.0f}% of full (~{speed} faster)" if full_pages else "")
          + f", {nq} questions ===")
    for d in s:
        print(f"   {d:24s} {cats[d]:20s} {pages.get(d,'?')}pp")
    print(f"\ncomposition: { {c: a[c] for c in order} }")
    print(f"\nPROOF — {args.base}, {len(trials)} trials:")
    print("  trial   subset   full")
    for i, t in enumerate(trials):
        print(f"   {t.split('-')[-1]:>4}   {v[i]:6.1f}  {fullv[i]:6.1f}")
    print(f"  mean    {st.mean(v):6.2f}  {fm:6.2f}   (Δ={st.mean(v)-fm:+.2f}pp)")
    print(f"  std     {st.stdev(v):6.2f}  {st.stdev(fullv):6.2f}")
    print(f"  max per-trial dev: {mx:.2f}pp | Pearson r: {r:.3f}")

    if args.cross_base:
        p2, _, _, t2 = load(args.cross_base)
        d2 = [d for d in docs if all(p2.get(d, {}).get(t) for t in t2)] if t2 else []
        if d2 and set(s) <= set(d2):
            fv2 = trial_vec(d2, p2, t2)
            sv2 = trial_vec(s, p2, t2)
            print(f"\nCROSS-CONFIG [{args.cross_base}] (not used for selection): "
                  f"subset {st.mean(sv2):.2f} vs full {st.mean(fv2):.2f} "
                  f"(Δ={st.mean(sv2)-st.mean(fv2):+.2f}pp), r={pearson(sv2, fv2):.3f}")
        else:
            print(f"\nCROSS-CONFIG [{args.cross_base}]: skipped (subset docs not all present).")

    print("\ndoc_ids =", s)
    if args.json_out:
        json.dump(
            {"base": args.base, "doc_ids": s, "composition": {c: a[c] for c in order},
             "subset_mean": st.mean(v), "full_mean": fm, "pearson_r": r, "max_dev": mx,
             "pages": pg, "full_pages": full_pages, "seed": args.seed},
            open(args.json_out, "w"), indent=2,
        )
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
