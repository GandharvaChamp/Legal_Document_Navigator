"""Paired bootstrap comparison of two runs on the same split: difference = B - A.
    python scripts/compare_runs.py experiments/runs/<A> experiments/runs/<B> --k 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from evaluation.aggregate import GROUPS, bootstrap, load_per_query, summarize

ap = argparse.ArgumentParser()
ap.add_argument("run_a")
ap.add_argument("run_b")
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--metrics", default="recall,hit,mrr,ndcg,precision,doc_hit")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

qa, sa, XA, na = load_per_query(Path(args.run_a) / "per_query_metrics.csv")
qb, sb, XB, nb = load_per_query(Path(args.run_b) / "per_query_metrics.csv")
assert qa == qb and na == nb and (sa == sb).all(), "runs are not on the same questions"

S = summarize(XB - XA, sa, n_boot=args.boot, seed=args.seed)
pa, _ = bootstrap(XA, sa, n_boot=1, seed=args.seed)
pb, _ = bootstrap(XB, sb, n_boot=1, seed=args.seed)


def verdict(lo, hi):
    return "B BETTER" if lo > 0 else "B WORSE" if hi < 0 else "within noise"


print(f"A = {args.run_a}\nB = {args.run_b}\ndifference = B - A, 95% paired bootstrap, k={args.k}")
for m in args.metrics.split(","):
    j = na.index(f"{m}@{args.k}")
    print(f"\n--- {m}@{args.k} ---")
    print(f"{'group':<13}{'A':>8}{'B':>8}{'diff':>9}   {'95% interval':<20}verdict")
    for g in [g for g in GROUPS if g in S]:
        p, lo, hi = S[g]
        print(f"{g:<13}{pa[g][j]:>8.3f}{pb[g][j]:>8.3f}{p[j]:>+9.3f}   "
              f"[{lo[j]:+.3f}, {hi[j]:+.3f}]".ljust(56) + verdict(lo[j], hi[j]))