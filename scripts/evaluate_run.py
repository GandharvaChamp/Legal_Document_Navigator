"""Score a saved retrieval run with the hit rule, bootstrap intervals and logging.
    python scripts/evaluate_run.py --run experiments/runs/<run_id>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from config import chunks_path, load
from evaluation.aggregate import (METRICS, SOURCES, col, compute_per_query,
                                  print_summary, save_per_query, summarize)

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--allow-test", action="store_true")
args = ap.parse_args()

run_dir = Path(args.run)
rc = json.load(open(run_dir / "config_resolved.json", encoding="utf-8"))
meta = json.load(open(run_dir / "retrieval_meta.json", encoding="utf-8"))
split = rc["data"]["split"]
if split == "test" and not args.allow_test:
    sys.exit("split is 'test'. Scored only in Phase 6 (pass --allow-test then).")

queries = [json.loads(l) for l in open(Path("data/processed") / f"{split}.jsonl", encoding="utf-8")]
npz = np.load(run_dir / "retrieval.npz")
assert list(npz["qids"]) == [q["qid"] for q in queries], "saved results do not match the split file"

cfg = load(run_dir / "config.yaml")
chunk_meta = []
with open(chunks_path(cfg), encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        chunk_meta.append({"doc": c["doc"], "start": c["start"], "end": c["end"]})
print("chunks loaded:", len(chunk_meta), "| questions:", len(queries))

X = compute_per_query(queries, npz["ids"], chunk_meta)
sources = [q["source"] for q in queries]
save_per_query(run_dir / "per_query_metrics.csv", [q["qid"] for q in queries], sources, X)

summary = summarize(X, sources, n_boot=args.boot, seed=rc["seed"])
print_summary(summary, k=10)

from evaluation.aggregate import column_names
names = column_names()
json.dump({g: {n: [float(summary[g][0][j]), float(summary[g][1][j]), float(summary[g][2][j])]
               for j, n in enumerate(names)} for g in summary},
          open(run_dir / "summary.json", "w", encoding="utf-8"), indent=1)

# ---- append one row to the experiment table
row = {
    "run_id": run_dir.name, "run_name": rc["run_name"], "split": split,
    "n_queries": len(queries), "retrieval": rc["retrieval"]["type"],
    "rerank": rc["rerank"]["enabled"], "query_variant": meta["query_variant"],
    "model": meta["model"], "chunking": f'{rc["chunking"]["method"]}_{rc["chunking"]["chunk_size"]}_{rc["chunking"]["overlap"]}',
    "depth": meta["depth"], "seed": rc["seed"], "boot": args.boot,
    "latency_p50_ms": meta["latency_p50_ms"], "latency_p95_ms": meta["latency_p95_ms"],
    "index_build_seconds": meta["index_build_seconds"], "vectors_mb": meta["vectors_mb"],
    "macro_recall@100": round(float(summary["macro"][0][col("recall", 100)]), 4),
}
for m in METRICS:
    j = col(m, 10)
    p, lo, hi = summary["macro"]
    row[f"macro_{m}@10"], row[f"macro_{m}@10_lo"], row[f"macro_{m}@10_hi"] = (
        round(float(p[j]), 4), round(float(lo[j]), 4), round(float(hi[j]), 4))
    row[f"pooled_{m}@10"] = round(float(summary["pooled"][0][j]), 4)
for s in SOURCES:
    row[f"{s}_recall@10"] = round(float(summary[s][0][col("recall", 10)]), 4)

table = Path("experiments/experiment_table.csv")
table.parent.mkdir(exist_ok=True)
new = not table.exists()
with open(table, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
print("\nlogged to", table)