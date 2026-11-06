"""Print questions with the gold clause and the top retrieved chunks.
    python scripts/inspect_run.py --run experiments/runs/<run> --kind wrong_clause --n 5
kinds: wrong_clause (right contract in top 10, clause missed), success (clause hit),
       wrong_contract (no chunk from the right contract in the top 10)
"""
import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from config import chunks_path, load
from evaluation.aggregate import load_per_query
from evaluation.retrieval_metrics import is_hit

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--kind", default="wrong_clause", choices=["wrong_clause", "success", "wrong_contract"])
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--top", type=int, default=3)
ap.add_argument("--source", default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--allow-test", action="store_true")
args = ap.parse_args()

run_dir = Path(args.run)
rc = json.load(open(run_dir / "config_resolved.json", encoding="utf-8"))
split = rc["data"]["split"]
if split == "test" and not args.allow_test:
    sys.exit("split is 'test'. Inspected only in Phase 6 (pass --allow-test then).")

queries = [json.loads(l) for l in open(Path("data/processed") / f"{split}.jsonl", encoding="utf-8")]
npz = np.load(run_dir / "retrieval.npz")
qids, sources, X, names = load_per_query(run_dir / "per_query_metrics.csv")
assert qids == [q["qid"] for q in queries]
H, D = names.index("hit@10"), names.index("doc_hit@10")

keep = {"wrong_clause": lambda i: X[i, D] == 1 and X[i, H] == 0,
        "success": lambda i: X[i, H] == 1,
        "wrong_contract": lambda i: X[i, D] == 0}[args.kind]
cand = [i for i in range(len(qids)) if keep(i) and (args.source in (None, sources[i]))]
print(f"{len(cand)} questions of kind '{args.kind}'" + (f" in {args.source}" if args.source else ""))
picked = random.Random(args.seed).sample(cand, min(args.n, len(cand)))

cfg = load(run_dir / "config.yaml")
chunks = [json.loads(l) for l in open(chunks_path(cfg), encoding="utf-8")]
short = lambda s, w=300: textwrap.shorten(" ".join(s.split()), width=w, placeholder=" ...")

for i in picked:
    q = queries[i]
    gold_docs = {g["doc"] for g in q["gold"]}
    print("\n" + "=" * 100)
    print(f"{q['qid']} [{q['source']}]  hit@10={X[i, H]:.0f}  doc_hit@10={X[i, D]:.0f}")
    print("QUERY:", short(q["query"], 400))
    for g in q["gold"]:
        print(f"GOLD  {g['doc'][-60:]}  [{g['start']}:{g['end']}]")
        print("      ", short(g["answer"]))
    first = next((r for r, j in enumerate(npz["ids"][i], 1) if chunks[j]["doc"] in gold_docs), None)
    print(f"first chunk from the right contract: rank {first}")
    for r in range(args.top):
        j = int(npz["ids"][i][r])
        c = chunks[j]
        flag = "HIT " if any(is_hit(c, g) for g in q["gold"]) else "miss"
        mark = "right contract" if c["doc"] in gold_docs else "OTHER contract"
        print(f"  #{r + 1} score={npz['scores'][i][r]:.3f} {flag} {mark}  {c['doc'][-45:]} [{c['start']}:{c['end']}]")
        print("      ", short(c["text"], 260))