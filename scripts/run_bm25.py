"""BM25 retrieval for one config: saves the top-`depth` chunks per question.
    python scripts/run_bm25.py --config configs/a1_bm25.yaml
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from config import chunks_path, load, set_seeds, start_run
from retrieval.bm25 import build, search, time_single, tokenize_queries


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--n-latency", type=int, default=200)
ap.add_argument("--allow-test", action="store_true")
args = ap.parse_args()

cfg = load(args.config)
if cfg.data.split == "test" and not args.allow_test:
    sys.exit("split is 'test'. The test set is opened only in Phase 6 (pass --allow-test then).")
assert cfg.retrieval.type == "bm25", "this script runs BM25 only"
set_seeds(cfg.seed)

cpath = chunks_path(cfg)
chunks = [json.loads(l) for l in open(cpath, encoding="utf-8")]
assert [c["cid"] for c in chunks] == list(range(len(chunks))), "chunk ids must be 0..N-1 in order"
texts = [c["text"] for c in chunks]

run_dir = start_run(cfg, args.config)

retriever, build_s = build(texts, cfg.retrieval.bm25_k1, cfg.retrieval.bm25_b)
ch = cfg.chunking
index_dir = Path("data/indexes") / (f"bm25_{ch.method}_{ch.chunk_size}_{ch.overlap}"
                                    f"_k1{cfg.retrieval.bm25_k1}_b{cfg.retrieval.bm25_b}")
index_dir.parent.mkdir(parents=True, exist_ok=True)
retriever.save(str(index_dir))
index_mb = sum(p.stat().st_size for p in index_dir.rglob("*") if p.is_file()) / 1e6
print(f"chunks: {len(texts)} | index build: {build_s:.1f}s | index on disk: {index_mb:.0f} MB")

queries = [json.loads(l) for l in open(Path("data/processed") / f"{cfg.data.split}.jsonl", encoding="utf-8")]
qids = [q["qid"] for q in queries]
qtok = tokenize_queries([q["query"] for q in queries])
print("questions:", len(qids), "| mean query tokens:", round(sum(map(len, qtok)) / len(qtok), 1))

t0 = time.perf_counter()
ids, scores = search(retriever, qtok, k=cfg.retrieval.depth)
batch_s = time.perf_counter() - t0
p50, p95 = time_single(retriever, qtok, n=args.n_latency, k=cfg.retrieval.depth)
print(f"batch search: {batch_s:.1f}s | single-question latency p50={p50:.1f} ms p95={p95:.1f} ms")
print("zero-score results in the top 100 (share):", round(float((scores <= 0).mean()), 4))

np.savez_compressed(run_dir / "retrieval.npz", qids=np.array(qids), ids=ids, scores=scores)
json.dump({
    "run_id": run_dir.name, "split": cfg.data.split, "query_variant": "full_text",
    "backend": "bm25s", "n_queries": len(qids), "depth": cfg.retrieval.depth,
    "index_build_seconds": round(build_s, 3),
    "vectors_mb": round(index_mb, 1),            # for BM25 this is the index size on disk
    "batch_search_seconds": round(batch_s, 2), "latency_p50_ms": round(p50, 2),
    "latency_p95_ms": round(p95, 2), "cross_check": None,
    "model": "bm25s-lucene", "model_revision": version("bm25s"),
    "k1": cfg.retrieval.bm25_k1, "b": cfg.retrieval.bm25_b,
    "chunks_sha256": sha256(cpath), "created": datetime.utcnow().isoformat() + "Z",
}, open(run_dir / "retrieval_meta.json", "w", encoding="utf-8"), indent=2)

print("\nrun folder:", run_dir)
print("next: python scripts/evaluate_run.py --run", run_dir)