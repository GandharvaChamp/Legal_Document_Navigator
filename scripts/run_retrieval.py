"""Dense retrieval for one config: saves the top-`depth` chunks per question.
    python scripts/run_retrieval.py --config configs/a0a_dense_instr.yaml
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from config import chunks_path, embeddings_dir, load, set_seeds, start_run
from retrieval.dense import DenseIndex, cross_check, load_vectors, time_single


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--backend", default="auto", choices=["auto", "faiss", "numpy"])
ap.add_argument("--n-latency", type=int, default=200)
ap.add_argument("--allow-test", action="store_true")
args = ap.parse_args()

cfg = load(args.config)
if cfg.data.split == "test" and not args.allow_test:
    sys.exit("split is 'test'. The test set is opened only in Phase 6 (pass --allow-test then).")
assert cfg.retrieval.type == "dense", "this script runs dense retrieval only"
set_seeds(cfg.seed)

emb_dir = embeddings_dir(cfg)
man = json.load(open(emb_dir / "manifest.json", encoding="utf-8"))
assert man["model"] == cfg.embedding.model, "config model differs from the embedded model"
assert sha256(chunks_path(cfg)) == man["chunks_sha256"], "chunk file differs from the embedded one"
variant = cfg.embedding.query_variant
assert variant in man["query_variants"], f"unknown variant {variant}"

run_dir = start_run(cfg, args.config)

qids_all = json.load(open(emb_dir / "query_ids.json", encoding="utf-8"))
pos = {q: i for i, q in enumerate(qids_all)}
queries = [json.loads(l) for l in open(Path("data/processed") / f"{cfg.data.split}.jsonl", encoding="utf-8")]
qids = [q["qid"] for q in queries]
Q = np.load(emb_dir / f"query_vectors_{variant}.npy")[[pos[q] for q in qids]].astype(np.float32)

V = load_vectors(emb_dir)
index = DenseIndex(V, args.backend)
print(f"backend: {index.backend} | vectors: {V.shape} ({V.nbytes / 1e6:.0f} MB) | "
      f"build: {index.build_seconds:.2f}s | questions: {len(qids)} | variant: {variant}")

check = cross_check(index, Q, k=10, n=200) if index.backend == "faiss" else None
print("numpy cross-check:", check if check else "skipped (numpy backend)")

t0 = time.perf_counter()
ids, scores = index.search(Q, k=cfg.retrieval.depth)
batch_s = time.perf_counter() - t0
p50, p95 = time_single(index, Q, n=args.n_latency, k=cfg.retrieval.depth)
print(f"batch search: {batch_s:.1f}s | single-question latency p50={p50:.1f} ms p95={p95:.1f} ms")

np.savez_compressed(run_dir / "retrieval.npz", qids=np.array(qids), ids=ids, scores=scores)
json.dump({
    "run_id": run_dir.name, "split": cfg.data.split, "query_variant": variant,
    "backend": index.backend, "n_queries": len(qids), "depth": cfg.retrieval.depth,
    "index_build_seconds": round(index.build_seconds, 3), "vectors_mb": round(V.nbytes / 1e6, 1),
    "batch_search_seconds": round(batch_s, 2), "latency_p50_ms": round(p50, 2),
    "latency_p95_ms": round(p95, 2), "cross_check": check,
    "model": man["model"], "model_revision": man["model_revision"],
    "chunks_sha256": man["chunks_sha256"], "created": datetime.utcnow().isoformat() + "Z",
}, open(run_dir / "retrieval_meta.json", "w", encoding="utf-8"), indent=2)

print("\nrun folder:", run_dir)
print("next: python scripts/evaluate_run.py --run", run_dir)