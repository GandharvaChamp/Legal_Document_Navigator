import hashlib
import json
import random
from pathlib import Path

import numpy as np

E = Path("data/embeddings/bge-base-en-v1.5")
CHUNKS = Path("data/processed/chunks_recursive_512_64.jsonl")
QUERIES = Path("data/processed/queries_for_embedding.jsonl")
DEV = Path("data/processed/dev.jsonl")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


man = json.load(open(E / "manifest.json", encoding="utf-8"))
print("model:", man["model"], "| revision:", man["model_revision"], "| dim:", man["dim"])

# 1. the vectors belong to exactly these files
assert sha256(CHUNKS) == man["chunks_sha256"], "chunk file differs from the one that was embedded"
assert sha256(QUERIES) == man["queries_sha256"], "query file differs from the one that was embedded"
print("1) input files match the manifest: OK")

# 2. row order = chunk order
chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
ids = np.load(E / "chunk_ids.npy")
assert list(ids) == [c["cid"] for c in chunks], "chunk id order mismatch"
V = np.load(E / "chunk_vectors.npy").astype(np.float32)
assert V.shape == (len(chunks), man["dim"]), V.shape
norms = np.linalg.norm(V, axis=1)
print(f"2) chunk vectors {V.shape} in chunk order: OK | norms min/max {norms.min():.3f}/{norms.max():.3f}")

# 3. query vectors line up with query ids
queries = [json.loads(l) for l in open(QUERIES, encoding="utf-8")]
qids = json.load(open(E / "query_ids.json"))
assert qids == [q["qid"] for q in queries]
pos = {q: i for i, q in enumerate(qids)}
QV = {}
for name in man["query_variants"]:
    QV[name] = np.load(E / f"query_vectors_{name}.npy").astype(np.float32)
    assert QV[name].shape == (len(qids), man["dim"]), name
print("3) query vectors:", {k: v.shape for k, v in QV.items()}, ": OK")

# 4. smoke test (NOT a result): is the right CONTRACT in the top 10 for 100 dev questions?
dev = [json.loads(l) for l in open(DEV, encoding="utf-8")]
sample = random.Random(0).sample(dev, 100)
doc_of = [c["doc"] for c in chunks]
for name in ["full_instr", "full_plain"]:
    Q = QV[name][[pos[q["qid"]] for q in sample]]
    S = Q @ V.T
    top = np.argpartition(-S, 10, axis=1)[:, :10]
    hits = sum(any(doc_of[j] in {g["doc"] for g in q["gold"]} for j in row)
               for row, q in zip(top, sample))
    print(f"4) {name}: right contract in top-10 for {hits}/100 dev questions")