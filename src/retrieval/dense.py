"""Dense retrieval over the saved chunk vectors: FAISS flat index + numpy fallback."""
import time
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


def load_vectors(emb_dir) -> np.ndarray:
    V = np.load(Path(emb_dir) / "chunk_vectors.npy").astype(np.float32)
    return np.ascontiguousarray(V)


def _numpy_topk(V, q, k):
    """Exact top-k by dot product. Returns (scores, ids), best first."""
    S = q @ V.T
    part = np.argpartition(-S, k, axis=1)[:, :k]
    ps = np.take_along_axis(S, part, axis=1)
    order = np.argsort(-ps, axis=1)
    ix = np.take_along_axis(part, order, axis=1)
    return np.take_along_axis(S, ix, axis=1), ix


class DenseIndex:
    def __init__(self, vectors, backend="auto"):
        self.V = vectors
        if backend == "auto":
            backend = "faiss" if faiss is not None else "numpy"
        if backend == "faiss" and faiss is None:
            raise RuntimeError("faiss-cpu is not installed (pip install faiss-cpu)")
        self.backend = backend
        t0 = time.perf_counter()
        if backend == "faiss":
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
        self.build_seconds = time.perf_counter() - t0

    def search(self, Q, k=100, batch=256):
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        ids = np.empty((len(Q), k), dtype=np.int32)
        scores = np.empty((len(Q), k), dtype=np.float32)
        for i in range(0, len(Q), batch):
            q = Q[i:i + batch]
            if self.backend == "faiss":
                s, ix = self.index.search(q, k)
            else:
                s, ix = _numpy_topk(self.V, q, k)
            ids[i:i + batch], scores[i:i + batch] = ix, s
        return ids, scores


def cross_check(index, Q, k=10, n=200, batch=64):
    """Compare the index with plain numpy on the first n queries.
    Scores are compared, not order, because near-equal scores can swap places."""
    Q = np.ascontiguousarray(Q[:n], dtype=np.float32)
    a_ids, a_s = index.search(Q, k)
    b_ids, b_s = np.empty_like(a_ids), np.empty_like(a_s)
    for i in range(0, len(Q), batch):
        s, ix = _numpy_topk(index.V, Q[i:i + batch], k)
        b_ids[i:i + batch], b_s[i:i + batch] = ix, s
    same_set = float(np.mean([set(x) == set(y) for x, y in zip(a_ids, b_ids)]))
    return {"queries": len(Q), "same_top_k_set": same_set,
            "max_score_diff": float(np.abs(a_s - b_s).max())}


def time_single(index, Q, n=200, k=100):
    """Per-question latency (batch size 1). Returns (p50_ms, p95_ms)."""
    for q in Q[:3]:                                   # warm-up
        index.search(q[None, :], k)
    ms = []
    for q in Q[:n]:
        t = time.perf_counter()
        index.search(q[None, :], k)
        ms.append((time.perf_counter() - t) * 1000)
    return float(np.percentile(ms, 50)), float(np.percentile(ms, 95))