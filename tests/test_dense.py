import sys

import numpy as np
import pytest

sys.path.insert(0, "src")
from retrieval.dense import DenseIndex, _numpy_topk, cross_check, faiss


def make(n=500, d=16, q=7, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n, d)).astype(np.float32)
    Q = rng.normal(size=(q, d)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return V, Q


def test_numpy_topk_matches_full_sort():
    V, Q = make()
    s, ix = _numpy_topk(V, Q, 10)
    S = Q @ V.T
    expected = np.argsort(-S, axis=1)[:, :10]
    assert np.array_equal(ix, expected)
    assert np.allclose(s, np.take_along_axis(S, expected, axis=1))


def test_numpy_backend_shapes_batching_and_order():
    V, Q = make(q=300)
    ids, sc = DenseIndex(V, "numpy").search(Q, k=5, batch=64)
    assert ids.shape == (300, 5) and sc.shape == (300, 5)
    assert (np.diff(sc, axis=1) <= 1e-6).all()


@pytest.mark.skipif(faiss is None, reason="faiss-cpu not installed")
def test_faiss_agrees_with_numpy():
    V, Q = make()
    r = cross_check(DenseIndex(V, "faiss"), Q, k=10, n=7)
    assert r["same_top_k_set"] == 1.0 and r["max_score_diff"] < 1e-5