import sys

import numpy as np

sys.path.insert(0, "src")
from evaluation.aggregate import SOURCES, bootstrap, col, compute_per_query, summarize


def sources():
    return np.array(["privacy_qa"] * 10 + ["contractnli"] * 20 + ["cuad"] * 60 + ["maud"] * 30)


def test_constant_scores_have_zero_width_interval():
    s = summarize(np.full((120, 3), 0.5), sources(), n_boot=200)
    for g, (p, lo, hi) in s.items():
        assert np.allclose(p, 0.5) and np.allclose(lo, 0.5) and np.allclose(hi, 0.5), g


def test_identical_runs_have_zero_difference():
    X = np.random.default_rng(0).random((120, 4))
    s = summarize(X - X, sources(), n_boot=200)
    for g, (p, lo, hi) in s.items():
        assert np.allclose(p, 0) and np.allclose(lo, 0) and np.allclose(hi, 0), g


def test_macro_is_mean_of_sources_and_pooled_is_overall_mean():
    src = sources()
    X = np.random.default_rng(1).random((120, 2))
    point, _ = bootstrap(X, src, n_boot=50)
    assert np.allclose(point["macro"], np.mean([X[src == s].mean(axis=0) for s in SOURCES], axis=0))
    assert np.allclose(point["pooled"], X.mean(axis=0))
    assert not np.allclose(point["macro"], point["pooled"])      # source sizes are unbalanced


def test_point_inside_interval_and_deterministic():
    X = np.random.default_rng(2).random((120, 3))
    a = summarize(X, sources(), n_boot=300, seed=1)
    b = summarize(X, sources(), n_boot=300, seed=1)
    for g in a:
        p, lo, hi = a[g]
        assert (lo <= p + 1e-9).all() and (p <= hi + 1e-9).all()
        assert np.array_equal(lo, b[g][1]) and np.array_equal(hi, b[g][2])


def test_paired_difference_detects_a_clear_improvement():
    A = np.random.default_rng(3).random((120, 2)) * 0.1 + 0.4
    p, lo, hi = summarize((A + 0.05) - A, sources(), n_boot=300)["macro"]
    assert lo.min() > 0


def test_compute_per_query_toy_example():
    chunk_meta = [{"doc": "a", "start": 0, "end": 512},
                  {"doc": "a", "start": 512, "end": 1024},
                  {"doc": "b", "start": 0, "end": 512}]
    queries = [{"gold": [{"doc": "a", "start": 100, "end": 400, "answer": ""}]},
               {"gold": [{"doc": "a", "start": 600, "end": 900, "answer": ""}]}]
    ids = np.array([[0, 2], [2, 1]])
    ks = [1, 2]
    X = compute_per_query(queries, ids, chunk_meta, ks=ks)
    c = lambda m, k: col(m, k, ks)
    assert X[0, c("recall", 1)] == 1 and X[0, c("mrr", 2)] == 1
    assert X[1, c("recall", 1)] == 0 and X[1, c("recall", 2)] == 1
    assert X[1, c("mrr", 2)] == 0.5
    assert X[1, c("doc_hit", 1)] == 0 and X[1, c("doc_hit", 2)] == 1