import sys

import numpy as np

sys.path.insert(0, "src")
from retrieval.bm25 import build, search, tokenize_queries


def corpus():
    docs = [f"the party shall comply with the contract number {i} and the law" for i in range(30)]
    docs[7] = "the party shall comply with the contract and pay Zyxel Holdings"
    docs[19] = "the contract terminates on December 31"
    return docs


def top1(retriever, query):
    ids, _ = search(retriever, tokenize_queries([query]), k=3)
    return int(ids[0, 0])


def test_rare_word_outweighs_common_words():
    r, _ = build(corpus())
    assert top1(r, "contract Zyxel") == 7


def test_case_is_ignored():
    r, _ = build(corpus())
    assert top1(r, "CONTRACT ZYXEL") == 7


def test_unknown_words_do_not_crash():
    r, _ = build(corpus())
    assert top1(r, "qqqqzz zyxel") == 7


def test_shapes_and_descending_scores():
    r, _ = build(corpus())
    ids, sc = search(r, tokenize_queries(["contract zyxel", "terminates december"]), k=5)
    assert ids.shape == (2, 5) and sc.shape == (2, 5)
    assert (np.diff(sc, axis=1) <= 1e-6).all()
    assert ids[1, 0] == 19


def test_batch_equals_single():
    r, _ = build(corpus())
    qs = ["contract zyxel", "terminates december"]
    ids_b, sc_b = search(r, tokenize_queries(qs), k=5)
    for i, q in enumerate(qs):
        ids_s, sc_s = search(r, tokenize_queries([q]), k=5)
        assert np.array_equal(ids_b[i], ids_s[0]) and np.allclose(sc_b[i], sc_s[0])