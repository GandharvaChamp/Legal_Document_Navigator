import math
import sys

import pytest

sys.path.insert(0, "src")
from evaluation.retrieval_metrics import is_hit, query_metrics


def span(doc, start, end):
    return {"doc": doc, "start": start, "end": end}


def test_different_document_is_never_a_hit():
    assert not is_hit(span("a", 0, 512), span("b", 0, 512))


def test_no_overlap_is_a_miss():
    assert not is_hit(span("a", 0, 512), span("a", 600, 900))


def test_chunk_containing_whole_short_gold_is_hit():
    assert is_hit(span("a", 900, 1412), span("a", 1000, 1300))


def test_half_of_short_gold_boundary():
    gold = span("a", 1000, 1300)                     # 300 chars, half = 150
    assert not is_hit(span("a", 1151, 1663), gold)   # overlap 149 -> miss
    assert is_hit(span("a", 1150, 1662), gold)       # overlap 150 -> hit


def test_long_gold_chunk_fully_inside_is_hit():
    assert is_hit(span("a", 1000, 1512), span("a", 0, 5000))


def test_long_gold_partial_overlap_is_miss():
    # overlap 200 < half of the 512-char chunk (256)
    assert not is_hit(span("a", 4800, 5312), span("a", 0, 5000))


def test_any_overlap_rule_is_lenient():
    assert is_hit(span("a", 1299, 1811), span("a", 1000, 1300), rule="any_overlap")


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        is_hit(span("a", 0, 10), span("a", 0, 10), rule="nope")


def test_metrics_perfect_first_rank():
    m = query_metrics([span("a", 1000, 1512)], [span("a", 1000, 1300)], k=10)
    assert m["recall"] == 1 and m["hit"] == 1 and m["mrr"] == 1
    assert m["ndcg"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(0.1)      # 1 relevant of k=10
    assert m["doc_hit"] == 1


def test_metrics_second_rank_and_two_gold_clauses():
    gold = [span("a", 0, 300), span("a", 5000, 5300)]
    retrieved = [span("b", 0, 512), span("a", 0, 512)]   # wrong doc, then hit on gold 1
    m = query_metrics(retrieved, gold, k=2)
    assert m["recall"] == pytest.approx(0.5)             # only one of two gold clauses
    assert m["mrr"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert m["ndcg"] == pytest.approx(expected)


def test_right_contract_wrong_clause_vs_wrong_contract():
    gold = [span("a", 0, 300)]
    near = query_metrics([span("a", 9000, 9512)], gold, k=5)
    far = query_metrics([span("b", 0, 512)], gold, k=5)
    assert near["hit"] == 0 and near["doc_hit"] == 1
    assert far["hit"] == 0 and far["doc_hit"] == 0