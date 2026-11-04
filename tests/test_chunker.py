import sys

import pytest

sys.path.insert(0, "src")
from chunking.chunker import chunk_text, make_chunks


def covered(text, spans):
    cov = [False] * len(text)
    for s, e in spans:
        for i in range(s, e):
            cov[i] = True
    return cov


def test_empty_and_whitespace_only():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    t = "This Agreement shall terminate on December 31."
    assert chunk_text(t) == [(0, len(t))]


def test_bad_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_paragraph_boundaries_are_preferred():
    p1 = "A" * 300 + "\n\n"
    p2 = "B" * 300
    t = p1 + p2
    spans = chunk_text(t, size=512, overlap=64)
    assert len(spans) == 2
    assert t[spans[0][0]:spans[0][1]] == "A" * 300      # trimmed paragraph 1
    assert t[spans[1][0]:spans[1][1]] == "B" * 300


def test_text_without_separators_is_hard_cut():
    t = "x" * 1300
    spans = chunk_text(t, size=500, overlap=0)
    assert all(e - s <= 500 for s, e in spans)
    assert all(covered(t, spans))


def test_invariants_on_realistic_text():
    para = ("The Company shall not disclose Confidential Information to any third party. "
            "This obligation survives termination of this Agreement. ") * 4
    t = "\n\n".join(f"{i}. {para}" for i in range(60))
    size, overlap = 512, 64
    spans = chunk_text(t, size, overlap)

    assert all(0 < e - s <= size for s, e in spans)                  # size limit
    assert spans == sorted(spans)                                    # in order
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert s2 >= s1 and e2 > e1                                  # moves forward
        assert max(0, e1 - s2) <= overlap                            # overlap limit
    cov = covered(t, spans)
    assert all(c or ch.isspace() for c, ch in zip(cov, t))           # nothing lost


def test_chunk_text_matches_offsets_and_keys():
    docs = {"b/two.txt": "Second contract.\n\nMore text here.", "a/one.txt": "First."}
    chunks = make_chunks(docs, size=512, overlap=64)
    assert [c["doc"] for c in chunks] == ["a/one.txt", "b/two.txt"]
    assert [c["cid"] for c in chunks] == [0, 1]
    for c in chunks:
        assert docs[c["doc"]][c["start"]:c["end"]] == c["text"]