"""BM25 retrieval over the chunk texts (bm25s library, Lucene-style BM25).

Decided up front (one change per experiment): lowercase word tokens,
NO stopword removal, NO stemming, default k1 and b.
"""
import time

import bm25s
import numpy as np


def build(texts, k1=1.5, b=0.75):
    """Tokenize the chunk texts and build the index. Returns (retriever, seconds)."""
    t0 = time.perf_counter()
    tokens = bm25s.tokenize(texts, stopwords=None, stemmer=None)
    retriever = bm25s.BM25(method="lucene", k1=k1, b=b)
    retriever.index(tokens)
    return retriever, time.perf_counter() - t0


def tokenize_queries(queries):
    """List of query strings -> list of token-string lists (same rules as the corpus)."""
    return bm25s.tokenize(queries, stopwords=None, stemmer=None, return_ids=False)


def search(retriever, query_tokens, k=100):
    """Returns (ids, scores), each of shape (n_queries, k), best first."""
    ids, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)
    return np.asarray(ids, dtype=np.int32), np.asarray(scores, dtype=np.float32)


def time_single(retriever, query_tokens, n=200, k=100):
    """Per-question latency (batch size 1). Returns (p50_ms, p95_ms)."""
    for q in query_tokens[:3]:                      # warm-up
        search(retriever, [q], k)
    ms = []
    for q in query_tokens[:n]:
        t = time.perf_counter()
        search(retriever, [q], k)
        ms.append((time.perf_counter() - t) * 1000)
    return float(np.percentile(ms, 50)), float(np.percentile(ms, 95))