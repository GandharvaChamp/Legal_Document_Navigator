"""Hit rule + per-query retrieval metrics.

Chunk and gold are dicts with keys: doc, start, end (character offsets).
`retrieved` is a list of chunks ordered best-first.
"""
import math


def overlap_chars(a_start, a_end, b_start, b_end) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def is_hit(chunk, gold, rule="half_shorter") -> bool:
    """A chunk is a hit for a gold clause if it is from the same document AND
    overlaps by at least half of the shorter of the two spans.
    rule="any_overlap" is the lenient robustness check (>= 1 character)."""
    if rule not in ("half_shorter", "any_overlap"):
        raise ValueError(f"unknown rule: {rule}")
    if chunk["doc"] != gold["doc"]:
        return False
    ov = overlap_chars(chunk["start"], chunk["end"], gold["start"], gold["end"])
    if ov == 0:
        return False
    if rule == "any_overlap":
        return True
    shorter = min(chunk["end"] - chunk["start"], gold["end"] - gold["start"])
    return shorter > 0 and ov >= 0.5 * shorter


def query_metrics(retrieved, gold_list, k=10, rule="half_shorter") -> dict:
    retrieved = retrieved[:k]

    # earliest rank (1-based) at which each gold clause is covered
    first_rank = []
    for g in gold_list:
        r = next((i for i, c in enumerate(retrieved, 1) if is_hit(c, g, rule)), None)
        first_rank.append(r)
    covered = [r for r in first_rank if r is not None]

    relevant_chunks = sum(
        any(is_hit(c, g, rule) for g in gold_list) for c in retrieved
    )
    gold_docs = {g["doc"] for g in gold_list}

    dcg = sum(1 / math.log2(r + 1) for r in covered)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(gold_list)) + 1))

    return {
        "recall": len(covered) / len(gold_list),       # share of gold clauses found
        "hit": 1.0 if covered else 0.0,                # any gold clause found
        "precision": relevant_chunks / k,              # share of top-k that are relevant
        "mrr": 1 / min(covered) if covered else 0.0,   # rank of first relevant chunk
        "ndcg": dcg / ideal if ideal else 0.0,         # credit once per gold clause
        "doc_hit": 1.0 if any(c["doc"] in gold_docs for c in retrieved) else 0.0,
    }