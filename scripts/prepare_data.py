"""Phase 1: dev/test split, LLM samples, leakage checks.
Run from the project root:  python scripts/prepare_data.py
"""
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from ingestion.loader import load_corpus, load_queries, verify, source_of

SEED = 42
DEV_FRAC = 0.30
LLM_TEST_PER_SOURCE = 50
LLM_DEV_PER_SOURCE = 15
CLAUSE_SAMPLE = 300          # gold clauses checked for verbatim copies elsewhere

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 1. load
docs = load_corpus(RAW / "corpus")
queries, skipped = load_queries(RAW / "benchmarks", docs)
mismatches = verify(queries, docs)
print(f"documents: {len(docs)} | queries: {len(queries)} | "
      f"skipped: {skipped} | offset mismatches: {mismatches}")
assert skipped == 0 and mismatches == 0, "fix the data before splitting"

# ------------------------------------------- 2. parse "Consider <title>; <question>"
PREFIX = re.compile(r"^Consider (?P<title>.+?);\s+(?P<question>.+)$", re.S)
unparsed = 0
multi_semicolon = 0
for q in queries:
    m = PREFIX.match(q["query"])
    q["title"] = m["title"] if m else None
    q["question"] = m["question"] if m else None
    unparsed += m is None
    multi_semicolon += q["query"].count("; ") > 1
print(f"prefix parsed: {len(queries) - unparsed} | unparsed: {unparsed} | "
      f"queries with more than one '; ': {multi_semicolon}")

# ------------------------------------- 3. split (stratified by source, groups kept whole)
by_src = collections.defaultdict(lambda: collections.defaultdict(list))
for q in queries:
    by_src[q["source"]][q["query"]].append(q)       # identical query texts stay together

rng = random.Random(SEED)
dev, test = [], []
for src in sorted(by_src):
    groups = by_src[src]
    texts = sorted(groups)
    rng.shuffle(texts)
    target = round(DEV_FRAC * sum(len(g) for g in groups.values()))
    n_dev = 0
    for t in texts:
        if n_dev < target:
            dev.extend(groups[t])
            n_dev += len(groups[t])
        else:
            test.extend(groups[t])


def stratified_sample(pool, per_source, seed):
    r = random.Random(seed)
    out = []
    for src in sorted({q["source"] for q in pool}):
        items = sorted((q for q in pool if q["source"] == src), key=lambda q: q["qid"])
        r.shuffle(items)
        out.extend(items[:per_source])
    return out


llm_test = stratified_sample(test, LLM_TEST_PER_SOURCE, SEED + 1)
llm_dev = stratified_sample(dev, LLM_DEV_PER_SOURCE, SEED + 2)
assert len(llm_test) == 4 * LLM_TEST_PER_SOURCE and len(llm_dev) == 4 * LLM_DEV_PER_SOURCE

# ---------------------------------------------------------------- 4. save
for name, split in [("dev", dev), ("test", test),
                    ("llm_dev_sample", llm_dev), ("llm_test_sample", llm_test)]:
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for q in split:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

json.dump(
    {"seed": SEED, "dev_frac": DEV_FRAC, "n_docs": len(docs), "n_queries": len(queries),
     "skipped": skipped, "n_dev": len(dev), "n_test": len(test),
     "n_llm_dev": len(llm_dev), "n_llm_test": len(llm_test)},
    open(OUT / "split_meta.json", "w", encoding="utf-8"), indent=2)

print("\nsource         dev   test  llm_dev  llm_test")
for src in sorted(by_src):
    row = [sum(q["source"] == src for q in s) for s in (dev, test, llm_dev, llm_test)]
    print(f"{src:<12} {row[0]:>5} {row[1]:>6} {row[2]:>8} {row[3]:>9}")
print(f"{'total':<12} {len(dev):>5} {len(test):>6} {len(llm_dev):>8} {len(llm_test):>9}")

# ------------------------------------------------------- 5. leakage checks
print("\n--- leakage checks ---")

# a) no identical query text on both sides
shared = {q["query"] for q in dev} & {q["query"] for q in test}
print("a) identical query texts in both dev and test:", len(shared))
assert not shared

# b) no question id on both sides; samples come from the right pile
assert not ({q["qid"] for q in dev} & {q["qid"] for q in test})
assert {q["qid"] for q in llm_test} <= {q["qid"] for q in test}
assert {q["qid"] for q in llm_dev} <= {q["qid"] for q in dev}
print("b) qid overlap dev/test: 0 | samples are subsets of their own pile: OK")

# c) exact duplicate contracts (same text under different names)
by_hash = collections.defaultdict(list)
for k, t in docs.items():
    by_hash[hashlib.sha256(t.encode("utf-8")).hexdigest()].append(k)
dups = [v for v in by_hash.values() if len(v) > 1]
print("c) groups of exact duplicate contracts:", len(dups))
for v in dups[:3]:
    print("   ", v)

# d) contracts that appear in gold for both dev and test (accepted limitation)
for src in sorted(by_src):
    d = {g["doc"] for q in dev if q["source"] == src for g in q["gold"]}
    t = {g["doc"] for q in test if q["source"] == src for g in q["gold"]}
    print(f"d) {src:<12} contracts in dev gold: {len(d):>3} | in test gold: {len(t):>3} "
          f"| in both: {len(d & t):>3}")

# e) gold clauses that also appear word-for-word in OTHER contracts
r = random.Random(SEED + 3)
snips = [(q["source"], g) for q in queries for g in q["gold"] if len(g["answer"]) >= 50]
sample = r.sample(snips, CLAUSE_SAMPLE)
has_copy = collections.Counter()
seen = collections.Counter()
for src, g in sample:
    seen[src] += 1
    if any(k != g["doc"] and g["answer"] in t for k, t in docs.items()):
        has_copy[src] += 1
print(f"e) gold clauses (sample of {CLAUSE_SAMPLE}) found verbatim in another contract:")
for src in sorted(seen):
    print(f"   {src:<12} {has_copy[src]:>3} of {seen[src]:>3}")