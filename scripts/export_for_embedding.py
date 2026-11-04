import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from ingestion.loader import load_corpus, load_queries

RAW = Path("data/raw")
OUT = Path("data/processed/queries_for_embedding.jsonl")
SPLIT = re.compile(r"^Consider (?P<title>.+?);\s+(?P<question>.+)$", re.S)

docs = load_corpus(RAW / "corpus")
queries, skipped = load_queries(RAW / "benchmarks", docs)
assert skipped == 0

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for q in queries:
        m = SPLIT.match(q["query"])
        assert m, q["qid"]
        row = {"qid": q["qid"], "query": q["query"], "question": m["question"]}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("wrote", len(queries), "queries to", OUT)