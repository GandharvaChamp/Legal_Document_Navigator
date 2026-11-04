"""Chunk the corpus, save it, and measure the chunking ceiling.
Run from the project root:
    python scripts/build_chunks.py --config configs/baseline.yaml
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from config import load
from evaluation.retrieval_metrics import is_hit
from chunking.chunker import make_chunks
from ingestion.loader import load_corpus, load_queries, source_of

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/baseline.yaml")
args = p.parse_args()
cfg = load(args.config)
assert cfg.chunking.method == "recursive", "this script only builds recursive chunks"
size, overlap = cfg.chunking.chunk_size, cfg.chunking.overlap

docs = load_corpus(cfg.data.corpus_dir)
t0 = time.time()
chunks = make_chunks(docs, size, overlap)
build_s = time.time() - t0

out = Path("data/processed") / f"chunks_recursive_{size}_{overlap}.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

# ---- sanity: every chunk's text is exactly its range of the document
bad = sum(docs[c["doc"]][c["start"]:c["end"]] != c["text"] for c in chunks)

# ---- size statistics
L = sorted(c["end"] - c["start"] for c in chunks)
pct = lambda q: L[int(len(L) * q)]
print(f"chunks: {len(chunks)} | build time: {build_s:.0f}s | offset/text mismatches: {bad}")
print(f"chunk length (chars): mean={sum(L)/len(L):.0f} median={L[len(L)//2]} "
      f"p5={pct(0.05)} p95={pct(0.95)} max={L[-1]}")
per_src = collections.Counter(source_of(c["doc"]) for c in chunks)
print("chunks per source:", dict(per_src))

# ---- ceiling: share of gold clauses that at least one chunk can hit
queries, skipped = load_queries(Path(cfg.data.benchmarks_dir), docs)
by_doc = collections.defaultdict(list)
for c in chunks:
    by_doc[c["doc"]].append(c)

tot = collections.Counter()
hit = collections.Counter()
inside = collections.Counter()
for q in queries:
    for g in q["gold"]:
        tot[q["source"]] += 1
        cs = by_doc[g["doc"]]
        if any(is_hit(c, g) for c in cs):
            hit[q["source"]] += 1
        if any(c["start"] <= g["start"] and c["end"] >= g["end"] for c in cs):
            inside[q["source"]] += 1

print("\nchunking ceiling (gold clauses that some chunk can hit under the hit rule):")
print("source         gold   can_hit   fully_inside_one_chunk")
for src in sorted(tot):
    print(f"{src:<12} {tot[src]:>6} {hit[src]/tot[src]:>8.1%} {inside[src]/tot[src]:>14.1%}")

json.dump({"size": size, "overlap": overlap, "n_chunks": len(chunks),
           "build_seconds": round(build_s, 1), "chunk_len_mean": sum(L) / len(L),
           "ceiling": {s: hit[s] / tot[s] for s in tot}},
          open(Path("data/processed") / f"chunks_recursive_{size}_{overlap}_meta.json", "w"),
          indent=2)
print("\nsaved:", out)