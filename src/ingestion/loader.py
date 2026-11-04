"""Single entry point for reading the LegalBench-RAG corpus and benchmarks.

Rules this module enforces for the whole project:
1. Corpus files are read ONLY through read_text() (default newline handling,
   because the benchmark offsets were built on \n-normalized text).
2. File paths are matched ONLY through path_key() (Unicode NFC + Windows-safe
   characters), because file names differ between the benchmark and Windows.
3. Queries whose gold document is missing are skipped, and the count is
   returned, never hidden.
"""
import json
import re
import unicodedata
from pathlib import Path

SOURCES = ["privacy_qa", "contractnli", "cuad", "maud"]
_ILLEGAL = re.compile(r'[<>:"|?*]')


def path_key(p: str) -> str:
    """Normalize a path so benchmark paths match Windows file names."""
    return _ILLEGAL.sub("_", unicodedata.normalize("NFC", p))


def read_text(path) -> str:
    # Default newline handling on purpose: \r\n -> \n, as in the benchmark.
    with open(path, encoding="utf-8") as f:
        return f.read()


def source_of(key: str) -> str:
    """'cuad/Some Contract.txt' -> 'cuad'"""
    return key.split("/", 1)[0]


def load_corpus(corpus_dir) -> dict[str, str]:
    """Return {path_key: full text} for every .txt file under corpus_dir."""
    corpus_dir = Path(corpus_dir)
    return {
        path_key(p.relative_to(corpus_dir).as_posix()): read_text(p)
        for p in sorted(corpus_dir.rglob("*.txt"))
    }


def load_queries(benchmarks_dir, docs: dict[str, str]):
    """Return (queries, n_skipped).

    Each query: {qid, source, query, gold: [{doc, start, end, answer}]}.
    A query is skipped if ANY of its gold documents is missing from docs.
    """
    queries, skipped = [], 0
    for src in SOURCES:
        with open(Path(benchmarks_dir) / f"{src}.json", encoding="utf-8") as f:
            tests = json.load(f)["tests"]
        for i, t in enumerate(tests):
            gold = []
            for s in t["snippets"]:
                k = path_key(s["file_path"])
                if k not in docs:
                    gold = None
                    break
                gold.append({"doc": k, "start": s["span"][0],
                             "end": s["span"][1], "answer": s["answer"]})
            if gold is None:
                skipped += 1
                continue
            queries.append({"qid": f"{src}-{i}", "source": src,
                            "query": t["query"], "gold": gold})
    return queries, skipped


def verify(queries, docs) -> int:
    """Count gold spans whose text differs from the stored answer. Must be 0."""
    return sum(
        docs[g["doc"]][g["start"]:g["end"]] != g["answer"]
        for q in queries for g in q["gold"]
    )


if __name__ == "__main__":
    import collections
    import sys

    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    docs = load_corpus(raw / "corpus")
    queries, skipped = load_queries(raw / "benchmarks", docs)

    print("documents:", len(docs))
    print("documents per source:", dict(collections.Counter(source_of(k) for k in docs)))
    print("queries kept:", len(queries), "| skipped:", skipped)
    print("queries per source:", dict(collections.Counter(q["source"] for q in queries)))
    print("offset mismatches:", verify(queries, docs))
    print("total characters:", sum(len(t) for t in docs.values()))