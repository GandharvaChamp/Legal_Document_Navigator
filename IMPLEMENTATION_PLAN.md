Phase 1: Data and eval set

Build:

prepare_data.py: load the four benchmark files, tag each query with its source (CUAD, MAUD, ContractNLI, PrivacyQA), and create a stratified dev/test split with a fixed seed. Stratify by source so every subset appears in both splits.
tests/test_span_overlap.py: unit tests for span overlap, written before any retrieval code.
no_retrieval_baseline.py: ask the generator each dev question with no context and judge the answers.

Decisions to make here:

What counts as a retrieval hit: a chunk overlapping the gold span at all, or by a minimum fraction. I’ll recommend a rule, and it stays fixed for the whole project.
Whether to keep or strip the “Consider <document title>;” prefix. The default is to strip it for the main experiments and run keep-vs-strip as one separate ablation.
Whether the benchmark files are the 776-query mini set. If not, we sample 194 per source with a fixed seed.

Exit check: split files saved, metric tests pass, no-retrieval baseline scored, leakage check done (gold snippets are not in any index or prompt).

Phase 2: Component menu

For each component I’ll list every technique labelled MUST LEARN, OPTIONAL or SKIP. My preview:

Chunking: fixed. The chunking method is not what this project studies, so it stays constant.
Embeddings: one fixed model for the whole project.
Retrieval: MUST LEARN: dense, BM25, RRF, weighted fusion. OPTIONAL: BM25 tokenization variants, MMR.
Reranking: MUST LEARN: cross-encoder. OPTIONAL: LLM listwise reranking. SKIP: ColBERT (a later project).
Query transformation, parent-doc, self-correction: SKIP, they belong to later projects.
Phase 3: Hypotheses

Before running anything, you write what you expect each change to do and why, in analysis/hypotheses.md. I’ll challenge weak ones. Examples of what I’ll look for: which subset you expect BM25 to help most, whether you expect the reranker to help recall@k or only precision at the top, and what you expect hybrid to do to latency.

Phase 4: Baseline
Ingest and chunk, recording each chunk’s file_path and character offsets.
Build the embedding index and BM25 index once, stored under data/indexes/.
Run dense-only retrieval at top-k 10 on the dev split.
Log everything: config, retrieval metrics, latency and index size go into experiment_table.csv.

Exit check: baseline scored with confidence intervals, per-subset and pooled.

Phase 5: Ablations (one change per run, against the same baseline)
Run	Change from baseline
A0	Baseline: dense only
A1	BM25 only
A2	Hybrid, RRF fusion
A3	Hybrid, weighted fusion (weight tuned on dev only)
A4	Dense + cross-encoder rerank
A5	Hybrid RRF + cross-encoder rerank
A6	Rerank depth: top_n 20 vs 50 vs 100
A7	Query prefix kept vs stripped, on baseline and on the best config
A8 (optional)	BM25 tokenization / stopword variants

Only the best two or three retrieval configs get the generation step. Running generation and judging on all of them is expensive and adds little at this stage.

Phase 6: Evaluation and error analysis
Run the best config once on the test split. This is the only test-set run.
Report distributions, per-subset results and bootstrap 95% confidence intervals, not only averages.
Manually review 30 to 50 failures and classify them with the taxonomy (retrieval miss, ranking miss, chunk fragmentation, and so on).
Calibrate the LLM judge on 20 to 30 hand-labelled samples. The judge must be from a different model family than the generator.
Compare against the no-retrieval baseline and the Project 1 best config (by relative gain only, since the domain is different).
Phase 7 and 8: Understanding note and exit gate

You write the note on what worked and why. I review it for gaps and ask 3 to 5 questions. We don’t move to Project 3 until the gate checks pass.

Build order (what we code, in sequence)
config.py and run_experiment.py skeleton (done)
Span-overlap function with tests
prepare_data.py and the dev/test split
ingest.py and chunk.py with offsets
embed_index.py: dense index, then BM25 index
retrieve.py: dense, then BM25, then hybrid
retrieval_metrics.py and bootstrap.py
rerank.py
generate.py and the judge
Error analysis tooling



rag-series/
└── project2_hybrid_rerank/
    ├── README.md                  # problem statement, dataset, licenses, how to run
    ├── requirements.txt           # pinned versions (pip freeze after install)
    ├── .gitignore                 # data/raw, data/processed, indexes, .env
    ├── .env                       # API keys (never commit)
    │
    ├── data/
    │   ├── raw/legalbench_rag/    # downloaded corpus + benchmarks (read-only)
    │   ├── processed/             # chunks.jsonl, dev.jsonl, test.jsonl
    │   └── indexes/               # faiss / bm25 artifacts, one subfolder per config
    │
    ├── configs/
    │   ├── baseline.yaml          # dense-only baseline
    │   ├── bm25.yaml
    │   ├── hybrid_rrf.yaml
    │   ├── hybrid_weighted.yaml
    │   └── hybrid_rrf_rerank.yaml
    │
    ├── src/
    │   ├── config.py              # dataclass loader, seeds
    │   ├── ingest.py              # read corpus files
    │   ├── chunk.py               # chunking, keeps file_path + char offsets
    │   ├── embed_index.py         # embeddings, FAISS and BM25 build
    │   ├── retrieve.py            # dense, bm25, hybrid (RRF / weighted)
    │   ├── rerank.py              # cross-encoder reranker
    │   ├── generate.py            # prompt + LLM call
    │   ├── evaluate/
    │   │   ├── retrieval_metrics.py   # span overlap, recall@k, precision@k, MRR, nDCG
    │   │   ├── generation_metrics.py  # LLM judge (different family than generator)
    │   │   └── bootstrap.py           # confidence intervals
    │   └── utils.py               # logging, timing, token/cost counters
    │
    ├── scripts/
    │   ├── prepare_data.py        # build chunks, dev/test split (fixed seed)
    │   ├── no_retrieval_baseline.py
    │   ├── run_experiment.py      # --config configs/xxx.yaml
    │   └── judge_calibration.py   # 20-30 hand-labelled samples vs judge
    │
    ├── experiments/
    │   ├── experiment_table.csv   # one row per run: config, metrics, latency, cost
    │   └── runs/<run_id>/         # config copy, retrieved results, answers, logs
    │
    ├── analysis/
    │   ├── error_analysis.csv     # 30-50 failures with taxonomy label
    │   └── hypotheses.md          # written BEFORE running ablations
    │
    ├── notes/
    │   └── understanding_note.md  # Phase 7
    │
    └── tests/
        └── test_span_overlap.py   # unit test the metric first