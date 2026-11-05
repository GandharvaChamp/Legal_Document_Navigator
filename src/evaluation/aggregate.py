"""Per-question scoring, stratified bootstrap, paired bootstrap."""
import csv

import numpy as np

from evaluation.retrieval_metrics import query_metrics

SOURCES = ["privacy_qa", "contractnli", "cuad", "maud"]
METRICS = ["recall", "hit", "precision", "mrr", "ndcg", "doc_hit"]
KS = [1, 5, 10, 20, 50, 100]
GROUPS = ["pooled", "macro"] + SOURCES


def col(metric, k, ks=KS, metrics=METRICS):
    """Column index of metric@k in the per-question matrix (metric-major, k-minor)."""
    return metrics.index(metric) * len(ks) + ks.index(k)


def column_names(ks=KS, metrics=METRICS):
    return [f"{m}@{k}" for m in metrics for k in ks]


def compute_per_query(queries, ids, chunk_meta, ks=KS, metrics=METRICS, rule="half_shorter"):
    """queries: dicts with 'gold'. ids: (n, depth) chunk row numbers, best first.
    chunk_meta: list of {'doc','start','end'} indexed by chunk row.
    Returns X of shape (n, len(metrics) * len(ks))."""
    X = np.zeros((len(queries), len(metrics) * len(ks)), dtype=np.float32)
    for i, q in enumerate(queries):
        retrieved = [chunk_meta[j] for j in ids[i]]
        for ki, k in enumerate(ks):
            r = query_metrics(retrieved, q["gold"], k=k, rule=rule)
            for mi, m in enumerate(metrics):
                X[i, mi * len(ks) + ki] = r[m]
    return X


def bootstrap(X, sources, n_boot=1000, seed=42):
    """Stratified bootstrap: questions are resampled WITHIN each source, so every
    source keeps its size. Returns (point, boot): dicts group -> (M,) / (n_boot, M).
    The same seed and the same questions give the same resamples, which is what
    makes a paired comparison possible (bootstrap the difference of two runs)."""
    X = np.asarray(X, dtype=np.float64)
    sources = np.asarray(sources)
    rng = np.random.default_rng(seed)
    n_total, M = X.shape
    present = [s for s in SOURCES if (sources == s).any()]
    boot, point = {}, {}
    pooled_b = np.zeros((n_boot, M))
    macro_b = np.zeros((n_boot, M))
    for s in present:
        Xs = X[sources == s]
        n_s = len(Xs)
        R = rng.integers(0, n_s, size=(n_boot, n_s))
        means = np.empty((n_boot, M))
        for j in range(M):
            means[:, j] = Xs[:, j][R].mean(axis=1)
        boot[s] = means
        point[s] = Xs.mean(axis=0)
        pooled_b += means * (n_s / n_total)
        macro_b += means / len(present)
    boot["pooled"], boot["macro"] = pooled_b, macro_b
    point["pooled"] = X.mean(axis=0)
    point["macro"] = np.mean([point[s] for s in present], axis=0)
    return point, boot


def summarize(X, sources, n_boot=1000, seed=42, ci=95):
    """group -> (point, lo, hi), each an array of length M."""
    point, boot = bootstrap(X, sources, n_boot, seed)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {g: (point[g],
                np.percentile(boot[g], lo_q, axis=0),
                np.percentile(boot[g], hi_q, axis=0)) for g in point}


def print_summary(summary, k=10, ks=KS, metrics=METRICS):
    groups = [g for g in GROUPS if g in summary]
    print(f"\n=== point estimates at k={k} ===")
    print(f"{'group':<13}" + "".join(f"{m:>11}" for m in metrics))
    for g in groups:
        p = summary[g][0]
        print(f"{g:<13}" + "".join(f"{p[col(m, k, ks, metrics)]:>11.3f}" for m in metrics))

    print(f"\n=== 95% intervals at k={k} (macro, pooled) ===")
    for g in ("macro", "pooled"):
        p, lo, hi = summary[g]
        for m in metrics:
            j = col(m, k, ks, metrics)
            print(f"{g:<8}{m:<10}{p[j]:.3f}  [{lo[j]:.3f}, {hi[j]:.3f}]")

    for m in ("recall", "hit"):
        print(f"\n=== {m}@k by depth (point estimates) ===")
        print(f"{'group':<13}" + "".join(f"{'@' + str(kk):>8}" for kk in ks))
        for g in groups:
            p = summary[g][0]
            print(f"{g:<13}" + "".join(f"{p[col(m, kk, ks, metrics)]:>8.3f}" for kk in ks))


def save_per_query(path, qids, sources, X, ks=KS, metrics=METRICS):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "source"] + column_names(ks, metrics))
        for qid, s, row in zip(qids, sources, X):
            w.writerow([qid, s] + [f"{v:.6f}" for v in row])


def load_per_query(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    qids = [x[0] for x in rows]
    sources = np.array([x[1] for x in rows])
    X = np.array([[float(v) for v in x[2:]] for x in rows], dtype=np.float64)
    return qids, sources, X, header[2:]