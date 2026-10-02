#!/usr/bin/env python3
"""P1 store-parity benchmark: is ArcadeDB LSM_VECTOR (JVector ANN) a faithful
substitute for the dense retrieval we run today, on the IDENTICAL bge-m3 vectors?

Ground truth = EXACT cosine k-NN computed with numpy over the full 28,693×1024
matrix (vectors are L2-normalized → cosine == dot product). We then measure how
much of that exact top-k each system recovers:

  * ArcadeDB  — vector.neighbors() over the loaded LSM_VECTOR index (HTTP :2490)
  * Chroma    — the live corpus via noted-rag /search_by_vector (HNSW, :8201)

Both are ANN, so both are scored against the same exact ground truth — an
apples-to-apples ANN-recall comparison. Queries are a seeded random sample of
real chunk vectors (self-hit excluded). Also reports ArcadeDB query latency.

Usage: python bench_parity.py [--sample 300] [--k 10]
"""
import argparse
import json
import os
import time

import httpx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(HERE, "snapshot")
ARCADE = os.getenv("ARCADE_URL", "http://localhost:2490")
ARCADE_AUTH = (os.getenv("ARCADE_USER", "root"), os.getenv("ARCADE_PW") or os.environ["ARCADEDB_PW"])
NOTED_RAG = os.getenv("NOTED_RAG_URL", "http://localhost:8201")
CORPUS = "bulario__corpus"


def arcade_neighbors(client, qvec, k):
    t = time.perf_counter()
    r = client.post(f"{ARCADE}/api/v1/command/poc", auth=ARCADE_AUTH,
                    json={"language": "sql",
                          "command": "SELECT chunk_id, distance FROM "
                          "(SELECT expand(vector.neighbors('Chunk[embedding]', :q, :k)))",
                          "params": {"q": qvec, "k": k}}, timeout=30)
    dt = (time.perf_counter() - t) * 1000
    r.raise_for_status()
    rows = r.json().get("result", [])
    return [row["chunk_id"] for row in rows], dt


def chroma_by_vector(client, qvec, k):
    """Raw dense neighbors from the live Chroma (no rerank) for parity."""
    t = time.perf_counter()
    try:
        r = client.post(f"{NOTED_RAG}/search_by_vector",
                        json={"vector": qvec, "collection": CORPUS, "top_k": k,
                              "rerank_min_score": -1e9}, timeout=30)
        dt = (time.perf_counter() - t) * 1000
        r.raise_for_status()
        chunks = r.json().get("chunks") or []
        return [c.get("id") for c in chunks], dt
    except Exception as e:
        return None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    emb = np.load(os.path.join(SNAP, "embeddings.f32.npy"))  # [N,1024] normalized
    ids = [json.loads(l)["id"] for l in open(os.path.join(SNAP, "records.jsonl"))]
    N = emb.shape[0]
    assert len(ids) == N
    rng = np.random.default_rng(args.seed)
    qidx = rng.choice(N, size=min(args.sample, N), replace=False)

    client = httpx.Client()
    k = args.k
    arc_rec, chr_rec, arc_ms, chr_ms = [], [], [], []
    chroma_ok = True
    for n, qi in enumerate(qidx, 1):
        qvec = emb[qi]
        # EXACT top-(k+1) by cosine; drop self; keep k.
        sims = emb @ qvec
        top = np.argpartition(-sims, k + 1)[:k + 1]
        top = top[np.argsort(-sims[top])]
        exact = [ids[j] for j in top if j != qi][:k]
        exact_set = set(exact)

        a_ids, a_ms = arcade_neighbors(client, [float(x) for x in qvec], k + 1)
        a_ids = [c for c in a_ids if c != ids[qi]][:k]
        arc_rec.append(len(exact_set & set(a_ids)) / k)
        arc_ms.append(a_ms)

        if chroma_ok:
            c_ids, c_ms = chroma_by_vector(client, [float(x) for x in qvec], k + 1)
            if c_ids is None:
                chroma_ok = False
                print(f"  (chroma /search_by_vector unavailable: {c_ms}; skipping chroma leg)")
            else:
                c_ids = [c for c in c_ids if c != ids[qi]][:k]
                chr_rec.append(len(exact_set & set(c_ids)) / k)
                chr_ms.append(c_ms)
        if n % 50 == 0:
            print(f"  {n}/{len(qidx)}", flush=True)

    def pct(x): return round(100 * float(np.mean(x)), 2)
    def p50(x): return round(float(np.percentile(x, 50)), 1)
    def p95(x): return round(float(np.percentile(x, 95)), 1)

    out = {
        "sample": len(qidx), "k": k, "corpus": N,
        "arcadedb": {"ann_recall_vs_exact_pct": pct(arc_rec),
                     "latency_ms_p50": p50(arc_ms), "latency_ms_p95": p95(arc_ms)},
    }
    if chr_rec:
        out["chroma"] = {"ann_recall_vs_exact_pct": pct(chr_rec),
                         "latency_ms_p50": p50(chr_ms), "latency_ms_p95": p95(chr_ms)}
    print("\n=== P1 PARITY RESULT ===")
    print(json.dumps(out, indent=2))
    os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
    with open(os.path.join(HERE, "..", "results", "parity_latest.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
