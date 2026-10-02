#!/usr/bin/env python3
"""Chroma leg of the P1 parity test — runs INSIDE noted-rag (owns chromadb).
Uses the SAME frozen snapshot + SAME seed/N as bench_parity.py, so the sampled
query vectors are IDENTICAL to the ArcadeDB run → directly comparable ANN recall
vs the exact numpy ground truth. Chroma's raw HNSW via col.query (no rerank).

Reads /tmp/snap/{embeddings.f32.npy,records.jsonl} (copied in). Prints JSON.
"""
import json
import time

import chromadb
import numpy as np

SNAP = "/tmp/snap"
import sys; SAMPLE, K, SEED = int(sys.argv[1]) if len(sys.argv)>1 else 300, 10, 42

emb = np.load(f"{SNAP}/embeddings.f32.npy")
ids = [json.loads(l)["id"] for l in open(f"{SNAP}/records.jsonl")]
N = emb.shape[0]
col = chromadb.PersistentClient("/data/vectordb").get_collection("bulario__corpus")

rng = np.random.default_rng(SEED)
qidx = rng.choice(N, size=min(SAMPLE, N), replace=False)

rec, ms = [], []
for qi in qidx:
    qvec = emb[qi]
    sims = emb @ qvec
    top = np.argpartition(-sims, K + 1)[:K + 1]
    top = top[np.argsort(-sims[top])]
    exact = set(ids[j] for j in top if j != qi)
    exact = set(list(exact)[:K])

    t = time.perf_counter()
    g = col.query(query_embeddings=[[float(x) for x in qvec]], n_results=K + 1,
                  include=[])
    ms.append((time.perf_counter() - t) * 1000)
    got = [c for c in (g["ids"][0]) if c != ids[qi]][:K]
    rec.append(len(exact & set(got)) / K)

print(json.dumps({
    "sample": len(qidx), "k": K, "corpus_snapshot": N,
    "chroma": {"ann_recall_vs_exact_pct": round(100 * float(np.mean(rec)), 2),
               "latency_ms_p50": round(float(np.percentile(ms, 50)), 1),
               "latency_ms_p95": round(float(np.percentile(ms, 95)), 1)},
}, indent=2))
