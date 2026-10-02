#!/usr/bin/env python3
"""Answer-quality benchmark: for grounded doctor questions, does each retriever
surface the RIGHT leaflet (doc-level relevance)? Compares, on identical queries:

  * current flow  = noted-rag /search  (Chroma dense + bge-m3 rerank)   [production]
  * arcade_dense  = ArcadeDB vector.neighbors (LSM_VECTOR, NO rerank)

Relevance is doc-level: a hit = a retrieved chunk whose source_path contains the
case's expected drug (expected_source_substr). Metrics: recall@k (relevant doc
in top-k) and MRR (reciprocal rank of first relevant hit). Query vectors come
from noted-rag /embed (same bge-m3) so both sides use identical vectors.

Gold set = grounded cases from the medical test suite, filtered to drugs present
in the loaded snapshot.
"""
import json
import os
import sys

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(HERE, "snapshot")
ARCADE = "http://localhost:2490"
ARCADE_AUTH = ("root", os.environ["ARCADEDB_PW"])
NOTED_RAG = "http://localhost:8201"
CORPUS = "bulario__corpus"
K = 10

sys.path.insert(0, "/home/logus/env/labs/medical/tests")
import expected_answers as ea  # noqa: E402

_c = httpx.Client(timeout=60)


def embed(text):
    r = _c.post(f"{NOTED_RAG}/embed", json={"texts": [text]})
    r.raise_for_status()
    return r.json()["vectors"][0]


def current_flow(query):
    r = _c.post(f"{NOTED_RAG}/search",
                json={"query": query, "collection": CORPUS, "top_k": K,
                      "rerank_min_score": -1e9})
    r.raise_for_status()
    return [c.get("source_path", "") for c in (r.json().get("chunks") or [])]


def arcade_dense(qvec):
    r = _c.post(f"{ARCADE}/api/v1/command/poc", auth=ARCADE_AUTH,
                json={"language": "sql",
                      "command": "SELECT source_path FROM "
                      "(SELECT expand(vector.neighbors('Chunk[embedding]', :q, :k)))",
                      "params": {"q": qvec, "k": K}})
    r.raise_for_status()
    return [row.get("source_path", "") for row in r.json().get("result", [])]


def first_hit_rank(paths, substr):
    for i, p in enumerate(paths, 1):
        if substr in (p or "").lower():
            return i
    return 0


def snapshot_drugs():
    drugs = set()
    for line in open(os.path.join(SNAP, "records.jsonl")):
        drugs.add(json.loads(line)["metadata"].get("source_path", "").lower())
    return drugs


def main():
    corpus_paths = snapshot_drugs()
    def present(sub): return any(sub in p for p in corpus_paths)

    cases = [c for c in ea.ALL_CASES
             if c.expected_source_substr and present(c.expected_source_substr.lower())]
    print(f"gold cases usable (drug in snapshot): {len(cases)} / {len(ea.ALL_CASES)}")

    agg = {"current": {"hit": 0, "rr": 0.0}, "arcade": {"hit": 0, "rr": 0.0}}
    rows = []
    for c in cases:
        sub = c.expected_source_substr.lower()
        qvec = embed(c.question)
        cf = current_flow(c.question)
        ad = arcade_dense(qvec)
        r_cf = first_hit_rank(cf, sub)
        r_ad = first_hit_rank(ad, sub)
        agg["current"]["hit"] += 1 if r_cf else 0
        agg["arcade"]["hit"] += 1 if r_ad else 0
        agg["current"]["rr"] += (1.0 / r_cf) if r_cf else 0.0
        agg["arcade"]["rr"] += (1.0 / r_ad) if r_ad else 0.0
        rows.append({"id": c.id, "drug": sub, "cat": c.category,
                     "rank_current": r_cf, "rank_arcade": r_ad})

    n = len(cases)
    out = {
        "corpus": len(corpus_paths), "gold_cases": n, "k": K,
        "current_flow_chroma_rerank": {
            "recall_at_k_pct": round(100 * agg["current"]["hit"] / n, 1),
            "mrr": round(agg["current"]["rr"] / n, 3)},
        "arcade_dense_no_rerank": {
            "recall_at_k_pct": round(100 * agg["arcade"]["hit"] / n, 1),
            "mrr": round(agg["arcade"]["rr"] / n, 3)},
    }
    print(json.dumps(out, indent=2))
    print("\nper-case (rank of first correct-drug chunk; 0=miss):")
    for r in rows:
        print(f"  {r['drug']:13} {r['cat']} {r['id']:22} "
              f"current={r['rank_current']:>2}  arcade={r['rank_arcade']:>2}")
    with open(os.path.join(HERE, "..", "results", "quality_1000docs.json"), "w") as f:
        json.dump({"summary": out, "per_case": rows}, f, indent=2)


if __name__ == "__main__":
    main()
