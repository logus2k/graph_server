#!/usr/bin/env python3
"""Load the frozen Chroma snapshot into ArcadeDB (26.7.2, PoC :2490), REUSING
the exact bge-m3 vectors. Bulk-insert as documents, THEN build the LSM_VECTOR
index over the full set (faster + the recommended pattern; verified that
load-then-index picks up existing rows).

Join key: `chunk_id` == the Chroma id, so results map 1:1 back to the corpus.

Usage:
  python load_arcadedb.py [--limit N] [--batch 200] [--reset]
"""
import argparse
import json
import os
import time

import httpx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.getenv("SNAP_DIR", os.path.join(HERE, "snapshot"))
URL = os.getenv("ARCADE_URL", "http://localhost:2490")
DB = os.getenv("ARCADE_DB", "poc")
AUTH = (os.getenv("ARCADE_USER", "root"), os.getenv("ARCADE_PW") or os.environ["ARCADEDB_PW"])
DIM = 1024

_c = httpx.Client(timeout=180)


def cmd(command, params=None, language="sql"):
    body = {"language": language, "command": command}
    if params is not None:
        body["params"] = params
    r = _c.post(f"{URL}/api/v1/command/{DB}", json=body, auth=AUTH)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    j = r.json()
    if "error" in j:
        raise RuntimeError(f"{j.get('exception')}: {j.get('detail')}")
    return j.get("result", [])


def meta_get(m, k, default=None):
    v = m.get(k, default)
    return v if v not in ("", None) else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(SNAP, "manifest.json")))
    print("manifest:", manifest)
    emb = np.load(os.path.join(SNAP, "embeddings.f32.npy"), mmap_mode="r")
    n = args.limit or emb.shape[0]

    if args.reset:
        cmd("DROP TYPE Chunk IF EXISTS UNSAFE")
    # Sparse (bge-m3 lexical) is optional — included only if gen_sparse.py has run.
    sparse_path = os.path.join(SNAP, "sparse.jsonl")
    has_sparse = os.path.exists(sparse_path)
    cmd("CREATE VERTEX TYPE Chunk IF NOT EXISTS")
    props = [("chunk_id", "STRING"), ("text", "STRING"),
             ("source_path", "STRING"), ("section_path", "STRING"),
             ("page_no", "INTEGER"), ("embedding", "ARRAY_OF_FLOATS")]
    if has_sparse:
        props += [("sidx", "ARRAY_OF_INTEGERS"), ("swt", "ARRAY_OF_FLOATS")]
    for prop, typ in props:
        cmd(f"CREATE PROPERTY Chunk.{prop} IF NOT EXISTS {typ}")
    cmd("CREATE INDEX IF NOT EXISTS ON Chunk (chunk_id) UNIQUE")
    # LSM_SPARSE_VECTOR does NOT backfill pre-existing rows (unlike LSM_VECTOR),
    # so it MUST be created BEFORE the bulk load — rows are indexed on insert.
    if has_sparse:
        cmd('CREATE INDEX ON Chunk (sidx, swt) LSM_SPARSE_VECTOR '
            'METADATA {"dimensions":250002,"modifier":"IDF"}')
    print(f"sparse: {'INCLUDED (index-first)' if has_sparse else 'not present (dense-only)'}")

    t0 = time.time()
    loaded = 0
    sf = open(sparse_path) if has_sparse else None
    with open(os.path.join(SNAP, "records.jsonl")) as f:
        batch = []
        for i in range(n):
            rec = json.loads(f.readline())
            m = rec["metadata"] or {}
            pg = meta_get(m, "page_no")
            doc = {
                "chunk_id": rec["id"],
                "text": rec["text"],
                "source_path": meta_get(m, "source_path", ""),
                "section_path": meta_get(m, "section_path", ""),
                "page_no": int(pg) if isinstance(pg, (int, float)) else -1,
                "embedding": [float(x) for x in emb[i]],
            }
            if sf is not None:
                sp = json.loads(sf.readline())
                doc["sidx"] = sp["indices"]
                doc["swt"] = sp["weights"]
            batch.append(doc)
            if len(batch) >= args.batch or i == n - 1:
                # One INSERT per row in a sqlscript, each doc a BOUND param
                # (:dK). Params are parsed as JSON, not as SQL-literal float
                # arrays — a literal array blows the SQL parser's heap at scale.
                stmts, params = [], {}
                for k, doc in enumerate(batch):
                    stmts.append(f"INSERT INTO Chunk CONTENT :d{k};")
                    params[f"d{k}"] = doc
                cmd("\n".join(stmts), params=params, language="sqlscript")
                loaded += len(batch)
                batch = []
                if loaded % 2000 == 0 or i == n - 1:
                    print(f"  loaded {loaded}/{n} ({loaded/(time.time()-t0):.0f}/s)", flush=True)

    print(f"insert done in {time.time()-t0:.1f}s; building LSM_VECTOR index...")
    ti = time.time()
    res = cmd(f'CREATE INDEX ON Chunk (embedding) LSM_VECTOR '
              f'METADATA {{"dimensions":{DIM},"similarity":"COSINE"}}')
    print("dense index:", res, f"({time.time()-ti:.1f}s)")
    cnt = cmd("SELECT count(*) AS n FROM Chunk")
    print("VERIFY Chunk count:", cnt)


if __name__ == "__main__":
    main()
