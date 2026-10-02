"""job2cool migration Phase A — copy Chroma dense vectors -> ArcadeDB (DENSE-ONLY,
no re-embed, no sparse). Run in the noted-rag image (chromadb 0.5.23) on noted-network:

  docker run --rm --network noted-network --entrypoint python \
    -v /home/logus/env/assets/noted/data/vectordb:/vdb:ro \
    -v /home/logus/env/assets/graph_server/harness:/mig \
    services-noted-rag /mig/job2cool_migrate.py <DB> <coll1>[,<coll2>,...]

Idempotency: creates db+schema IF NOT EXISTS; chunk_id NOTUNIQUE (content domains can
share ids). Re-run only against a freshly-(re)created db.
"""
import os
import sys, hashlib, json, time
import chromadb, httpx

GS = "http://graph-server-arcadedb:2480"; GA = ("root", os.environ["ARCADEDB_PW"])
DB = sys.argv[1]
COLLS = sys.argv[2].split(",")
BATCH = 500
c = httpx.Client(timeout=900)


def cmd(command, params=None, language="sql"):
    body = {"language": language, "command": command}
    if params:
        body["params"] = params
    r = c.post(f"{GS}/api/v1/command/{DB}", auth=GA, json=body)
    r.raise_for_status()
    return r.json().get("result")


# 1. create db (tolerate exists)
r = c.post(f"{GS}/api/v1/server", auth=GA, json={"command": f"create database {DB}"})
print("create db:", r.status_code, r.text[:120], flush=True)

# 2. schema (DENSE-ONLY: no sidx/swt)
cmd("CREATE VERTEX TYPE Chunk IF NOT EXISTS")
for prop, typ in [("chunk_id", "STRING"), ("text", "STRING"), ("source_path", "STRING"),
                  ("section_path", "STRING"), ("page_no", "INTEGER"), ("regions_json", "STRING"),
                  ("cite_hex", "STRING"), ("collection", "STRING"), ("metadata_json", "STRING"),
                  ("position", "STRING"), ("primary_keyword", "STRING"),
                  ("english_level", "STRING"), ("experience_years", "INTEGER"),
                  ("embedding", "ARRAY_OF_FLOATS")]:
    cmd(f"CREATE PROPERTY Chunk.{prop} IF NOT EXISTS {typ}")
cmd("CREATE INDEX IF NOT EXISTS ON Chunk (chunk_id) NOTUNIQUE")
cmd("CREATE INDEX IF NOT EXISTS ON Chunk (cite_hex) NOTUNIQUE")
cmd("CREATE INDEX IF NOT EXISTS ON Chunk (collection) NOTUNIQUE")
cmd("CREATE INDEX IF NOT EXISTS ON Chunk (primary_keyword) NOTUNIQUE")
cmd("CREATE INDEX IF NOT EXISTS ON Chunk (english_level) NOTUNIQUE")
cmd("CREATE INDEX IF NOT EXISTS ON Chunk (experience_years) NOTUNIQUE")

client = chromadb.PersistentClient(path="/vdb")


def regions_from_meta(m):
    keys = ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")
    if all(m.get(k) is not None for k in keys) and m.get("page_no") is not None:
        try:
            return json.dumps([{"page_no": int(m["page_no"]),
                                "bbox": [float(m["bbox_x0"]), float(m["bbox_y0"]),
                                         float(m["bbox_x1"]), float(m["bbox_y1"])]}])
        except Exception:
            return ""
    return ""


total = 0
t0 = time.time()
for coll in COLLS:
    col = client.get_collection(coll)
    n = col.count()
    print(f"{coll}: {n} chunks", flush=True)
    off = 0
    while off < n:
        g = col.get(limit=BATCH, offset=off, include=['embeddings', 'documents', 'metadatas'])
        ids, embs, docs_, metas = g['ids'], g['embeddings'], g['documents'], g['metadatas']
        if not ids:
            break
        stmts, params = [], {}
        for k, cid in enumerate(ids):
            m = metas[k] or {}
            pg = m.get("page_no")
            params[f"d{k}"] = {
                "chunk_id": cid, "text": docs_[k] or "",
                "source_path": m.get("source_path", "") or "",
                "section_path": m.get("section_path", "") or "",
                "page_no": int(pg) if isinstance(pg, (int, float)) else -1,
                "regions_json": regions_from_meta(m),
                "cite_hex": hashlib.sha1(cid.encode("utf-8")).hexdigest()[:12],
                "collection": coll,
                "metadata_json": json.dumps(m, ensure_ascii=False),
                "position": m.get("position") or "",
                "primary_keyword": m.get("primary_keyword") or "",
                "english_level": m.get("english_level") or "",
                "experience_years": int(m["experience_years"]) if isinstance(m.get("experience_years"), (int, float)) else -1,
                "embedding": [float(x) for x in embs[k]],
            }
            stmts.append(f"INSERT INTO Chunk CONTENT :d{k};")
        cmd("\n".join(stmts), params=params, language="sqlscript")
        off += len(ids); total += len(ids)
        if off % 5000 < BATCH:
            print(f"  {coll}: {off}/{n}  ({total/(time.time()-t0):.0f} rows/s)", flush=True)

# 3. dense index AFTER load (backfills)
print(f"loaded {total} chunks in {time.time()-t0:.0f}s; creating dense LSM_VECTOR index...", flush=True)
cmd('CREATE INDEX ON Chunk (embedding) LSM_VECTOR METADATA {"dimensions":1024,"similarity":"COSINE"}')
print("VERIFY:", cmd("SELECT count(*) AS n, count(cite_hex) AS hex, count(regions_json) AS reg FROM Chunk"), flush=True)
