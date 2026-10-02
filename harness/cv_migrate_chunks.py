"""CV migration Phase A1 — load cv__corpus (69 chunks) into graph_server `cv` db
as Chunk vertices with dense (noted-rag) + sparse (embeddings_server) + regions
+ cite_hex. Mirrors load_arcadedb.py (sparse index BEFORE load, dense AFTER).
Run INSIDE a container on noted-network (reaches all services by hostname).
"""
import os
import httpx, hashlib, time
GS = "http://graph-server-arcadedb:2480"; GA = ("root", os.environ["ARCADEDB_PW"])
NR = "http://noted-rag:8200"; EMB = "http://embeddings-server:8600"
c = httpx.Client(timeout=300)


def cmd(db, command, params=None, language="sql"):
    body = {"language": language, "command": command}
    if params:
        body["params"] = params
    r = c.post(f"{GS}/api/v1/command/{db}", auth=GA, json=body)
    r.raise_for_status()
    return r.json().get("result")


# 1. create db (tolerate exists)
r = c.post(f"{GS}/api/v1/server", auth=GA, json={"command": "create database cv"})
print("create db:", r.status_code, r.text[:140])

# 2. pull chunks + compute dense + sparse
recs = c.post(f"{NR}/list_records", json={"collection": "cv__corpus", "limit": 500}).json()["records"]
texts = [x["text"] for x in recs]
print(f"pulled {len(recs)} chunks")
dense = c.post(f"{NR}/embed", json={"texts": texts}).json()["vectors"]
sparse = c.post(f"{EMB}/embed", json={"texts": texts, "dense": False, "sparse": True}).json()["sparse"]
print(f"dense={len(dense)} sparse={len(sparse)} dim={len(dense[0])}")

# 3. schema + indexes (sparse BEFORE load)
cmd("cv", "CREATE VERTEX TYPE Chunk IF NOT EXISTS")
for prop, typ in [("chunk_id", "STRING"), ("text", "STRING"), ("source_path", "STRING"),
                  ("section_path", "STRING"), ("page_no", "INTEGER"), ("regions_json", "STRING"),
                  ("cite_hex", "STRING"), ("embedding", "ARRAY_OF_FLOATS"),
                  ("sidx", "ARRAY_OF_INTEGERS"), ("swt", "ARRAY_OF_FLOATS")]:
    cmd("cv", f"CREATE PROPERTY Chunk.{prop} IF NOT EXISTS {typ}")
cmd("cv", "CREATE INDEX IF NOT EXISTS ON Chunk (chunk_id) UNIQUE")
cmd("cv", "CREATE INDEX IF NOT EXISTS ON Chunk (cite_hex) NOTUNIQUE")
cmd("cv", 'CREATE INDEX ON Chunk (sidx, swt) LSM_SPARSE_VECTOR METADATA {"dimensions":250002,"modifier":"IDF"}')

# 4. insert (per-row CONTENT :dK bound params in one sqlscript)
docs = []
for i, rec in enumerate(recs):
    m = rec.get("metadata") or {}
    pg = m.get("page_no")
    docs.append({
        "chunk_id": rec["id"], "text": rec["text"],
        "source_path": m.get("source_path", ""), "section_path": m.get("section_path", ""),
        "page_no": int(pg) if isinstance(pg, (int, float)) else -1,
        "regions_json": m.get("regions_json", "") or "",
        "cite_hex": hashlib.sha1(rec["id"].encode("utf-8")).hexdigest()[:12],
        "embedding": [float(x) for x in dense[i]],
        "sidx": sparse[i]["indices"], "swt": sparse[i]["weights"],
    })
stmts, params = [], {}
for k, doc in enumerate(docs):
    stmts.append(f"INSERT INTO Chunk CONTENT :d{k};")
    params[f"d{k}"] = doc
t0 = time.time()
cmd("cv", "\n".join(stmts), params=params, language="sqlscript")
print(f"inserted {len(docs)} in {time.time()-t0:.1f}s")

# 5. dense index AFTER load (backfills)
cmd("cv", 'CREATE INDEX ON Chunk (embedding) LSM_VECTOR METADATA {"dimensions":1024,"similarity":"COSINE"}')

# verify
print("VERIFY:", cmd("cv", "SELECT count(*) AS n, count(sidx) AS with_sparse, count(cite_hex) AS with_hex FROM Chunk"))
