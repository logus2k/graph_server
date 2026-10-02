"""CV migration Phase A2 — graph from noted-arcadedb `cv` → graph_server `cv`.
Entities (incl markdown_chunk/community/community_summary), all RELATES edges
(chunked_into/mentions/member_of/summarizes/similar_to), entity embeddings
re-derived from cv__gr_entities text (bge-m3 deterministic). Run inside a
container on noted-network.
"""
import os
import httpx, time
GS = "http://graph-server-arcadedb:2480"; GA = ("root", os.environ["ARCADEDB_PW"])
NA = "http://noted-arcadedb:2480"; NAA = ("root", "noted-dev")
NR = "http://noted-rag:8200"
c = httpx.Client(timeout=300)


def gcmd(db, command, params=None, language="sql"):
    body = {"language": language, "command": command}
    if params:
        body["params"] = params
    r = c.post(f"{GS}/api/v1/command/{db}", auth=GA, json=body)
    r.raise_for_status()
    return r.json().get("result")


def nquery(command):
    r = c.post(f"{NA}/api/v1/query/cv", auth=NAA, json={"language": "sql", "command": command})
    r.raise_for_status()
    return r.json().get("result")


# 1. source graph
ents = nquery("SELECT id, label, type, project_ids, community_id, properties_json FROM Entity")
edges = nquery("SELECT outV().id AS src, inV().id AS dst, type FROM RELATES")
print(f"source: {len(ents)} entities, {len(edges)} edges")

# 2. entity embeddings from cv__gr_entities text (re-embed = same bge-m3 vectors)
gr = c.post(f"{NR}/list_records", json={"collection": "cv__gr_entities", "limit": 500}).json()["records"]
gr_emb = c.post(f"{NR}/embed", json={"texts": [g.get("text") or "" for g in gr]}).json()["vectors"]
emb_by_id = {g["id"]: gr_emb[i] for i, g in enumerate(gr)}
print(f"entity embeddings: {len(emb_by_id)}")

# 3. schema
gcmd("cv", "CREATE VERTEX TYPE Entity IF NOT EXISTS")
for prop, typ in [("id", "STRING"), ("label", "STRING"), ("type", "STRING"),
                  ("properties_json", "STRING"), ("project_ids", "LIST"),
                  ("community_id", "INTEGER"), ("embedding", "ARRAY_OF_FLOATS")]:
    gcmd("cv", f"CREATE PROPERTY Entity.{prop} IF NOT EXISTS {typ}")
gcmd("cv", "CREATE INDEX IF NOT EXISTS ON Entity (id) UNIQUE")
gcmd("cv", "CREATE INDEX IF NOT EXISTS ON Entity (type) NOTUNIQUE")
gcmd("cv", "CREATE EDGE TYPE RELATES IF NOT EXISTS")
gcmd("cv", "CREATE PROPERTY RELATES.type IF NOT EXISTS STRING")
gcmd("cv", "CREATE INDEX IF NOT EXISTS ON RELATES (type) NOTUNIQUE")

# 4. insert entities (embedding only where available)
docs = []
for e in ents:
    d = {"id": e["id"], "label": e.get("label"), "type": e.get("type"),
         "properties_json": e.get("properties_json") or "", "project_ids": e.get("project_ids") or []}
    if e.get("community_id") is not None:
        d["community_id"] = e["community_id"]
    if e["id"] in emb_by_id:
        d["embedding"] = [float(x) for x in emb_by_id[e["id"]]]
    docs.append(d)
for i in range(0, len(docs), 100):
    grp = docs[i:i + 100]; stmts = []; params = {}
    for k, d in enumerate(grp):
        stmts.append(f"INSERT INTO Entity CONTENT :e{k};"); params[f"e{k}"] = d
    gcmd("cv", "\n".join(stmts), params=params, language="sqlscript")
print(f"entities inserted: {len(docs)}")

# 5. edges via id lookup
made = 0
for i in range(0, len(edges), 60):
    grp = edges[i:i + 60]; stmts = []; params = {}
    for k, e in enumerate(grp):
        if not e.get("src") or not e.get("dst"):
            continue
        stmts.append(f"CREATE EDGE RELATES FROM (SELECT FROM Entity WHERE id=:s{k}) "
                     f"TO (SELECT FROM Entity WHERE id=:t{k}) SET type=:y{k};")
        params[f"s{k}"] = e["src"]; params[f"t{k}"] = e["dst"]; params[f"y{k}"] = e["type"]
    if stmts:
        gcmd("cv", "\n".join(stmts), params=params, language="sqlscript"); made += len(stmts)
print(f"edges created: {made}")

# 6. entity vector index (after insert)
gcmd("cv", 'CREATE INDEX ON Entity (embedding) LSM_VECTOR METADATA {"dimensions":1024,"similarity":"COSINE"}')

print("VERIFY entities:", gcmd("cv", "SELECT count(*) AS n, count(embedding) AS emb FROM Entity"))
print("VERIFY types:", gcmd("cv", "SELECT type, count(*) AS n FROM Entity GROUP BY type"))
print("VERIFY edges:", gcmd("cv", "SELECT type, count(*) AS n FROM RELATES GROUP BY type"))
