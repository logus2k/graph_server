"""job2cool graph migration — per-domain graphs from noted-arcadedb -> graph_server
`job2cool` db, tagged by `domain`. Entities (all types incl markdown_chunk) + RELATES
edges (domain-scoped). Entity embeddings read DIRECTLY from Chroma <domain>__gr_entities
(noted-rag is down). Run in the noted-rag image (chromadb) on noted-network:

  docker run --rm --network noted-network --entrypoint python \
    -v /home/logus/env/assets/noted/data/vectordb:/vdb:ro \
    -v /home/logus/env/assets/graph_server/harness:/mig \
    services-noted-rag /mig/job2cool_graph_migrate.py <dom1>,<dom2>,...
"""
import os
import sys, time
import httpx, chromadb

GS = "http://graph-server-arcadedb:2480"; GA = ("root", os.environ["ARCADEDB_PW"])
NA = "http://noted-arcadedb:2480"; NAA = ("root", "noted-dev")
DB = "job2cool"
DOMAINS = sys.argv[1].split(",")
c = httpx.Client(timeout=600)
client = chromadb.PersistentClient(path="/vdb")


def gcmd(command, params=None, language="sql"):
    body = {"language": language, "command": command}
    if params:
        body["params"] = params
    r = c.post(f"{GS}/api/v1/command/{DB}", auth=GA, json=body)
    r.raise_for_status()
    return r.json().get("result")


def nquery(db, command):
    r = c.post(f"{NA}/api/v1/query/{db}", auth=NAA,
               json={"language": "sql", "command": command})
    r.raise_for_status()
    return r.json().get("result") or []


# schema (Entity tagged by domain; (domain,id) unique so same id can exist per domain)
gcmd("CREATE VERTEX TYPE Entity IF NOT EXISTS")
for prop, typ in [("id", "STRING"), ("label", "STRING"), ("type", "STRING"),
                  ("domain", "STRING"), ("properties_json", "STRING"),
                  ("embedding", "ARRAY_OF_FLOATS")]:
    gcmd(f"CREATE PROPERTY Entity.{prop} IF NOT EXISTS {typ}")
gcmd("CREATE INDEX IF NOT EXISTS ON Entity (domain, id) UNIQUE")
gcmd("CREATE INDEX IF NOT EXISTS ON Entity (type) NOTUNIQUE")
gcmd("CREATE INDEX IF NOT EXISTS ON Entity (domain) NOTUNIQUE")
gcmd("CREATE EDGE TYPE RELATES IF NOT EXISTS")
gcmd("CREATE PROPERTY RELATES.type IF NOT EXISTS STRING")
gcmd("CREATE PROPERTY RELATES.domain IF NOT EXISTS STRING")
gcmd("CREATE INDEX IF NOT EXISTS ON RELATES (type) NOTUNIQUE")

tot_e = tot_x = 0
t0 = time.time()
for dom in DOMAINS:
    try:
        ents = nquery(dom, "SELECT id, label, type, properties_json FROM Entity")
        edges = nquery(dom, "SELECT outV().id AS src, inV().id AS dst, type FROM RELATES")
    except Exception as e:
        print(f"{dom}: SKIP ({str(e)[:60]})", flush=True)
        continue
    emb_by_id = {}
    try:
        col = client.get_collection(f"{dom}__gr_entities")
        g = col.get(include=['embeddings'], limit=col.count())
        for i, cid in enumerate(g['ids']):
            emb_by_id[cid] = g['embeddings'][i]
    except Exception:
        pass
    print(f"{dom}: {len(ents)} ents, {len(edges)} edges, {len(emb_by_id)} emb", flush=True)
    docs = []
    for e in ents:
        d = {"id": e["id"], "label": e.get("label"), "type": e.get("type"), "domain": dom,
             "properties_json": e.get("properties_json") or ""}
        if e["id"] in emb_by_id:
            d["embedding"] = [float(x) for x in emb_by_id[e["id"]]]
        docs.append(d)
    for i in range(0, len(docs), 100):
        grp = docs[i:i + 100]; stmts = []; params = {}
        for k, d in enumerate(grp):
            stmts.append(f"INSERT INTO Entity CONTENT :e{k};"); params[f"e{k}"] = d
        gcmd("\n".join(stmts), params=params, language="sqlscript")
    for i in range(0, len(edges), 50):
        grp = edges[i:i + 50]; stmts = []; params = {}
        for k, e in enumerate(grp):
            if not e.get("src") or not e.get("dst"):
                continue
            stmts.append(
                f"CREATE EDGE RELATES FROM (SELECT FROM Entity WHERE domain=:d{k} AND id=:s{k}) "
                f"TO (SELECT FROM Entity WHERE domain=:d{k} AND id=:t{k}) SET type=:y{k}, domain=:d{k};")
            params[f"d{k}"] = dom; params[f"s{k}"] = e["src"]
            params[f"t{k}"] = e["dst"]; params[f"y{k}"] = e["type"]
        if stmts:
            gcmd("\n".join(stmts), params=params, language="sqlscript")
    tot_e += len(docs); tot_x += len(edges)

print(f"inserted {tot_e} entities, {tot_x} edges in {time.time()-t0:.0f}s; "
      "ensuring Entity vector index (dense LSM_VECTOR backfills)...", flush=True)
try:
    gcmd('CREATE INDEX ON Entity (embedding) LSM_VECTOR METADATA {"dimensions":1024,"similarity":"COSINE"}')
except Exception as e:
    print("  index already exists (backfills automatically):", str(e)[:70])
print("VERIFY:", gcmd("SELECT count(*) AS n, count(embedding) AS emb FROM Entity"), flush=True)
print("by domain:", gcmd("SELECT domain, count(*) AS n FROM Entity GROUP BY domain"), flush=True)
