#!/usr/bin/env python3
"""Content-level integrity check of the consolidated new instance (ArcadeDB
26.7.2, :2490): both vector (dense/sparse/hybrid) and graph, both domains,
comparing ACTUAL returned content against the source of truth (Chroma for
vectors, production ArcadeDB for graph)."""
import os
import json, httpx
from FlagEmbedding import BGEM3FlagModel

POC, PA = "http://localhost:2490", ("root", os.environ["ARCADEDB_PW"])
NR = "http://localhost:8201"
PROD, PRA = "http://localhost:2480", ("root", "noted-dev")
c = httpx.Client(timeout=60)


def q(base, auth, db, cmd, params=None):
    body = {"language": "sql", "command": cmd}
    if params:
        body["params"] = params
    r = c.post(f"{base}/api/v1/query/{db}", auth=auth, json=body)
    r.raise_for_status()
    return r.json().get("result", [])


def embed(t):
    return c.post(f"{NR}/embed", json={"texts": [t]}).json()["vectors"][0]


print("loading bge-m3 for query sparse...", flush=True)
model = BGEM3FlagModel("/home/logus/env/assets/noted/data/models/bge-m3", use_fp16=True, devices="cuda")


def sparse(t):
    lw = model.encode([t], return_dense=False, return_sparse=True, return_colbert_vecs=False)["lexical_weights"][0]
    it = sorted((int(k), float(v)) for k, v in lw.items())
    return [k for k, _ in it], [v for _, v in it]


def snip(s, n=90):
    return (s or "").replace("\n", " ")[:n]


def dense(db, vec, k=3):
    return q(POC, PA, db, "SELECT source_path, text FROM (SELECT expand(vector.neighbors('Chunk[embedding]', :v, :k)))", {"v": vec, "k": k})


def sparse_q(db, qi, qw, k=3):
    return q(POC, PA, db, "SELECT source_path, text FROM (SELECT expand(vector.sparseNeighbors('Chunk[sidx,swt]', :qi, :qw, :k)))", {"qi": qi, "qw": qw, "k": k})


def hybrid(db, vec, qi, qw, k=3):
    return q(POC, PA, db, "SELECT source_path, text FROM (SELECT expand(vector.fuse("
             "vector.neighbors('Chunk[embedding]', :v, :k),"
             "vector.sparseNeighbors('Chunk[sidx,swt]', :qi, :qw, :k),{'fusion':'RRF'})))",
             {"v": vec, "qi": qi, "qw": qw, "k": k})


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------- BULARIO: VECTOR ----------------
section("BULARIO — VECTOR (dense / sparse / hybrid), real query")
query = "Qual a dose e indicação da amoxicilina?"
vec = embed(query); qi, qw = sparse(query)
for name, rows in [("DENSE", dense("bulario", vec)), ("SPARSE", sparse_q("bulario", qi, qw)), ("HYBRID", hybrid("bulario", vec, qi, qw))]:
    print(f"\n[{name}] '{query}'")
    for r in rows:
        print(f"   {r.get('source_path','')[:45]:45} | {snip(r.get('text'))}")

section("BULARIO — known-item self-return (embeddings intact?)")
_cands = q(POC, PA, "bulario", "SELECT chunk_id, text FROM Chunk WHERE source_path LIKE '%AMOXICILINA%' LIMIT 30")
row = next((r for r in _cands if len(r.get("text") or "") > 200), _cands[0])
selfvec = embed(row["text"])
top = dense("bulario", selfvec, 1)
print(f"  query = own text of {row['chunk_id'][:60]}")
print(f"  top-1 source: {top[0]['source_path'][:50]}  (text matches: {snip(top[0]['text'])==snip(row['text'])})")

section("BULARIO — VECTOR fidelity vs Chroma (same chunk_id, exact text?)")
cid = row["chunk_id"]
cro = c.post(f"{NR}/list_records", json={"collection": "bulario__corpus", "ids": [cid], "include_text": True}).json().get("records", [])
new_text = q(POC, PA, "bulario", "SELECT text FROM Chunk WHERE chunk_id = :id", {"id": cid})[0]["text"]
if cro:
    print(f"  chunk_id: {cid[:60]}")
    print(f"  Chroma len={len(cro[0].get('text',''))}  New len={len(new_text)}  EXACT MATCH: {cro[0].get('text')==new_text}")
else:
    print("  (chunk not found in Chroma by id)")

# ---------------- BULARIO: GRAPH ----------------
section("BULARIO — GRAPH: doc -> chunks traversal + content")
doc = q(POC, PA, "bulario", "SELECT id, label, out('RELATES').size() AS deg FROM Entity WHERE type='markdown_doc' AND id LIKE '%AMOXICILINA%' LIMIT 1")
if doc:
    d = doc[0]
    print(f"  doc: {d['label'][:50]} | outgoing edges: {d['deg']}")
    chunks = q(POC, PA, "bulario", "SELECT label, type FROM (SELECT expand(out('RELATES')) FROM Entity WHERE id=:id) LIMIT 3", {"id": d["id"]})
    for ch in chunks:
        print(f"     -> {ch.get('type')}: {ch.get('label','')[:60]}")

section("BULARIO — GRAPH: concept entity content + accents intact")
concept = q(POC, PA, "bulario", "SELECT id, label, type, properties_json FROM Entity WHERE type='concept' AND id LIKE '%ácido%' LIMIT 2")
for e in concept:
    print(f"  {e['type']}: {e['label'][:40]:40} | props: {snip(e.get('properties_json'),80)}")

section("BULARIO — GRAPH vs PROD: same entity's degree matches?")
sample = q(POC, PA, "bulario", "SELECT id, both('RELATES').size() AS deg FROM Entity WHERE type='concept' LIMIT 1")[0]
prod_deg = q(PROD, PRA, "bulario", "SELECT both('RELATES').size() AS deg FROM Entity WHERE id=:id", {"id": sample["id"]})
print(f"  entity {sample['id'][:45]}: NEW deg={sample['deg']}  PROD deg={prod_deg[0]['deg'] if prod_deg else '?'}  MATCH: {prod_deg and prod_deg[0]['deg']==sample['deg']}")

# ---------------- BULARIO_REF (FTN) ----------------
section("BULARIO_REF (FTN) — VECTOR + GRAPH")
fq = "interações e contraindicações"
fvec = embed(fq); fqi, fqw = sparse(fq)
print(f"[HYBRID] '{fq}' on FTN:")
for r in hybrid("bulario_ref", fvec, fqi, fqw):
    print(f"   {r.get('source_path','')[:30]:30} | {snip(r.get('text'))}")
fdoc = q(POC, PA, "bulario_ref", "SELECT label, out('RELATES').size() AS deg FROM Entity WHERE type='markdown_doc' LIMIT 1")[0]
print(f"  FTN doc: {fdoc['label']} -> {fdoc['deg']} edges")
fconc = q(POC, PA, "bulario_ref", "SELECT count(*) AS n FROM Entity WHERE type='concept'")[0]["n"]
print(f"  FTN concept entities: {fconc}")
print("\nDONE")
