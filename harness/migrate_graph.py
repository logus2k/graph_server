#!/usr/bin/env python3
"""Migrate a domain's knowledge graph (Entity + RELATES) from PROD ArcadeDB
(26.5.1, :2480) into the PoC (26.7.2, :2490), SAME database, alongside the
Chunk corpus (true consolidation: graph + vectors in one DB).

WHY keyset (not SKIP/LIMIT): ArcadeDB re-emits rows across bucket boundaries
even in a single scan (a raw `SELECT id ... LIMIT 500000` returned 20000 rows
with only 19607 distinct ids), and SKIP/LIMIT pages drift. Keyset pagination
(`WHERE id > :last ORDER BY id`) + client-side dedup is the only reliable full
traversal. Edges are read via each vertex's outE adjacency — every edge has
exactly one source vertex, so iterating all vertices covers every edge once.

Env: MIGRATE_DB (default bulario_ref).
"""
import os, time, httpx

SRC = ("http://localhost:2480", ("root", "noted-dev"))
DST = ("http://localhost:2490", ("root", os.environ["ARCADEDB_PW"]))
DB = os.getenv("MIGRATE_DB", "bulario_ref")
PAGE = 2000
BATCH = 200
c = httpx.Client(timeout=180)


def run(base_auth, db, cmd, params=None, lang="sql"):
    base, auth = base_auth
    body = {"language": lang, "command": cmd}
    if params is not None:
        body["params"] = params
    r = c.post(f"{base}/api/v1/command/{db}", auth=auth, json=body)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    j = r.json()
    if "error" in j:
        raise RuntimeError(f"{j.get('exception')}: {j.get('detail')}")
    return j.get("result", [])


def query(base_auth, db, cmd, params=None):
    base, auth = base_auth
    body = {"language": "sql", "command": cmd}
    if params is not None:
        body["params"] = params
    r = c.post(f"{base}/api/v1/query/{db}", auth=auth, json=body)
    r.raise_for_status()
    return r.json().get("result", [])


def insert_vertices(batch):
    stmts, params = [], {}
    for k, e in enumerate(batch):
        stmts.append(f"INSERT INTO Entity CONTENT :d{k};")
        params[f"d{k}"] = e
    run(DST, DB, "\n".join(stmts), params=params, lang="sqlscript")


def create_edges(batch):
    stmts, params = [], {}
    for n, (f, t, ty) in enumerate(batch):
        stmts.append(f"CREATE EDGE RELATES FROM (SELECT FROM Entity WHERE id=:f{n}) "
                     f"TO (SELECT FROM Entity WHERE id=:t{n}) SET type=:y{n};")
        params[f"f{n}"] = f; params[f"t{n}"] = t; params[f"y{n}"] = ty
    run(DST, DB, "\n".join(stmts), params=params, lang="sqlscript")


def main():
    total_v = query(SRC, DB, "SELECT count(*) AS n FROM Entity")[0]["n"]
    total_e = query(SRC, DB, "SELECT count(*) AS n FROM RELATES")[0]["n"]
    print(f"source: {total_v} vertices, {total_e} edges", flush=True)

    # reset target graph only (Chunk corpus stays)
    run(DST, DB, "DROP TYPE RELATES IF EXISTS UNSAFE")
    run(DST, DB, "DROP TYPE Entity IF EXISTS UNSAFE")
    run(DST, DB, "CREATE VERTEX TYPE Entity")
    for p in ("id", "type", "label", "properties_json", "tags_json"):
        run(DST, DB, f"CREATE PROPERTY Entity.{p} STRING")
    run(DST, DB, "CREATE PROPERTY Entity.project_ids LIST")
    run(DST, DB, "CREATE INDEX ON Entity (id) UNIQUE")
    run(DST, DB, "CREATE EDGE TYPE RELATES")
    run(DST, DB, "CREATE PROPERTY RELATES.type STRING")

    # vertices — keyset by id + dedup
    print("migrating vertices (keyset)...", flush=True)
    seen, last, buf, t0 = set(), "", [], time.time()
    while True:
        # First page omits WHERE: ArcadeDB NPEs on `id > ''` (empty-string keyset).
        where = "WHERE id > :last" if last else ""
        rows = query(SRC, DB, "SELECT id,type,label,properties_json,tags_json,project_ids "
                     f"FROM Entity {where} ORDER BY id LIMIT {PAGE}",
                     {"last": last} if last else None)
        if not rows:
            break
        last = rows[-1]["id"]
        for r in rows:
            i = r["id"]
            if i in seen:
                continue
            seen.add(i)
            buf.append({"id": i, "type": r.get("type"), "label": r.get("label"),
                        "properties_json": r.get("properties_json") or "{}",
                        "tags_json": r.get("tags_json") or "[]",
                        "project_ids": r.get("project_ids") or []})
            if len(buf) >= BATCH:
                insert_vertices(buf); buf = []
        print(f"  vertices {len(seen)}/{total_v} ({len(seen)/(time.time()-t0):.0f}/s)", flush=True)
    if buf:
        insert_vertices(buf)

    # edges — iterate vertex ids by keyset, pull outE adjacency per batch, dedup
    print("migrating edges (outE per vertex batch)...", flush=True)
    eseen, last, ebuf, done_e, t1 = set(), "", [], 0, time.time()
    while True:
        where = "WHERE id > :last" if last else ""
        idrows = query(SRC, DB, f"SELECT id FROM Entity {where} ORDER BY id LIMIT {PAGE}",
                       {"last": last} if last else None)
        if not idrows:
            break
        last = idrows[-1]["id"]
        ids = list({r["id"] for r in idrows})
        erows = query(SRC, DB, "SELECT outV().id AS f, inV().id AS t, type "
                      "FROM (SELECT expand(outE('RELATES')) FROM Entity WHERE id IN :ids)",
                      {"ids": ids})
        for e in erows:
            f, t, ty = e.get("f"), e.get("t"), e.get("type")
            if not f or not t:
                continue
            key = (f, t, ty)
            if key in eseen:
                continue
            eseen.add(key)
            ebuf.append((f, t, ty))
            if len(ebuf) >= BATCH:
                create_edges(ebuf); done_e += len(ebuf); ebuf = []
        print(f"  edges {done_e}/{total_e} ({done_e/(time.time()-t1):.0f}/s)", flush=True)
    if ebuf:
        create_edges(ebuf); done_e += len(ebuf)

    v = query(DST, DB, "SELECT count(*) AS n FROM Entity")[0]["n"]
    e = query(DST, DB, "SELECT count(*) AS n FROM RELATES")[0]["n"]
    print(f"DONE: target Entity={v}/{total_v}  RELATES={e}/{total_e} "
          f"(dedup: {len(seen)} uniq vertices, {len(eseen)} uniq edges)")


if __name__ == "__main__":
    main()
