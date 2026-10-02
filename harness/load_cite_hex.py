"""Add cite_hex (sha1(chunk_id)[:12] — the value used in the [markdown_chunk:<hex>]
citation tag) to ArcadeDB Chunk + index, so citations resolve persistently
(not just from the app's in-memory cache)."""
import os, json, time, hashlib, httpx
HERE=os.path.dirname(os.path.abspath(__file__)); DB=os.environ["ARCADE_DB"]; SNAP=os.path.join(HERE, os.environ["SNAP_SUB"])
URL="http://localhost:2490"; A=("root",os.environ["ARCADEDB_PW"]); c=httpx.Client(timeout=180)
def cmd(command,params=None,lang="sql"):
    b={"language":lang,"command":command}
    if params: b["params"]=params
    r=c.post(f"{URL}/api/v1/command/{DB}",auth=A,json=b); r.raise_for_status()
    j=r.json()
    if "error" in j: raise RuntimeError(j.get("detail"))
    return j.get("result",[])
def cite_hex(cid):
    if cid.startswith("markdown_chunk:"): return cid.split(":",1)[1]
    return hashlib.sha1(cid.encode("utf-8")).hexdigest()[:12]
cmd("CREATE PROPERTY Chunk.cite_hex IF NOT EXISTS STRING")
ids=[json.loads(l)["id"] for l in open(f"{SNAP}/records.jsonl")]
t0=time.time(); n=0; B=200
for s in range(0,len(ids),B):
    stmts,params=[],{}
    for k in range(s,min(s+B,len(ids))):
        stmts.append(f"UPDATE Chunk SET cite_hex=:h{k-s} WHERE chunk_id=:i{k-s};")
        params[f"h{k-s}"]=cite_hex(ids[k]); params[f"i{k-s}"]=ids[k]
    cmd("\n".join(stmts),params=params,lang="sqlscript"); n+=len(stmts)
    if n%20000==0 or n>=len(ids): print(f"  {n}/{len(ids)} ({n/(time.time()-t0):.0f}/s)",flush=True)
cmd("CREATE INDEX IF NOT EXISTS ON Chunk (cite_hex) NOTUNIQUE")
print("DONE with-hex:", cmd("SELECT count(*) AS n FROM Chunk WHERE cite_hex IS NOT NULL")[0]["n"])
