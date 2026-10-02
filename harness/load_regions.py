import os, sys, json, time, httpx
HERE=os.path.dirname(os.path.abspath(__file__))
DB=os.environ["ARCADE_DB"]; SNAP=os.path.join(HERE, os.environ["SNAP_SUB"])
URL="http://localhost:2490"; A=("root",os.environ["ARCADEDB_PW"]); c=httpx.Client(timeout=180)
def cmd(command,params=None,lang="sql"):
    b={"language":lang,"command":command}
    if params: b["params"]=params
    r=c.post(f"{URL}/api/v1/command/{DB}",auth=A,json=b); r.raise_for_status()
    j=r.json()
    if "error" in j: raise RuntimeError(j.get("detail"))
    return j.get("result",[])
cmd("CREATE PROPERTY Chunk.regions_json IF NOT EXISTS STRING")
rows=[]
for line in open(f"{SNAP}/records.jsonl"):
    r=json.loads(line); rj=(r.get("metadata") or {}).get("regions_json")
    if rj and rj!="[]": rows.append((r["id"], rj))
print(f"chunks with regions: {len(rows)}", flush=True)
t0=time.time(); n=0; B=200
for s in range(0,len(rows),B):
    stmts,params={},{}; stmts=[]
    for k in range(s,min(s+B,len(rows))):
        stmts.append(f"UPDATE Chunk SET regions_json=:r{k-s} WHERE chunk_id=:i{k-s};")
        params[f"r{k-s}"]=rows[k][1]; params[f"i{k-s}"]=rows[k][0]
    cmd("\n".join(stmts),params=params,lang="sqlscript"); n+=len(stmts)
    if n%10000==0 or n>=len(rows): print(f"  {n}/{len(rows)} ({n/(time.time()-t0):.0f}/s)",flush=True)
print("DONE", cmd("SELECT count(*) AS n FROM Chunk WHERE regions_json IS NOT NULL")[0]["n"])
