"""Attach entity embeddings to ArcadeDB Entity vertices (UPDATE by id) + build
LSM_VECTOR index, so entity seed search runs natively in ArcadeDB."""
import os, sys, time, json, httpx, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
DB=os.environ["ARCADE_DB"]; SRC=os.path.join(HERE, os.environ["ENT_DIR"])
URL="http://localhost:2490"; A=("root",os.environ["ARCADEDB_PW"]); c=httpx.Client(timeout=180)
def cmd(command,params=None,lang="sql"):
    b={"language":lang,"command":command}
    if params: b["params"]=params
    r=c.post(f"{URL}/api/v1/command/{DB}",auth=A,json=b); r.raise_for_status()
    j=r.json()
    if "error" in j: raise RuntimeError(j.get("detail"))
    return j.get("result",[])
ids=[l.strip() for l in open(f"{SRC}/entity_ids.txt")]
emb=np.load(f"{SRC}/entity_emb.f32.npy")
assert len(ids)==emb.shape[0]
cmd("CREATE PROPERTY Entity.embedding IF NOT EXISTS ARRAY_OF_FLOATS")
t0=time.time(); n=0; B=200
for s in range(0,len(ids),B):
    stmts,params=[],{}
    for k in range(s,min(s+B,len(ids))):
        stmts.append(f"UPDATE Entity SET embedding=:e{k-s} WHERE id=:i{k-s};")
        params[f"e{k-s}"]=[float(x) for x in emb[k]]; params[f"i{k-s}"]=ids[k]
    cmd("\n".join(stmts),params=params,lang="sqlscript"); n+=len(stmts)
    if n%4000==0 or n>=len(ids): print(f"  updated {n}/{len(ids)} ({n/(time.time()-t0):.0f}/s)",flush=True)
print("building LSM_VECTOR index on Entity.embedding...",flush=True)
r=cmd('CREATE INDEX ON Entity (embedding) LSM_VECTOR METADATA {"dimensions":1024,"similarity":"COSINE"}')
print("index:",r)
print("entities with embedding:",cmd("SELECT count(*) AS n FROM Entity WHERE embedding IS NOT NULL")[0]["n"])
