"""Export entity-name embeddings (gr_entities) from Chroma -> id + vector,
for attaching to ArcadeDB Entity vertices (seed vector search)."""
import json, os, chromadb, numpy as np
COLL=os.environ["ENT_COLLECTION"]; OUT=os.environ["OUT_DIR"]
os.makedirs(OUT, exist_ok=True)
col=chromadb.PersistentClient("/data/vectordb").get_collection(COLL)
n=col.count(); print(f"{COLL}: {n} entities", flush=True)
ids_all=sorted(col.get(include=[]).get("ids") or [])
embs=[]; out=open(f"{OUT}/entity_ids.txt","w")
for s in range(0,len(ids_all),2000):
    ids=ids_all[s:s+2000]
    g=col.get(ids=ids, include=["embeddings"])
    by={g["ids"][i]:i for i in range(len(g["ids"]))}
    for _id in ids:
        embs.append(np.asarray(g["embeddings"][by[_id]],dtype=np.float32)); out.write(_id+"\n")
out.close()
arr=np.vstack(embs); np.save(f"{OUT}/entity_emb.f32.npy",arr)
print(f"DONE {arr.shape} dim={arr.shape[1]}", flush=True)
