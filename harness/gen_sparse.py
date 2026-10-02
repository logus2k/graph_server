"""Generate bge-m3 SPARSE (lexical) vectors for the snapshot chunks, on CPU.
Output per chunk: sorted (indices:int[], weights:float[]) — exactly what
ArcadeDB LSM_SPARSE_VECTOR wants. Aligned to records.jsonl order.
"""
import argparse, json, os, time
from FlagEmbedding import BGEM3FlagModel
HERE=os.path.dirname(os.path.abspath(__file__)); SNAP=os.getenv("SNAP_DIR", os.path.join(HERE,"snapshot"))
MODEL=os.getenv("BGE_M3_PATH","/home/logus/env/assets/noted/data/models/bge-m3")
ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=None); ap.add_argument("--batch",type=int,default=64)
ap.add_argument("--device",default=os.getenv("SPARSE_DEVICE","cpu"))
a=ap.parse_args()
texts=[json.loads(l)["text"] for l in open(f"{SNAP}/records.jsonl")]
if a.limit: texts=texts[:a.limit]
use_fp16 = a.device.startswith("cuda")
print(f"loading bge-m3 ({a.device}, fp16={use_fp16}) from {MODEL} ...",flush=True)
model=BGEM3FlagModel(MODEL,use_fp16=use_fp16,devices=a.device)
t0=time.time(); out=open(f"{SNAP}/sparse.jsonl","w"); n=0; nnz=[]
for i in range(0,len(texts),a.batch):
    b=texts[i:i+a.batch]
    r=model.encode(b,batch_size=len(b),max_length=512,return_dense=False,return_sparse=True,return_colbert_vecs=False)
    for lw in r["lexical_weights"]:
        items=sorted((int(k),float(v)) for k,v in lw.items())
        out.write(json.dumps({"indices":[k for k,_ in items],"weights":[round(v,5) for _,v in items]})+"\n")
        nnz.append(len(items))
    n+=len(b)
    if n % 512==0 or n>=len(texts): print(f"  {n}/{len(texts)} ({n/(time.time()-t0):.1f}/s, avg nnz {sum(nnz)/len(nnz):.0f})",flush=True)
out.close(); print(f"DONE {n} chunks in {time.time()-t0:.1f}s -> {SNAP}/sparse.jsonl",flush=True)
