"""ArcadeDB dense (retrieve top-N) -> SAME bge-reranker -> top-10. Proves the
reranker is store-agnostic: ArcadeDB+rerank should ~match Chroma+rerank."""
import os
import json, os, sys, httpx
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,"/home/logus/env/labs/medical/tests"); import expected_answers as ea
A=("root",os.environ["ARCADEDB_PW"]); RERANK="http://localhost:8500/v1/rerank"; NR="http://localhost:8201"
c=httpx.Client(timeout=60); CAND=20; K=10
def embed(t): return c.post(f"{NR}/embed",json={"texts":[t]}).json()["vectors"][0]
def arc(qv,k):
    r=c.post("http://localhost:2490/api/v1/command/poc",auth=A,json={"language":"sql",
      "command":"SELECT source_path, text FROM (SELECT expand(vector.neighbors('Chunk[embedding]', :q, :k)))","params":{"q":qv,"k":k}})
    return [(x.get("source_path",""),x.get("text","")) for x in r.json()["result"]]
def rerank(q,docs):
    r=c.post(RERANK,json={"model":"bge-reranker","query":q,"documents":docs}); r.raise_for_status()
    res=r.json()["results"]; return sorted(range(len(res)),key=lambda i:-float(res[i]["relevance_score"]))
paths=set(json.loads(l)["metadata"].get("source_path","").lower() for l in open(f"{HERE}/snapshot/records.jsonl"))
cases=[x for x in ea.ALL_CASES if x.expected_source_substr and any(x.expected_source_substr.lower() in p for p in paths)]
hit=0; rr=0.0; rows=[]
for x in cases:
    sub=x.expected_source_substr.lower(); qv=embed(x.question)
    cand=arc(qv,CAND)                          # top-20 dense candidates from ArcadeDB
    order=rerank(x.question,[t for _,t in cand])   # rerank by text
    ranked=[cand[i][0] for i in order][:K]         # reranked source_paths, top-10
    rk=next((i+1 for i,p in enumerate(ranked) if sub in (p or "").lower()),0)
    hit+=1 if rk else 0; rr+=(1.0/rk) if rk else 0.0
    rows.append((sub,x.category,x.id,rk))
n=len(cases)
print(json.dumps({"gold_cases":n,"k":K,"candidates_before_rerank":CAND,
  "arcade_dense_PLUS_rerank":{"recall_at_k_pct":round(100*hit/n,1),"mrr":round(rr/n,3)}},indent=2))
for s,cat,cid,rk in rows: print(f"  {s:13} {cat} {cid:22} arcade+rerank={rk:>2}")
