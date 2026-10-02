#!/usr/bin/env python3
"""Hybrid bench: dense vs sparse vs RRF-fused (native vector.fuse), each then
reranked, on the gold cases. Measures answer-quality (recall@k / MRR) AND
retrieval latency (dense vs sparse vs fuse). Query sparse vectors come from the
same bge-m3 (CPU) so both sides are consistent.

Run AFTER gen_sparse.py (full) + load_arcadedb.py --reset (with sparse.jsonl).
"""
import os
import json, os, sys, time, statistics as st, httpx
from FlagEmbedding import BGEM3FlagModel
HERE=os.path.dirname(os.path.abspath(__file__)); SNAP=os.path.join(HERE,"snapshot")
sys.path.insert(0,"/home/logus/env/labs/medical/tests"); import expected_answers as ea
A=("root",os.environ["ARCADEDB_PW"]); NR="http://localhost:8201"; RR="http://localhost:8500/v1/rerank"
MODEL=os.getenv("BGE_M3_PATH","/home/logus/env/assets/noted/data/models/bge-m3")
c=httpx.Client(timeout=60); K=20; TOPK=10
def cmd(command, params):
    r=c.post("http://localhost:2490/api/v1/command/poc",auth=A,
             json={"language":"sql","command":command,"params":params}); r.raise_for_status()
    j=r.json()
    if "error" in j: raise RuntimeError(j.get("detail"))
    return j["result"]
def embed(q): return c.post(f"{NR}/embed",json={"texts":[q]}).json()["vectors"][0]
def rerank(q,docs):
    res=c.post(RR,json={"model":"bge-reranker","query":q,"documents":docs}).json()["results"]
    return sorted(range(len(res)),key=lambda i:-float(res[i]["relevance_score"]))
DENSE="SELECT source_path, text FROM (SELECT expand(vector.neighbors('Chunk[embedding]', :dv, :k)))"
SPARSE="SELECT source_path, text FROM (SELECT expand(vector.sparseNeighbors('Chunk[sidx,swt]', :qi, :qw, :k)))"
FUSE=("SELECT source_path, text FROM (SELECT expand(vector.fuse("
      "vector.neighbors('Chunk[embedding]', :dv, :k),"
      "vector.sparseNeighbors('Chunk[sidx,swt]', :qi, :qw, :k),"
      "{'fusion':'RRF'})))")
def rows_to(r): return [(x.get("source_path",""),x.get("text","")) for x in r]

def main():
    print("loading bge-m3 (CPU) for query sparse encoding...",flush=True)
    model=BGEM3FlagModel(MODEL,use_fp16=False,devices="cpu")
    paths=set(json.loads(l)["metadata"].get("source_path","").lower() for l in open(f"{SNAP}/records.jsonl"))
    cases=[x for x in ea.ALL_CASES if x.expected_source_substr and any(x.expected_source_substr.lower() in p for p in paths)]
    print(f"gold cases: {len(cases)}")
    legs={"dense":DENSE,"sparse":SPARSE,"hybrid_fuse":FUSE}
    agg={k:{"hit":0,"rr":0.0,"lat":[]} for k in legs}
    aggR={k:{"hit":0,"rr":0.0} for k in legs}
    for x in cases:
        sub=x.expected_source_substr.lower()
        dv=embed(x.question)
        se=model.encode([x.question],return_dense=False,return_sparse=True,return_colbert_vecs=False)["lexical_weights"][0]
        items=sorted((int(k),float(v)) for k,v in se.items())
        qi=[k for k,_ in items]; qw=[v for _,v in items]
        params={"dv":dv,"qi":qi,"qw":qw,"k":K}
        for name,q in legs.items():
            t=time.perf_counter(); cand=rows_to(cmd(q,params)); lat=(time.perf_counter()-t)*1000
            agg[name]["lat"].append(lat)
            paths_ranked=[p for p,_ in cand]
            rk=next((i+1 for i,p in enumerate(paths_ranked) if sub in (p or "").lower()),0)
            agg[name]["hit"]+=1 if rk else 0; agg[name]["rr"]+=(1.0/rk) if rk else 0.0
            # + rerank
            if cand:
                order=rerank(x.question,[t for _,t in cand]); rr_paths=[cand[i][0] for i in order][:TOPK]
                rrk=next((i+1 for i,p in enumerate(rr_paths) if sub in (p or "").lower()),0)
                aggR[name]["hit"]+=1 if rrk else 0; aggR[name]["rr"]+=(1.0/rrk) if rrk else 0.0
    n=len(cases)
    def pct(h): return round(100*h/n,1)
    out={"gold_cases":n,"k_candidates":K,"topk":TOPK}
    for name in legs:
        out[name]={"raw_recall_pct":pct(agg[name]["hit"]),"raw_mrr":round(agg[name]["rr"]/n,3),
                   "plus_rerank_recall_pct":pct(aggR[name]["hit"]),"plus_rerank_mrr":round(aggR[name]["rr"]/n,3),
                   "retrieve_lat_p50_ms":round(st.median(agg[name]["lat"]),1)}
    print(json.dumps(out,indent=2))
    json.dump(out,open(f"{HERE}/../results/hybrid_1000docs.json","w"),indent=2)

if __name__=="__main__": main()
