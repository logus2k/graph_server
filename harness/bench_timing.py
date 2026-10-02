"""Clean end-to-end latency comparison (GPU idle). Breaks each retrieval path
into components + total, over the real gold queries. Warmup pass first."""
import os
import json, os, sys, time, statistics as st, httpx
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,"/home/logus/env/labs/medical/tests"); import expected_answers as ea
A=("root",os.environ["ARCADEDB_PW"]); NR="http://localhost:8201"; RERANK="http://localhost:8500/v1/rerank"
c=httpx.Client(timeout=60)
def t(fn):
    s=time.perf_counter(); r=fn(); return (time.perf_counter()-s)*1000, r
def embed(q): return c.post(f"{NR}/embed",json={"texts":[q]}).json()["vectors"][0]
def chroma_search(q): return c.post(f"{NR}/search",json={"query":q,"collection":"bulario__corpus","top_k":10,"rerank_min_score":-1e9}).json()
def arc(qv,k):
    r=c.post("http://localhost:2490/api/v1/command/poc",auth=A,json={"language":"sql",
      "command":"SELECT source_path, text FROM (SELECT expand(vector.neighbors('Chunk[embedding]', :q, :k)))","params":{"q":qv,"k":k}})
    return [(x.get("source_path",""),x.get("text","")) for x in r.json()["result"]]
def rerank(q,docs): return c.post(RERANK,json={"model":"bge-reranker","query":q,"documents":docs}).json()
queries=[x.question for x in ea.ALL_CASES][:18]
# warmup
qv0=embed(queries[0]); _=chroma_search(queries[0]); cand0=arc(qv0,20); _=rerank(queries[0],[d for _,d in cand0])
comp={k:[] for k in ["embed","chroma_search_total","arc_neighbors10","arc_neighbors20","rerank20","arc_e2e","chroma_e2e"]}
for q in queries:
    e_ms,qv=t(lambda:embed(q)); comp["embed"].append(e_ms)
    cs_ms,_=t(lambda:chroma_search(q)); comp["chroma_search_total"].append(cs_ms); comp["chroma_e2e"].append(cs_ms)
    a10_ms,_=t(lambda:arc(qv,10)); comp["arc_neighbors10"].append(a10_ms)
    a20_ms,cand=t(lambda:arc(qv,20)); comp["arc_neighbors20"].append(a20_ms)
    rr_ms,_=t(lambda:rerank(q,[d for _,d in cand])); comp["rerank20"].append(rr_ms)
    comp["arc_e2e"].append(e_ms+a20_ms+rr_ms)  # embed + neighbors(20) + rerank(20)
def p(x,q): return round(st.quantiles(x,n=100)[q-1],1) if len(x)>=2 else round(x[0],1)
print(json.dumps({k:{"p50":p(v,50),"p95":p(v,95),"mean":round(st.mean(v),1)} for k,v in comp.items()},indent=2))
