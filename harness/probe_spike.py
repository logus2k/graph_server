"""Per-query latency, 2 passes, to isolate the 3s /search spike (measure, not guess)."""
import os, sys, time, httpx
sys.path.insert(0,"/home/logus/env/labs/medical/tests"); import expected_answers as ea
NR="http://localhost:8201"; RERANK="http://localhost:8500/v1/rerank"; A=("root",os.environ["ARCADEDB_PW"])
c=httpx.Client(timeout=60)
def embed(q): return c.post(f"{NR}/embed",json={"texts":[q]}).json()["vectors"][0]
def search(q): 
    s=time.perf_counter(); c.post(f"{NR}/search",json={"query":q,"collection":"bulario__corpus","top_k":10,"rerank_min_score":-1e9}); return (time.perf_counter()-s)*1000
def arc(qv,k):
    r=c.post("http://localhost:2490/api/v1/command/poc",auth=A,json={"language":"sql","command":"SELECT text FROM (SELECT expand(vector.neighbors('Chunk[embedding]', :q, :k)))","params":{"q":qv,"k":k}}); return [x["text"] for x in r.json()["result"]]
def rr(q,docs):
    s=time.perf_counter(); c.post(RERANK,json={"model":"bge-reranker","query":q,"documents":docs}); return (time.perf_counter()-s)*1000
qs=[x.question for x in ea.ALL_CASES][:18]
embed(qs[0]); search(qs[0])  # warmup
for pass_n in (1,2):
    print(f"--- pass {pass_n}: /search ms  | explicit rerank(20) ms ---")
    for q in qs:
        s_ms=search(q); qv=embed(q); cand=arc(qv,20); r_ms=rr(q,cand)
        flag=" <== SPIKE" if s_ms>800 else ""
        print(f"  search={s_ms:7.1f}  rerank20={r_ms:6.1f}  {q[:40]}{flag}")
