#!/usr/bin/env python3
"""Export a frozen snapshot of the live bulario__corpus from Chroma — REUSING
the exact bge-m3 vectors already stored (no re-embed → no GPU, no confound).

Runs INSIDE the noted-rag container (it owns chromadb + the /data/vectordb
persistence). A second read-only PersistentClient is safe: Chroma/SQLite WAL
gives the reader a consistent snapshot while ingestion keeps writing.

Writes to OUT_DIR (inside the container; copied out afterwards):
  embeddings.f32.npy   float32 [N, 1024]        — the reused bge-m3 vectors
  records.jsonl        {id, text, metadata}     — one row per chunk, same order
  manifest.json        {collection, count, dim, source_docs}

Deterministic order (ids sorted) so embeddings row i ⟷ records line i ⟷
downstream ArcadeDB record — the join key across the whole PoC.
"""
import json
import os

import chromadb
import numpy as np

COLL = os.getenv("EXPORT_COLLECTION", "bulario__corpus")
CHROMA = os.getenv("CHROMA_DIR", "/data/vectordb")
OUT_DIR = os.getenv("OUT_DIR", "/tmp/bulario_snapshot")
BATCH = int(os.getenv("EXPORT_BATCH", "2000"))

os.makedirs(OUT_DIR, exist_ok=True)
client = chromadb.PersistentClient(CHROMA)
col = client.get_collection(COLL)
total = col.count()
print(f"collection={COLL} count={total}", flush=True)

# Pull ids first (cheap) and sort for a stable, reproducible ordering.
all_ids = sorted(col.get(include=[]).get("ids") or [])
assert len(all_ids) == total, f"id count {len(all_ids)} != count {total}"

emb_rows: list = []
docs_seen: set = set()
with open(os.path.join(OUT_DIR, "records.jsonl"), "w") as fout:
    for start in range(0, total, BATCH):
        ids = all_ids[start:start + BATCH]
        g = col.get(ids=ids, include=["embeddings", "documents", "metadatas"])
        # Chroma returns in its own order; realign to our sorted `ids`.
        by_id = {g["ids"][i]: i for i in range(len(g["ids"]))}
        for _id in ids:
            j = by_id[_id]
            meta = g["metadatas"][j] or {}
            emb_rows.append(np.asarray(g["embeddings"][j], dtype=np.float32))
            docs_seen.add(meta.get("source_path"))
            fout.write(json.dumps({"id": _id,
                                   "text": g["documents"][j] or "",
                                   "metadata": meta}, ensure_ascii=False) + "\n")
        print(f"  exported {min(start + BATCH, total)}/{total}", flush=True)

emb = np.vstack(emb_rows)
np.save(os.path.join(OUT_DIR, "embeddings.f32.npy"), emb)
manifest = {"collection": COLL, "count": int(emb.shape[0]),
            "dim": int(emb.shape[1]), "source_docs": len([d for d in docs_seen if d])}
with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("DONE", json.dumps(manifest), flush=True)
