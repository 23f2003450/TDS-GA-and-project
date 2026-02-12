import json
import time
import requests
import numpy as np
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Load docs ----------
with open("docs.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# -------- Load FAISS index ----------
index = faiss.read_index("faiss.index")

# -------- Load Cross Encoder ----------
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -------- Embedding function ----------
def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return np.array(response.json()["embedding"]).astype("float32")

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = True
    rerankK: int = 3

@app.post("/search")
def search(req: SearchRequest):
    start = time.time()

    query_emb = get_embedding(req.query).reshape(1, -1)
    faiss.normalize_L2(query_emb)

    scores, indices = index.search(query_emb, req.k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        candidates.append({
            "id": int(documents[idx]["id"]),
            "score": float(score),
            "content": documents[idx]["content"],
            "metadata": documents[idx]["metadata"]
        })

    reranked = False

    if req.rerank:
        reranked = True

        pairs = [(req.query, c["content"]) for c in candidates]
        rerank_scores = cross_encoder.predict(pairs)

        # Normalize 0-1
        min_s = min(rerank_scores)
        max_s = max(rerank_scores)

        normalized = [
            (s - min_s) / (max_s - min_s + 1e-9)
            for s in rerank_scores
        ]

        for i in range(len(candidates)):
            candidates[i]["score"] = float(normalized[i])

        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:req.rerankK]

    latency = int((time.time() - start) * 1000)

    return {
        "results": candidates,
        "reranked": reranked,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
