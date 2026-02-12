import json
import requests
import numpy as np
import faiss

# --------- Embedding function ----------
def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return np.array(response.json()["embedding"])

# --------- Load documents ----------
with open("docs.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

embeddings = []

print("Generating embeddings...")

for doc in documents:
    emb = get_embedding(doc["content"])
    embeddings.append(emb)

embeddings = np.array(embeddings).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, "faiss.index")

print("Index built and saved as faiss.index")
