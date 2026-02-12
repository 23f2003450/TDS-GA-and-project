# 🔎 Semantic Search with Re-ranking (Fully Local, Free)

A two-stage semantic search system built using Ollama embeddings, FAISS vector search, and Cross-Encoder re-ranking via a FastAPI REST API. This system retrieves the top 5 semantically similar documents and re-ranks them for higher precision.

## 🏗 Architecture



1. **User Query**
2. **Ollama Embedding** (`nomic-embed-text`)
3. **FAISS Vector Search** (Top 5 Retrieval)
4. **Cross-Encoder Re-Ranking**
5. **Top 3 Results** (Final Output)

## 🧰 Tech Stack
* **Python**
* **FastAPI**
* **FAISS (CPU)**
* **sentence-transformers**
* **Ollama**
* **ngrok** (for public endpoint)

## 📁 Project Structure
```text
TDS-GA-and-project/
│
└── GA1/
    └── Q5-semantic-search/
        ├── app.py
        ├── build_index.py
        ├── docs.json
        ├── faiss.index
        ├── requirements.txt
        ├── .gitignore
        └── README.md
```


## ⚙️ Setup Instructions
✅ Step 1 — Create Virtual Environment
```bash
python -m venv venv
# Activate on Windows:
.\venv\Scripts\activate
```


✅ Step 2 — Install Dependencies
```bash
pip install fastapi uvicorn faiss-cpu numpy sentence-transformers requests
```


✅ Step 3 Install the embedding model:
```bash
ollama pull nomic-embed-text
```

✅ Step 4 — Build FAISS Index
Run the script to generate embeddings and save the index:
```bash
python build_index.py
```

✅ Step 6 — Run FastAPI Server
```bash
uvicorn app:app --reload
```

✅ Step 7 — Expose your local API via ngrok
(In other terminal)
```bash
ngrok http 8000
```

Submit URL of your semantic search endpoint. Add /search at the end of URL










