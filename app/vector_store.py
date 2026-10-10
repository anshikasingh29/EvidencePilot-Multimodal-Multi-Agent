import hashlib
import os
from pathlib import Path
from typing import Iterable
import chromadb
from openai import OpenAI
from .config import get_settings

class VectorStore:
    """Persistent Chroma store with explicit OpenAI embeddings."""
    def __init__(self):
        self.settings = get_settings()
        Path(self.settings.chroma_path).mkdir(parents=True, exist_ok=True)
        self.db = chromadb.PersistentClient(path=self.settings.chroma_path)
        self.collection = self.db.get_or_create_collection(
            name="evidencepilot_chunks",
            metadata={"hnsw:space": "cosine", "description": "Document chunks for EvidencePilot RAG"},
        )
        self.client = OpenAI(api_key=self._api_key())

    def _api_key(self) -> str:
        key = self.settings.openai_api_key
        if not key or key == "replace_me":
            raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and add your key.")
        return key

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.settings.openai_embedding_model, input=texts)
        return [item.embedding for item in response.data]

    def add_chunks(self, source: str, chunks: list[str]) -> int:
        if not chunks:
            return 0
        # Stable IDs make re-ingestion idempotent for unchanged source/chunk pairs.
        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            digest = hashlib.sha1(f"{source}:{i}:{chunk}".encode()).hexdigest()[:18]
            ids.append(digest); docs.append(chunk)
            metas.append({"source": source, "chunk_index": i})
        embeddings = self.embed(docs)
        # Chroma add ignores duplicates only if absent; upsert refreshes changed embeddings.
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        return len(docs)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        result = self.collection.query(query_embeddings=self.embed([query]), n_results=top_k,
                                       include=["documents", "metadatas", "distances"])
        rows = []
        for i, doc in enumerate((result.get("documents") or [[]])[0]):
            meta = (result.get("metadatas") or [[]])[0][i]
            distance = (result.get("distances") or [[]])[0][i]
            rows.append({"source": meta.get("source", "unknown"), "chunk_id": f"chunk-{meta.get('chunk_index', i)}",
                         "excerpt": doc, "score": round(1 - float(distance), 4)})
        return rows

    def count(self) -> int:
        return self.collection.count()
