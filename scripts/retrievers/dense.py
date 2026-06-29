"""B2: Dense-only retrieval (bge-m3 + ChromaDB cosine, NO metadata filter)."""
from __future__ import annotations
from pathlib import Path
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from .base import Retriever, PolicyResult


CHROMA_DB_PATH = str(Path(__file__).resolve().parents[2] / "data/vectors/chroma_db")
COLLECTION_NAME = "demo_welfare_policies"


class DenseRetriever(Retriever):
    name = "Dense"

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.model = SentenceTransformer("BAAI/bge-m3")

    def retrieve(self, query: str, persona: dict, k: int = 10) -> List[PolicyResult]:
        emb = self.model.encode([query])[0].tolist()
        res = self.collection.query(
            query_embeddings=[emb],
            n_results=k,
            include=["distances"],
        )
        ids = res["ids"][0]
        dists = res["distances"][0] if res.get("distances") else [0.0] * len(ids)
        out = []
        for rank, (pid, d) in enumerate(zip(ids, dists), 1):
            out.append(PolicyResult(policy_id=pid, score=float(1.0 - d), rank=rank))
        return out
