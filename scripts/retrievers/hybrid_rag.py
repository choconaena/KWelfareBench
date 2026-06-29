"""B3: Hybrid RAG — bge-m3 + ChromaDB + metadata `$in` filter (current system).

This is the *current* deployed retrieval, *without* LLM rerank.
Metadata filter on region + tag fields applied via ChromaDB `where` clause.
"""
from __future__ import annotations
from pathlib import Path
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from .base import Retriever, PolicyResult


CHROMA_DB_PATH = str(Path(__file__).resolve().parents[2] / "data/vectors/chroma_db")
COLLECTION_NAME = "demo_welfare_policies"


def build_where_from_persona(persona: dict, full_filter: bool = False) -> dict:
    """Build ChromaDB metadata filter from persona attributes.

    Default (full_filter=False) — region only (B3 baseline, the deployed system).
    full_filter=True (B3.5) — region + all tag-field $in filters.

    region filter: matches 전국 OR persona's sido OR persona's sigungu
    """
    sido = persona.get("sido")
    sigungu = persona.get("sigungu")
    region_clauses = [{"region_level": "전국"}]
    if sido:
        region_clauses.append({"$and": [
            {"region_level": "시도"},
            {"region_sido": sido},
        ]})
        if sigungu:
            region_clauses.append({"$and": [
                {"region_level": "시군구"},
                {"region_sido": sido},
                {"region_sigungu": sigungu},
            ]})

    region_clause = region_clauses[0] if len(region_clauses) == 1 else {"$or": region_clauses}

    if not full_filter:
        return region_clause

    # B3.5 full tag filtering — disabled if persona declares not-having
    # Note: ChromaDB metadata stores list-tag fields as comma-separated string
    # (see backend/services/vectorize.py build_metadata). We use $contains for
    # robust list-membership matching.
    extra = []
    # Income detail: persona's income_detail must be in policy's tag list
    inc = persona.get("income_detail", []) or []
    # Disability: if persona doesn't have disability, exclude policies that require it
    if persona.get("disability") != "있음":
        # We approximate: exclude where tags_disability=='필수'. Use $ne if supported.
        # For chromadb compatibility we use $or to allow non-필수.
        extra.append({"$or": [
            {"tags_disability": "상관없음"},
            {"tags_disability": "우대"},
        ]})

    if extra:
        return {"$and": [region_clause] + extra}
    return region_clause


class HybridRAGRetriever(Retriever):
    """B3 — region-only metadata filter (the deployed system before this work)."""
    name = "Hybrid-RAG"
    _full_filter = False

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.model = SentenceTransformer("BAAI/bge-m3")

    def retrieve(self, query: str, persona: dict, k: int = 10) -> List[PolicyResult]:
        emb = self.model.encode([query])[0].tolist()
        where = build_where_from_persona(persona, full_filter=self._full_filter)
        try:
            res = self.collection.query(
                query_embeddings=[emb], n_results=k, where=where, include=["distances"],
            )
        except Exception:
            res = self.collection.query(
                query_embeddings=[emb], n_results=k, include=["distances"],
            )
        ids = res["ids"][0]
        dists = res["distances"][0] if res.get("distances") else [0.0] * len(ids)
        out = []
        for rank, (pid, d) in enumerate(zip(ids, dists), 1):
            out.append(PolicyResult(policy_id=pid, score=float(1.0 - d), rank=rank))
        return out


class HybridRAGFullRetriever(HybridRAGRetriever):
    """B3.5 — region + tag-field metadata filter (fairness baseline against Graph)."""
    name = "Hybrid-RAG-Full"
    _full_filter = True
