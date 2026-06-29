"""B1: BM25 sparse retrieval baseline."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from .base import Retriever, PolicyResult

REPO = Path(__file__).resolve().parents[3]
POLICIES = REPO / "data/policies.json"


def korean_tokenize(text: str) -> list:
    """Lightweight Korean tokenizer — split on whitespace and punctuation,
    then split Korean text into 2-gram chars to handle agglutination roughly."""
    if not text:
        return []
    # whitespace and punctuation tokens
    coarse = re.split(r"[\s,.\(\)\[\]\{\}\?\!:;'\"·~/\\\-]+", text)
    coarse = [t for t in coarse if t]
    # add char-2gram for Korean tokens to improve recall
    fine = []
    for tok in coarse:
        fine.append(tok)
        if re.search(r"[가-힣]", tok) and len(tok) >= 2:
            for i in range(len(tok) - 1):
                fine.append(tok[i:i + 2])
    return fine


class BM25Retriever(Retriever):
    name = "BM25"

    def __init__(self, policies: list = None):
        if policies is None:
            with open(POLICIES) as f:
                policies = json.load(f)
        self.policies = policies
        self.policy_ids = [p["policy_id"] for p in policies]

        corpus = []
        for p in policies:
            text = " ".join([
                p.get("name", "") or "",
                p.get("summary", "") or "",
                p.get("description", "") or "",
                p.get("eligibility", "") or "",
                p.get("benefits", "") or "",
            ])
            corpus.append(korean_tokenize(text))
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, persona: dict, k: int = 10) -> List[PolicyResult]:
        tokens = korean_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out = []
        for rank, idx in enumerate(ranked, 1):
            out.append(PolicyResult(
                policy_id=self.policy_ids[idx],
                score=float(scores[idx]),
                rank=rank,
            ))
        return out
