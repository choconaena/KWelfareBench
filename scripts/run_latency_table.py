"""Latency table for §5.6 — wall-clock per-query for each baseline.

Adversarial reviewer [Serious-2]: ML latency advantage asserted but not measured.
"""
from __future__ import annotations

import json
import time
import sys
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from retrievers.bm25 import BM25Retriever, korean_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    print("Loading...", flush=True)
    policies = json.load(open(REPO / "data/policies.json"))
    bok = [p for p in policies if "복지로" in p.get("source", "") or "bokjiro" in p.get("url", "")]
    pool = json.load(open(REPO / "data/query_pool_v3.json"))
    queries = []
    cats = ['welfare', 'employment', 'education', 'housing', 'health', 'culture', 'living']
    for c in cats:
        for sti, st in enumerate(pool[c]['subtopics']):
            for qi, q in enumerate(st['queries']):
                queries.append({"qid": f"{c}_{sti:02d}_{qi}", "text": q})
    print(f"  policies={len(bok)}, queries={len(queries)}")

    # Setup BM25 (one-time index)
    print("\n[setup] BM25...", flush=True)
    t = time.time()
    bm25 = BM25Retriever(policies=bok)
    bm25_setup = time.time() - t
    print(f"  index build: {bm25_setup:.2f}s")

    # Setup Dense (one-time embed)
    print("\n[setup] Dense ko-SRoBERTa...", flush=True)
    t = time.time()
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    corpus = [" ".join([p.get("name","") or "", p.get("summary","") or "", (p.get("description") or "")[:300]]) for p in bok]
    pol_embs = model.encode(corpus, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    dense_setup = time.time() - t
    print(f"  index build: {dense_setup:.2f}s")

    # Per-query latency timing
    print("\n[per-query latency over 66 queries]", flush=True)
    bm25_lat = []
    dense_lat = []
    hybrid_lat = []

    for q in queries:
        # BM25
        t = time.time()
        scores = bm25.bm25.get_scores(korean_tokenize(q["text"]))
        ranked = sorted(zip(bm25.policy_ids, scores), key=lambda x: -x[1])
        bm25_top = [pid for pid, _ in ranked[:10]]
        bm25_lat.append((time.time() - t) * 1000)

        # Dense
        t = time.time()
        q_emb = model.encode([q["text"]], normalize_embeddings=True)
        sim = q_emb @ pol_embs.T
        idxs = np.argsort(-sim[0])[:10]
        dense_top = [bok[i]["policy_id"] for i in idxs]
        dense_lat.append((time.time() - t) * 1000)

        # Hybrid (RRF) — assumes BM25 + Dense already computed
        t = time.time()
        bm25_full = sorted(zip(bm25.policy_ids, bm25.bm25.get_scores(korean_tokenize(q["text"]))), key=lambda x: -x[1])
        bm25_full_ids = [p for p, _ in bm25_full[:200]]
        q_emb2 = model.encode([q["text"]], normalize_embeddings=True)
        sim2 = q_emb2 @ pol_embs.T
        dense_full_idxs = np.argsort(-sim2[0])[:200]
        dense_full_ids = [bok[i]["policy_id"] for i in dense_full_idxs]
        K_RRF = 60
        scores = defaultdict(float)
        for rank, pid in enumerate(bm25_full_ids):
            scores[pid] += 1 / (K_RRF + rank + 1)
        for rank, pid in enumerate(dense_full_ids):
            scores[pid] += 1 / (K_RRF + rank + 1)
        ranked_hybrid = sorted(scores.keys(), key=lambda p: -scores[p])[:10]
        hybrid_lat.append((time.time() - t) * 1000)

    def stats(lat_list):
        return {
            "mean_ms": round(statistics.mean(lat_list), 2),
            "median_ms": round(statistics.median(lat_list), 2),
            "p95_ms": round(sorted(lat_list)[int(0.95 * len(lat_list))], 2),
            "max_ms": round(max(lat_list), 2),
        }

    results = {
        "n_queries": len(queries),
        "n_policies": len(bok),
        "topK": 10,
        "setup_seconds": {"BM25": round(bm25_setup, 2), "Dense": round(dense_setup, 2)},
        "per_query_ms": {
            "BM25": stats(bm25_lat),
            "Dense": stats(dense_lat),
            "Hybrid_RRF": stats(hybrid_lat),
        },
    }
    print("\n=== LATENCY (per-query, ms) ===")
    for name, m in results["per_query_ms"].items():
        print(f"  {name:>12s}: mean={m['mean_ms']:>7.2f}  median={m['median_ms']:>7.2f}  p95={m['p95_ms']:>7.2f}  max={m['max_ms']:>7.2f}")

    out_path = REPO / "experiments/latency_results.json"
    json.dump(results, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
