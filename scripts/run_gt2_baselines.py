"""GT-2 baseline 평가 — 66 queries × 4937 policies × {BM25, Dense, Hybrid}.

Adversarial reviewer [Critical-2] 대응: Tab.2 GT-2 cells TBD 제거.

NDCG@K and Recall@K computed against GT-2 graded scores (0/1/2 OpenAI).
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from retrievers.bm25 import BM25Retriever, korean_tokenize


def load_data():
    policies = json.load(open(REPO / "data/policies.json"))
    bok = [p for p in policies if "복지로" in p.get("source", "") or "bokjiro" in p.get("url", "")]
    pol_ids = [p["policy_id"] for p in bok]

    pool = json.load(open(REPO / "data/query_pool_v3.json"))
    queries = []
    cats = ['welfare', 'employment', 'education', 'housing', 'health', 'culture', 'living']
    for c in cats:
        for sti, st in enumerate(pool[c]['subtopics']):
            for qi, q in enumerate(st['queries']):
                queries.append({
                    "qid": f"{c}_{sti:02d}_{qi}",
                    "text": q,
                    "category": c,
                    "subtopic": st['subtopic'],
                })

    # GT-2 scores
    gt2 = {}
    for line in open(REPO / "data/gt2_openai_final.jsonl"):
        d = json.loads(line)
        cid = d["custom_id"]
        parts = cid.split("__")
        if len(parts) == 2:
            gt2[(parts[0], parts[1])] = d["score"]
    return bok, pol_ids, queries, gt2


def ndcg_at_k(ranked_pids, qid, gt2, k):
    """DCG with graded relevance / IDCG."""
    rels = [gt2.get((qid, pid), 0) for pid in ranked_pids[:k]]
    dcg = sum((r) / math.log2(i + 2) for i, r in enumerate(rels))
    # ideal: sort all relevances for this query, take top-k
    all_rels = sorted([gt2.get((qid, pid), 0) for pid in pol_ids_global], reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(all_rels[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_pids, qid, gt2, k, threshold=1):
    """Fraction of relevant (score >= threshold) policies in top-K out of all relevant."""
    relevant = {pid for pid in pol_ids_global if gt2.get((qid, pid), 0) >= threshold}
    if not relevant:
        return None  # skip queries with no relevant
    hits = sum(1 for pid in ranked_pids[:k] if pid in relevant)
    return hits / len(relevant)


def precision_at_k(ranked_pids, qid, gt2, k, threshold=1):
    hits = sum(1 for pid in ranked_pids[:k] if gt2.get((qid, pid), 0) >= threshold)
    return hits / k


def evaluate(retriever_fn, name, queries, gt2, ks=(5, 10, 20)):
    """retriever_fn(query_text) -> ranked policy_ids list."""
    print(f"\n=== {name} ===", flush=True)
    metrics = defaultdict(list)
    t0 = time.time()
    for i, q in enumerate(queries):
        ranked = retriever_fn(q["text"])
        for k in ks:
            ndcg = ndcg_at_k(ranked, q["qid"], gt2, k)
            metrics[f"ndcg@{k}"].append(ndcg)
            r1 = recall_at_k(ranked, q["qid"], gt2, k, threshold=1)
            if r1 is not None:
                metrics[f"recall@{k}_t1"].append(r1)
            r2 = recall_at_k(ranked, q["qid"], gt2, k, threshold=2)
            if r2 is not None:
                metrics[f"recall@{k}_t2"].append(r2)
            metrics[f"prec@{k}_t1"].append(precision_at_k(ranked, q["qid"], gt2, k, threshold=1))
            metrics[f"prec@{k}_t2"].append(precision_at_k(ranked, q["qid"], gt2, k, threshold=2))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(queries)} done", flush=True)
    elapsed = time.time() - t0
    avg = {k: sum(v) / len(v) if v else 0 for k, v in metrics.items()}
    print(f"  elapsed {elapsed:.1f}s")
    print(f"  results:")
    for k in sorted(avg.keys()):
        print(f"    {k}: {avg[k]:.4f}")
    return avg


def main():
    global pol_ids_global
    print("Loading data...", flush=True)
    bok, pol_ids, queries, gt2 = load_data()
    pol_ids_global = pol_ids
    print(f"  policies: {len(bok)}, queries: {len(queries)}, GT-2 cells: {len(gt2):,}")

    results = {}

    # B1: BM25
    print("\nBuilding BM25 index...", flush=True)
    bm25 = BM25Retriever(policies=bok)
    def bm25_rank(query_text):
        tokens = korean_tokenize(query_text)
        scores = bm25.bm25.get_scores(tokens)
        # rank policy_ids by score desc
        ranked = sorted(zip(bm25.policy_ids, scores), key=lambda x: -x[1])
        return [pid for pid, _ in ranked]
    results["BM25"] = evaluate(bm25_rank, "B1 BM25", queries, gt2)

    # B2: Dense (ko-SRoBERTa)
    try:
        print("\nBuilding Dense (ko-SRoBERTa) embeddings...", flush=True)
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        # 정책 corpus
        corpus_texts = []
        for p in bok:
            text = " ".join([
                p.get("name", "") or "",
                p.get("summary", "") or "",
                (p.get("description") or "")[:300],
            ])
            corpus_texts.append(text)
        print(f"  embedding {len(corpus_texts)} policies...", flush=True)
        pol_embs = model.encode(corpus_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        print(f"  embedding {len(queries)} queries...", flush=True)
        q_embs = model.encode([q["text"] for q in queries], batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        # cosine = dot product since normalized
        sims = q_embs @ pol_embs.T  # (Q, P)

        def dense_rank_factory():
            ranked_per_q = {}
            for qi, q in enumerate(queries):
                idxs = np.argsort(-sims[qi])
                ranked_per_q[q["qid"]] = [bok[i]["policy_id"] for i in idxs]
            return ranked_per_q

        ranked_dict = dense_rank_factory()
        # eval
        def dense_rank(qtext):
            qid_for_text = next((q["qid"] for q in queries if q["text"] == qtext), None)
            return ranked_dict.get(qid_for_text, [])
        results["Dense"] = evaluate(dense_rank, "B2 Dense ko-SRoBERTa", queries, gt2)
        # save BM25 / Dense rankings for hybrid
        bm25_ranks = {}
        for q in queries:
            ranked = bm25_rank(q["text"])
            bm25_ranks[q["qid"]] = ranked

        # B3: Hybrid RRF
        print("\nB3: Hybrid (BM25 + Dense, RRF)", flush=True)
        K_RRF = 60
        def hybrid_rank(qtext):
            qid = next((q["qid"] for q in queries if q["text"] == qtext), None)
            bm25_list = bm25_ranks[qid]
            dense_list = ranked_dict[qid]
            scores = defaultdict(float)
            for rank, pid in enumerate(bm25_list):
                scores[pid] += 1 / (K_RRF + rank + 1)
            for rank, pid in enumerate(dense_list):
                scores[pid] += 1 / (K_RRF + rank + 1)
            ranked = sorted(scores.keys(), key=lambda p: -scores[p])
            return ranked
        results["Hybrid"] = evaluate(hybrid_rank, "B3 Hybrid RRF", queries, gt2)

    except Exception as e:
        print(f"Dense/Hybrid failed: {e}", flush=True)
        import traceback; traceback.print_exc()

    # save
    out = REPO / "experiments/gt2_baselines_results.json"
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n✓ saved {out}")
    print("\n=== SUMMARY ===")
    for name, m in results.items():
        print(f"\n{name}:")
        for k in ["ndcg@5", "ndcg@10", "ndcg@20", "recall@5_t1", "recall@10_t1", "recall@10_t2"]:
            if k in m:
                print(f"  {k}: {m[k]:.4f}")


if __name__ == "__main__":
    main()
