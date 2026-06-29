"""GT-3 baseline 평가 — (persona, query) → GT-1 ∩ GT-2 target set.

B1/B2/B3 (rule-free): query만 사용, target = GT-3 set
B5 Rule: eligible policies (random/all 순서, no query) — N/A query 없음
B7 Hybrid: GT-1 filter + Dense rerank by query
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
import numpy as np
from sentence_transformers import SentenceTransformer

DATA = REPO / "data"


def load_data():
    policies = json.load(open(REPO / "data/policies.json"))
    bok = [p for p in policies if "복지로" in p.get("source", "") or "bokjiro" in p.get("url", "")]
    pool = json.load(open(DATA / "query_pool_v3.json"))
    queries = []
    cats = ['welfare', 'employment', 'education', 'housing', 'health', 'culture', 'living']
    for c in cats:
        for sti, st in enumerate(pool[c]['subtopics']):
            for qi, q in enumerate(st['queries']):
                queries.append({"qid": f"{c}_{sti:02d}_{qi}", "text": q})
    gt3_strict = json.load(open(DATA / "gt3_region_strict.json"))
    gt3_lenient = json.load(open(DATA / "gt3_region_lenient.json"))
    gt1 = json.load(open(DATA / "ground_truth_v4.json"))
    return bok, queries, gt3_strict, gt3_lenient, gt1


def metrics_for_baseline(rank_dict, queries, gt3, ks=(10,)):
    """rank_dict[qid] → list of policy_ids ordered by score desc.
    For each (persona, qid), compute P@K, R@K, NDCG@K against gt3[persona][qid].
    """
    out = defaultdict(list)
    for pid, qmap in gt3.items():
        for qid, target_list in qmap.items():
            if not target_list:
                continue
            target = set(target_list)
            ranked = rank_dict.get(qid, [])
            for k in ks:
                topk = ranked[:k]
                hits = sum(1 for p in topk if p in target)
                p_at_k = hits / k
                r_at_k = hits / len(target)
                # NDCG with binary rel
                dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(topk) if p in target)
                idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(target))))
                ndcg = dcg / idcg if idcg > 0 else 0
                out[f"p@{k}"].append(p_at_k)
                out[f"r@{k}"].append(r_at_k)
                out[f"ndcg@{k}"].append(ndcg)
    return {k: sum(v) / len(v) if v else 0 for k, v in out.items()}


def hybrid_b7_rank(rank_dict_dense, queries, gt1, persona_id, qid):
    """Rule (GT-1) prefilter then Dense rerank."""
    eligible = set(gt1.get(persona_id, []))
    dense_ranked = rank_dict_dense.get(qid, [])
    # Filter to eligible only, preserve order
    return [p for p in dense_ranked if p in eligible]


def metrics_for_b7(rank_dict_dense, gt3, gt1, ks=(10,)):
    out = defaultdict(list)
    for pid, qmap in gt3.items():
        for qid, target_list in qmap.items():
            if not target_list:
                continue
            target = set(target_list)
            ranked = hybrid_b7_rank(rank_dict_dense, None, gt1, pid, qid)
            for k in ks:
                topk = ranked[:k]
                hits = sum(1 for p in topk if p in target)
                p_at_k = hits / k
                r_at_k = hits / len(target)
                dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(topk) if p in target)
                idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(target))))
                ndcg = dcg / idcg if idcg > 0 else 0
                out[f"p@{k}"].append(p_at_k)
                out[f"r@{k}"].append(r_at_k)
                out[f"ndcg@{k}"].append(ndcg)
    return {k: sum(v) / len(v) if v else 0 for k, v in out.items()}


def main():
    print("Loading...", flush=True)
    bok, queries, gt3_strict, gt3_lenient, gt1 = load_data()
    print(f"  policies={len(bok)}, queries={len(queries)}, gt3_strict cells={sum(len(q) for q in gt3_strict.values()):,}")

    # BM25
    print("\nB1 BM25 ranking...", flush=True)
    t0 = time.time()
    bm25 = BM25Retriever(policies=bok)
    bm25_ranks = {}
    for q in queries:
        scores = bm25.bm25.get_scores(korean_tokenize(q["text"]))
        ranked = sorted(zip(bm25.policy_ids, scores), key=lambda x: -x[1])
        bm25_ranks[q["qid"]] = [p for p, _ in ranked[:200]]  # top-200 cache
    print(f"  done {time.time()-t0:.1f}s", flush=True)

    # Dense
    print("\nB2 Dense ko-SRoBERTa...", flush=True)
    t0 = time.time()
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    corpus = [" ".join([p.get("name","") or "", p.get("summary","") or "", (p.get("description") or "")[:300]]) for p in bok]
    pol_embs = model.encode(corpus, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    q_embs = model.encode([q["text"] for q in queries], batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    sims = q_embs @ pol_embs.T
    dense_ranks = {}
    for qi, q in enumerate(queries):
        idxs = np.argsort(-sims[qi])
        dense_ranks[q["qid"]] = [bok[i]["policy_id"] for i in idxs[:200]]
    print(f"  done {time.time()-t0:.1f}s", flush=True)

    # Hybrid RRF
    K_RRF = 60
    hybrid_ranks = {}
    for q in queries:
        scores = defaultdict(float)
        for rank, pid in enumerate(bm25_ranks[q["qid"]]):
            scores[pid] += 1 / (K_RRF + rank + 1)
        for rank, pid in enumerate(dense_ranks[q["qid"]]):
            scores[pid] += 1 / (K_RRF + rank + 1)
        ranked = sorted(scores.keys(), key=lambda p: -scores[p])
        hybrid_ranks[q["qid"]] = ranked

    # Evaluate
    results = {}
    for setting_name, gt3 in [("strict", gt3_strict), ("lenient", gt3_lenient)]:
        results[f"gt3_{setting_name}"] = {}
        print(f"\n=== GT-3 {setting_name} ===", flush=True)
        for bname, ranks in [("BM25", bm25_ranks), ("Dense", dense_ranks), ("Hybrid", hybrid_ranks)]:
            m = metrics_for_baseline(ranks, queries, gt3, ks=(10,))
            results[f"gt3_{setting_name}"][bname] = m
            print(f"  {bname}: P@10={m['p@10']:.4f}  R@10={m['r@10']:.4f}  NDCG@10={m['ndcg@10']:.4f}", flush=True)
        # B7 (rule + dense)
        m = metrics_for_b7(dense_ranks, gt3, gt1, ks=(10,))
        results[f"gt3_{setting_name}"]["B7_RuleDense"] = m
        print(f"  B7 (Rule+Dense): P@10={m['p@10']:.4f}  R@10={m['r@10']:.4f}  NDCG@10={m['ndcg@10']:.4f}", flush=True)

    out = REPO / "experiments/gt3_baselines_results.json"
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n✓ saved {out}")


if __name__ == "__main__":
    main()
