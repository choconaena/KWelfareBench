"""Bootstrap 95% CI + paired Wilcoxon for GT-2 baseline metrics.

Adversarial reviewer demands these. Computed on 66 query bootstrap.
"""
from __future__ import annotations
import json, math, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from retrievers.bm25 import BM25Retriever, korean_tokenize
from sentence_transformers import SentenceTransformer

DATA = REPO / "data"
N_BOOT = 10000


def per_query_metrics(rank_dict, queries, gt2, bok, k=10):
    pol_ids_full = [p["policy_id"] for p in bok]
    out = {"ndcg10": [], "p10_t2": [], "r10_t2": []}
    for q in queries:
        ranked = rank_dict[q["qid"]]
        topk = ranked[:k]
        rels = [gt2.get((q["qid"], pid), 0) for pid in topk]
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
        all_rels = sorted([gt2.get((q["qid"], pid), 0) for pid in pol_ids_full], reverse=True)[:k]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(all_rels))
        out["ndcg10"].append(dcg / idcg if idcg > 0 else 0)
        hits = sum(1 for pid in topk if gt2.get((q["qid"], pid), 0) >= 2)
        out["p10_t2"].append(hits / k)
        relevant_t2 = sum(1 for pid in pol_ids_full if gt2.get((q["qid"], pid), 0) >= 2)
        out["r10_t2"].append(hits / relevant_t2 if relevant_t2 > 0 else None)
    return out


def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05):
    arr = np.array([v for v in values if v is not None])
    if len(arr) == 0:
        return {"mean": 0, "ci_lo": 0, "ci_hi": 0, "n": 0}
    boots = np.array([np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
    return {
        "mean": float(np.mean(arr)),
        "ci_lo": float(np.percentile(boots, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        "n": len(arr),
    }


def main():
    print("Loading...", flush=True)
    policies = json.load(open(REPO / "data/policies.json"))
    bok = [p for p in policies if "복지로" in p.get("source", "") or "bokjiro" in p.get("url", "")]
    pool = json.load(open(DATA / "query_pool_v3.json"))
    queries = []
    cats = ['welfare', 'employment', 'education', 'housing', 'health', 'culture', 'living']
    for c in cats:
        for sti, st in enumerate(pool[c]['subtopics']):
            for qi, q in enumerate(st['queries']):
                queries.append({"qid": f"{c}_{sti:02d}_{qi}", "text": q})
    gt2 = {}
    for line in open(REPO / "data/gt2_openai_final.jsonl"):
        d = json.loads(line)
        parts = d["custom_id"].split("__")
        if len(parts) == 2:
            gt2[(parts[0], parts[1])] = d["score"]
    print(f"  policies={len(bok)}, queries={len(queries)}, gt2={len(gt2):,}")
    np.random.seed(42)

    print("\nBuilding rankings...", flush=True)
    bm25 = BM25Retriever(policies=bok)
    bm25_ranks = {q["qid"]: [p for p, _ in sorted(zip(bm25.policy_ids, bm25.bm25.get_scores(korean_tokenize(q["text"]))), key=lambda x: -x[1])] for q in queries}
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    corpus = [" ".join([p.get("name","") or "", p.get("summary","") or "", (p.get("description") or "")[:300]]) for p in bok]
    pol_embs = model.encode(corpus, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    q_embs = model.encode([q["text"] for q in queries], batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    sims = q_embs @ pol_embs.T
    dense_ranks = {q["qid"]: [bok[i]["policy_id"] for i in np.argsort(-sims[qi])] for qi, q in enumerate(queries)}
    K_RRF = 60
    hybrid_ranks = {}
    for q in queries:
        scores = defaultdict(float)
        for rank, pid in enumerate(bm25_ranks[q["qid"]][:200]):
            scores[pid] += 1 / (K_RRF + rank + 1)
        for rank, pid in enumerate(dense_ranks[q["qid"]][:200]):
            scores[pid] += 1 / (K_RRF + rank + 1)
        hybrid_ranks[q["qid"]] = sorted(scores.keys(), key=lambda p: -scores[p])

    print("\nPer-query metrics...", flush=True)
    bm25_m = per_query_metrics(bm25_ranks, queries, gt2, bok)
    dense_m = per_query_metrics(dense_ranks, queries, gt2, bok)
    hybrid_m = per_query_metrics(hybrid_ranks, queries, gt2, bok)

    print(f"\nBootstrap CI (N={N_BOOT}):")
    results = {}
    for name, m in [("BM25", bm25_m), ("Dense", dense_m), ("Hybrid", hybrid_m)]:
        results[name] = {}
        print(f"\n  {name}:")
        for metric in ["ndcg10", "p10_t2", "r10_t2"]:
            ci = bootstrap_ci(m[metric])
            results[name][metric] = ci
            print(f"    {metric}: {ci['mean']:.4f}  [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]  (n={ci['n']})")

    # Paired Wilcoxon
    try:
        from scipy.stats import wilcoxon
        def pw(a, b):
            pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            if len(pairs) < 5: return None
            a_arr, b_arr = zip(*pairs)
            diff = np.array(a_arr) - np.array(b_arr)
            if np.all(diff == 0): return 1.0
            try:
                _, p = wilcoxon(a_arr, b_arr)
                return float(p)
            except Exception:
                return None
        print("\nPaired Wilcoxon p-values:")
        p1 = pw(dense_m["ndcg10"], bm25_m["ndcg10"])
        p2 = pw(hybrid_m["ndcg10"], dense_m["ndcg10"])
        p3 = pw(dense_m["p10_t2"], bm25_m["p10_t2"])
        print(f"  Dense > BM25 (NDCG@10): p={p1}")
        print(f"  Hybrid vs Dense (NDCG@10): p={p2}")
        print(f"  Dense > BM25 (P@10 t=2): p={p3}")
        results["paired_wilcoxon"] = {"Dense_vs_BM25_ndcg10": p1, "Hybrid_vs_Dense_ndcg10": p2, "Dense_vs_BM25_p10_t2": p3}
    except ImportError:
        print("scipy not installed, skip Wilcoxon")

    out = REPO / "experiments/baseline_ci_results.json"
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n✓ saved {out}")


if __name__ == "__main__":
    main()
