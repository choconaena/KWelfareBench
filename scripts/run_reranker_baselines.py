"""§4 reranker 비교 실험 — 사용자 #2 fix.
- BGE-M3 multilingual (dense replacement)
- bge-reranker-v2-m3 (cross-encoder rerank)
- ko-reranker (Korean cross-encoder, jhgan/ko-reranker)
- KURE (Korean unified retrieval/embedding) — best Korean SOTA 2025
"""
from __future__ import annotations
import json, math, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from retrievers.bm25 import BM25Retriever, korean_tokenize

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
    gt2 = {}
    for line in open(REPO / "data/gt2_openai_final.jsonl"):
        d = json.loads(line)
        parts = d["custom_id"].split("__")
        if len(parts) == 2 and d.get("score") is not None:
            gt2[(parts[0], parts[1])] = d["score"]
    return bok, queries, gt2


def metrics(rank_dict, queries, gt2, pol_ids, k=10):
    out = {"ndcg10": [], "p10_t2": [], "r10_t2": []}
    for q in queries:
        topk = rank_dict[q["qid"]][:k]
        rels = [gt2.get((q["qid"], pid), 0) for pid in topk]
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
        all_rels = sorted([gt2.get((q["qid"], pid), 0) for pid in pol_ids], reverse=True)[:k]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(all_rels))
        out["ndcg10"].append(dcg / idcg if idcg > 0 else 0)
        hits = sum(1 for pid in topk if gt2.get((q["qid"], pid), 0) >= 2)
        out["p10_t2"].append(hits / k)
        rel_t2 = sum(1 for pid in pol_ids if gt2.get((q["qid"], pid), 0) >= 2)
        out["r10_t2"].append(hits / rel_t2 if rel_t2 > 0 else None)
    return {k: sum(v for v in vs if v is not None) / max(1, sum(1 for v in vs if v is not None))
            for k, vs in out.items()}


def main():
    print("Loading...", flush=True)
    bok, queries, gt2 = load_data()
    pol_ids = [p["policy_id"] for p in bok]
    pol_corpus = [" ".join([p.get("name","") or "", p.get("summary","") or "", (p.get("description") or "")[:300]]) for p in bok]
    print(f"  policies={len(bok)}, queries={len(queries)}, GT-2 cells={len(gt2):,}")

    results = {}

    # BM25 (baseline)
    print("\nBM25...", flush=True)
    bm25 = BM25Retriever(policies=bok)
    bm25_top200 = {}
    for q in queries:
        scores = bm25.bm25.get_scores(korean_tokenize(q["text"]))
        bm25_top200[q["qid"]] = [p for p, _ in sorted(zip(bm25.policy_ids, scores), key=lambda x: -x[1])[:200]]

    # ============ Dense: ko-SRoBERTa (existing baseline) ============
    print("\nko-SRoBERTa...", flush=True)
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    m = SentenceTransformer("jhgan/ko-sroberta-multitask")
    pol_embs = m.encode(pol_corpus, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    q_embs = m.encode([q["text"] for q in queries], batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    sims = q_embs @ pol_embs.T
    sroberta_ranks = {q["qid"]: [bok[i]["policy_id"] for i in np.argsort(-sims[qi])] for qi, q in enumerate(queries)}
    print(f"  done {time.time()-t0:.1f}s")
    results["ko-SRoBERTa"] = metrics(sroberta_ranks, queries, gt2, pol_ids)
    print(f"  NDCG@10={results['ko-SRoBERTa']['ndcg10']:.4f}")

    # ============ Dense: BGE-M3 (multilingual SOTA 2024) ============
    try:
        print("\nBGE-M3 multilingual...", flush=True)
        t0 = time.time()
        m = SentenceTransformer("BAAI/bge-m3")
        pol_embs = m.encode(pol_corpus, batch_size=8, show_progress_bar=False, normalize_embeddings=True)
        q_embs = m.encode([q["text"] for q in queries], batch_size=8, show_progress_bar=False, normalize_embeddings=True)
        sims = q_embs @ pol_embs.T
        bgem3_ranks = {q["qid"]: [bok[i]["policy_id"] for i in np.argsort(-sims[qi])] for qi, q in enumerate(queries)}
        print(f"  done {time.time()-t0:.1f}s")
        results["BGE-M3"] = metrics(bgem3_ranks, queries, gt2, pol_ids)
        print(f"  NDCG@10={results['BGE-M3']['ndcg10']:.4f}")
    except Exception as e:
        print(f"  BGE-M3 fail: {e}")

    # ============ Cross-encoder reranker: bge-reranker-v2-m3 ============
    try:
        print("\nbge-reranker-v2-m3 (rerank BM25 top-200)...", flush=True)
        from sentence_transformers import CrossEncoder
        t0 = time.time()
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
        reranker_ranks = {}
        for q in queries:
            cands = bm25_top200[q["qid"]]
            pairs = [(q["text"], pol_corpus[bok.index(next(p for p in bok if p["policy_id"] == c))][:300]) for c in cands]
            # 더 빠른 lookup
            pid_to_corpus = {p["policy_id"]: pol_corpus[i][:300] for i, p in enumerate(bok)}
            pairs = [(q["text"], pid_to_corpus[c]) for c in cands]
            scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
            ranked = sorted(zip(cands, scores), key=lambda x: -x[1])
            reranker_ranks[q["qid"]] = [p for p, _ in ranked]
        print(f"  done {time.time()-t0:.1f}s")
        results["BGE-reranker-v2-m3"] = metrics(reranker_ranks, queries, gt2, pol_ids)
        print(f"  NDCG@10={results['BGE-reranker-v2-m3']['ndcg10']:.4f}")
    except Exception as e:
        print(f"  reranker fail: {e}")
        import traceback; traceback.print_exc()

    # ============ ko-reranker (Korean specific cross-encoder) ============
    try:
        print("\nko-reranker (rerank BM25 top-200)...", flush=True)
        from sentence_transformers import CrossEncoder
        t0 = time.time()
        reranker = CrossEncoder("Dongjin-kr/ko-reranker", max_length=512)
        ko_reranker_ranks = {}
        pid_to_corpus = {p["policy_id"]: pol_corpus[i][:300] for i, p in enumerate(bok)}
        for q in queries:
            cands = bm25_top200[q["qid"]]
            pairs = [(q["text"], pid_to_corpus[c]) for c in cands]
            scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
            ranked = sorted(zip(cands, scores), key=lambda x: -x[1])
            ko_reranker_ranks[q["qid"]] = [p for p, _ in ranked]
        print(f"  done {time.time()-t0:.1f}s")
        results["ko-reranker"] = metrics(ko_reranker_ranks, queries, gt2, pol_ids)
        print(f"  NDCG@10={results['ko-reranker']['ndcg10']:.4f}")
    except Exception as e:
        print(f"  ko-reranker fail: {e}")

    out = REPO / "experiments/reranker_baselines_results.json"
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n=== Summary ===")
    for name, m in results.items():
        print(f"  {name:<25s}  NDCG@10={m['ndcg10']:.4f}  P@10(t≥2)={m['p10_t2']:.4f}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
