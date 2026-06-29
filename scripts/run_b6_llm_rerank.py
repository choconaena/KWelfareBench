"""B6 LLM-rerank using Gemini (independent from OpenAI judge).

Top-50 from BM25 → Gemini scores each → reranked → top-10.
Cost: 66 queries × 50 = 3,300 calls × ~$0.0001/call = ~$0.33

Used as independent judge (different from OpenAI which labeled GT-2).
"""
from __future__ import annotations
import json, math, os, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from dotenv import load_dotenv
from google import genai
from google.genai import types
import numpy as np
from sentence_transformers import SentenceTransformer
from retrievers.bm25 import BM25Retriever, korean_tokenize

load_dotenv(REPO / ".env")
DATA = REPO / "data"
OUT_DIR = REPO / "experiments/b6_rerank"
OUT_DIR.mkdir(exist_ok=True, parents=True)

MODEL = "gemini-2.5-flash-lite"
TOP_N_CANDIDATES = 50  # rerank top-50 from BM25
N_THREADS = 20

PROMPT = """다음은 한국 복지정책 검색 query와 정책입니다.
정책이 query 주제와 얼마나 관련 있는지 0/1/2로 평가:
- 0 (무관): query 주제와 정책 주제 거의 무관
- 1 (인접): 같은 큰 카테고리/부분 관련
- 2 (직접): query가 직접 가리키는 주제

자격(eligibility) X. 주제·관심사만 평가."""


def score_pair(client, query, pol_name, pol_summary):
    body = (f"{PROMPT}\n\nquery: \"{query}\"\n정책: \"{pol_name}\"\n"
            f"요약: \"{(pol_summary or '')[:200]}\"\n\n등급 (0/1/2):")
    for attempt in range(3):
        try:
            r = client.models.generate_content(
                model=MODEL,
                contents=body,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=4),
            )
            text = (r.text or "").strip()
            for ch in text:
                if ch in "012":
                    return int(ch)
            return None
        except Exception as e:
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
            else:
                return None
    return None


def main():
    print("Loading...", flush=True)
    policies = json.load(open(REPO / "data/policies.json"))
    bok = [p for p in policies if "복지로" in p.get("source", "") or "bokjiro" in p.get("url", "")]
    pol_dict = {p["policy_id"]: p for p in bok}
    pool = json.load(open(DATA / "query_pool_v3.json"))
    queries = []
    cats = ['welfare', 'employment', 'education', 'housing', 'health', 'culture', 'living']
    for c in cats:
        for sti, st in enumerate(pool[c]['subtopics']):
            for qi, q in enumerate(st['queries']):
                queries.append({"qid": f"{c}_{sti:02d}_{qi}", "text": q})

    # GT-2 OpenAI scores (for evaluation, not generation)
    gt2 = {}
    for line in open(REPO / "data/gt2_openai_final.jsonl"):
        d = json.loads(line)
        parts = d["custom_id"].split("__")
        if len(parts) == 2:
            gt2[(parts[0], parts[1])] = d["score"]

    # BM25 candidates (top-50 per query)
    print("BM25 top-50 per query...", flush=True)
    bm25 = BM25Retriever(policies=bok)
    candidates = {}
    for q in queries:
        scores = bm25.bm25.get_scores(korean_tokenize(q["text"]))
        ranked = sorted(zip(bm25.policy_ids, scores), key=lambda x: -x[1])[:TOP_N_CANDIDATES]
        candidates[q["qid"]] = [p for p, _ in ranked]

    n_calls = sum(len(c) for c in candidates.values())
    print(f"Total LLM calls: {n_calls}")

    # Resume support
    out_path = OUT_DIR / "rerank_scores.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                d = json.loads(line)
                done.add((d["qid"], d["pid"]))
            except: pass
    print(f"Resume: {len(done)} done")

    # Test 10 calls first
    if len(done) < 10:
        print("\n[Test phase] 10 calls validation...", flush=True)
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        with open(out_path, "a") as f:
            count = 0
            for qid, pids in candidates.items():
                if count >= 10:
                    break
                for pid in pids[:10 - count]:
                    if (qid, pid) in done:
                        continue
                    pol = pol_dict[pid]
                    qtext = next(q["text"] for q in queries if q["qid"] == qid)
                    s = score_pair(client, qtext, pol["name"], pol.get("summary"))
                    f.write(json.dumps({"qid": qid, "pid": pid, "score": s}, ensure_ascii=False) + "\n")
                    f.flush()
                    done.add((qid, pid))
                    count += 1
                    if count >= 10: break
                if count >= 10: break
        print(f"  10건 test OK, 진행")

    # Full rerank
    pairs_todo = []
    for qid, pids in candidates.items():
        for pid in pids:
            if (qid, pid) not in done:
                pairs_todo.append((qid, pid))
    print(f"\n[Full rerank] {len(pairs_todo)} pairs × {N_THREADS} threads")

    out_f = open(out_path, "a", encoding="utf-8")
    import threading
    lock = threading.Lock()
    t0 = time.time()
    n_done = [0]
    n_fail = [0]

    def task(qid, pid):
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        pol = pol_dict[pid]
        qtext = next(q["text"] for q in queries if q["qid"] == qid)
        s = score_pair(client, qtext, pol["name"], pol.get("summary"))
        return qid, pid, s

    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futures = [ex.submit(task, qid, pid) for qid, pid in pairs_todo]
        for fut in as_completed(futures):
            qid, pid, s = fut.result()
            with lock:
                out_f.write(json.dumps({"qid": qid, "pid": pid, "score": s}, ensure_ascii=False) + "\n")
                n_done[0] += 1
                if s is None:
                    n_fail[0] += 1
                if n_done[0] % 200 == 0:
                    out_f.flush()
                    rate = n_done[0] / (time.time() - t0)
                    eta = (len(pairs_todo) - n_done[0]) / rate if rate > 0 else 0
                    print(f"  [{n_done[0]}/{len(pairs_todo)}] rate={rate:.1f}/s ETA={eta/60:.1f}min fail={n_fail[0]}", flush=True)
    out_f.close()
    print(f"\n  done {time.time()-t0:.1f}s, fail={n_fail[0]}")

    # Load all rerank scores
    rerank_scores = {}
    for line in open(out_path):
        try:
            d = json.loads(line)
            if d.get("score") is not None:
                rerank_scores[(d["qid"], d["pid"])] = d["score"]
        except: pass

    # Rerank: for each query, sort candidates by Gemini rerank score (desc), tie-break by original BM25 rank
    print("\nEvaluating B6 rerank on GT-2...", flush=True)
    b6_ranks = {}
    for q in queries:
        cands = candidates[q["qid"]]
        scored = [(p, rerank_scores.get((q["qid"], p), 0)) for p in cands]
        scored.sort(key=lambda x: -x[1])
        b6_ranks[q["qid"]] = [p for p, _ in scored]

    # Compute metrics
    pol_ids_full = [p["policy_id"] for p in bok]
    metrics = {"ndcg10": [], "p10_t2": [], "r10_t2": []}
    for q in queries:
        topk = b6_ranks[q["qid"]][:10]
        rels = [gt2.get((q["qid"], pid), 0) for pid in topk]
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
        all_rels = sorted([gt2.get((q["qid"], pid), 0) for pid in pol_ids_full], reverse=True)[:10]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(all_rels))
        metrics["ndcg10"].append(dcg / idcg if idcg > 0 else 0)
        hits_t2 = sum(1 for pid in topk if gt2.get((q["qid"], pid), 0) >= 2)
        metrics["p10_t2"].append(hits_t2 / 10)
        relevant_t2 = sum(1 for pid in pol_ids_full if gt2.get((q["qid"], pid), 0) >= 2)
        if relevant_t2 > 0:
            metrics["r10_t2"].append(hits_t2 / relevant_t2)
        else:
            metrics["r10_t2"].append(None)

    avg = {k: sum(v for v in vs if v is not None) / max(1, len([v for v in vs if v is not None])) for k, vs in metrics.items()}
    print(f"\nB6 LLM rerank (Gemini) GT-2:")
    print(f"  NDCG@10 = {avg['ndcg10']:.4f}")
    print(f"  P@10 (t≥2) = {avg['p10_t2']:.4f}")
    print(f"  R@10 (t≥2) = {avg['r10_t2']:.4f}")

    # Save
    json.dump({
        "model": MODEL,
        "n_calls": len(rerank_scores),
        "metrics_gt2": avg,
    }, open(OUT_DIR / "summary.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
