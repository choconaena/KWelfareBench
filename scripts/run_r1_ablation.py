"""R1-violation ablation: counterfactual non-orthogonal query pool.

For each persona-orthogonal sub-topic (e.g., "월세 지원"), prepend a
demographic prefix (예: "청년", "노인", "장애인") to make it
persona-encoding. Generate 1 query per violated sub-topic.

Then label sub-sample (5K cells) with same Gemini judge used for B6.
Compare GT-2 / GT-1 correlation: orthogonal pool → low correlation;
violated pool → high correlation (collapsing factorisation).
"""
from __future__ import annotations
import json, math, os, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(REPO / ".env")
DATA = REPO / "data"
OUT_DIR = REPO / "experiments/r1_ablation"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Counterfactual: persona-attribute prefix per category
DEMO_PREFIX = {
    "welfare": "노인",
    "employment": "청년",
    "education": "다자녀",
    "housing": "청년",
    "health": "장애인",
    "culture": "노인",
    "living": "기초수급",
}

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
                model="gemini-2.5-flash-lite",
                contents=body,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=4),
            )
            text = (r.text or "").strip()
            for ch in text:
                if ch in "012":
                    return int(ch)
            return None
        except Exception:
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    return None


def main():
    print("Loading...", flush=True)
    policies = json.load(open(REPO / "data/policies.json"))
    bok = [p for p in policies if "복지로" in p.get("source", "") or "bokjiro" in p.get("url", "")]
    pol_dict = {p["policy_id"]: p for p in bok}
    pool = json.load(open(DATA / "query_pool_v3.json"))

    # Construct counterfactual queries (R1 violation)
    queries_orig = []
    queries_violated = []
    cats = ['welfare', 'employment', 'education', 'housing', 'health', 'culture', 'living']
    for c in cats:
        prefix = DEMO_PREFIX[c]
        for sti, st in enumerate(pool[c]['subtopics']):
            for qi, q in enumerate(st['queries'][:1]):  # 1 query per sub-topic
                qid = f"{c}_{sti:02d}_{qi}"
                queries_orig.append({"qid": qid, "text": q})
                # Insert prefix into query text
                violated_q = f"{prefix} {q}"
                queries_violated.append({"qid": f"violated_{qid}", "text": violated_q, "orig_qid": qid})

    print(f"Sub-topics: {len(queries_orig)}")
    print(f"Sample original queries: {[q['text'] for q in queries_orig[:3]]}")
    print(f"Sample violated queries: {[q['text'] for q in queries_violated[:3]]}")

    # Sub-sample cells: 33 violated queries × ~150 random policies each = ~5K
    import random
    random.seed(42)
    pol_sample = random.sample([p["policy_id"] for p in bok], 150)
    cells = [(q["qid"], q["text"], pid) for q in queries_violated for pid in pol_sample]
    print(f"\nCells to label: {len(cells)}")

    # Resume
    out_path = OUT_DIR / "violated_scores.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                d = json.loads(line)
                done.add((d["qid"], d["pid"]))
            except: pass
    todo = [(q, qt, p) for q, qt, p in cells if (q, p) not in done]
    print(f"Resume: {len(done)} done, todo: {len(todo)}")

    # Test 10 first
    if len(done) < 10:
        print("\n[Test phase]")
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        with open(out_path, "a") as f:
            for q, qt, p in todo[:10]:
                pol = pol_dict[p]
                s = score_pair(client, qt, pol["name"], pol.get("summary"))
                f.write(json.dumps({"qid": q, "pid": p, "score": s}, ensure_ascii=False) + "\n")
                done.add((q, p))
        print(f"  10 test passed")
        todo = [(q, qt, p) for q, qt, p in cells if (q, p) not in done]

    # Full
    print(f"\n[Full] {len(todo)} pairs × 20 threads")
    out_f = open(out_path, "a", encoding="utf-8")
    lock = threading.Lock()
    t0 = time.time()
    n_done = [0]

    def task(q, qt, p):
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        pol = pol_dict[p]
        s = score_pair(client, qt, pol["name"], pol.get("summary"))
        return q, p, s

    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = [ex.submit(task, q, qt, p) for q, qt, p in todo]
        for fut in as_completed(futures):
            q, p, s = fut.result()
            with lock:
                out_f.write(json.dumps({"qid": q, "pid": p, "score": s}, ensure_ascii=False) + "\n")
                n_done[0] += 1
                if n_done[0] % 500 == 0:
                    out_f.flush()
                    rate = n_done[0] / (time.time() - t0)
                    print(f"  [{n_done[0]}/{len(todo)}] rate={rate:.1f}/s", flush=True)
    out_f.close()
    print(f"  done {time.time()-t0:.1f}s")

    # Load violated scores
    violated_scores = {}
    for line in open(out_path):
        try:
            d = json.loads(line)
            if d.get("score") is not None:
                violated_scores[(d["qid"], d["pid"])] = d["score"]
        except: pass
    print(f"\nViolated scores: {len(violated_scores)}")

    # Compare to original (orthogonal) scores from OpenAI GT-2
    orig_scores = {}
    for line in open(REPO / "data/gt2_openai_final.jsonl"):
        d = json.loads(line)
        parts = d["custom_id"].split("__")
        if len(parts) == 2 and d.get("score") is not None:
            orig_scores[(parts[0], parts[1])] = d["score"]

    # GT-1 (eligibility) for correlation analysis
    gt1 = json.load(open(DATA / "ground_truth_v4.json"))
    # Build per-policy eligibility set across personas
    pol_to_personas = defaultdict(set)
    for pid_per, lst in gt1.items():
        for pol in lst:
            pol_to_personas[pol].add(pid_per)

    # For each (orig_qid, pol), GT-2 score. For (violated_qid, pol), violated score.
    # Compute distribution shift.
    print("\n=== Score distribution comparison ===")
    from collections import Counter
    orig_dist = Counter()
    viol_dist = Counter()
    common_orig_qids = set(q["orig_qid"] for q in queries_violated)
    for (q, p), s in violated_scores.items():
        viol_dist[s] += 1
        orig_qid = q.replace("violated_", "")
        orig_s = orig_scores.get((orig_qid, p))
        if orig_s is not None:
            orig_dist[orig_s] += 1

    print(f"  Original (orthogonal) on same cells:")
    for s in [0, 1, 2]:
        n = orig_dist.get(s, 0)
        total = sum(orig_dist.values())
        print(f"    {s}: {n} ({n/total*100:.1f}%)" if total else f"    {s}: 0")
    print(f"  Violated (R1-violated):")
    for s in [0, 1, 2]:
        n = viol_dist.get(s, 0)
        total = sum(viol_dist.values())
        print(f"    {s}: {n} ({n/total*100:.1f}%)" if total else f"    {s}: 0")

    # Score change: if R1 violated, more high-score (2) due to demographic match?
    print("\n=== Score shifts (orig → violated) ===")
    pairs = 0
    shift_up = 0  # original 0 → violated 1+
    shift_down = 0
    same = 0
    for (q, p), v_s in violated_scores.items():
        orig_qid = q.replace("violated_", "")
        o_s = orig_scores.get((orig_qid, p))
        if o_s is not None:
            pairs += 1
            if v_s > o_s: shift_up += 1
            elif v_s < o_s: shift_down += 1
            else: same += 1
    print(f"  Total pairs: {pairs}")
    print(f"  Same: {same} ({same/pairs*100:.1f}%)" if pairs else "  Same: 0")
    print(f"  Shifted UP (orthogonal lower → violated higher): {shift_up} ({shift_up/pairs*100:.1f}%)" if pairs else "")
    print(f"  Shifted DOWN: {shift_down} ({shift_down/pairs*100:.1f}%)" if pairs else "")

    # GT-1 confound check: violated query encodes persona attribute → GT-2 score correlated with GT-1 eligibility?
    print("\n=== GT-1 ↔ GT-2 correlation check ===")
    # For original orthogonal: no correlation expected
    # For violated: correlation expected
    import numpy as np
    def corr_for_pool(scores_dict, n_violated_pool=False):
        # For each (qid, pol) cell, x=score, y=is policy eligible for any persona w/ matching demo?
        # Simplified: y = avg eligibility ratio across personas
        n_personas = sum(len(lst) for lst in gt1.values())
        x_arr, y_arr = [], []
        for (q, p), s in scores_dict.items():
            n_elig = len(pol_to_personas.get(p, set()))
            x_arr.append(s)
            y_arr.append(n_elig)
        if not x_arr:
            return 0
        return np.corrcoef(x_arr, y_arr)[0, 1]

    # Original on these specific cells
    orig_subset = {(q["orig_qid"], p): orig_scores.get((q["orig_qid"], p))
                   for q in queries_violated for p in pol_sample
                   if orig_scores.get((q["orig_qid"], p)) is not None}
    corr_orig = corr_for_pool(orig_subset)
    corr_viol = corr_for_pool(violated_scores)
    print(f"  Pearson corr (GT-2 score, eligible-persona-count):")
    print(f"    Original (orthogonal):  {corr_orig:.4f}")
    print(f"    Violated (R1 violated): {corr_viol:.4f}")
    print(f"  → Higher correlation in violated = factorization broken (expected)")

    # Save
    summary = {
        "n_violated_queries": len(queries_violated),
        "n_cells_labeled": len(violated_scores),
        "score_distribution": {
            "orig_orthogonal": dict(orig_dist),
            "violated": dict(viol_dist),
        },
        "score_shifts": {"same": same, "up": shift_up, "down": shift_down, "total": pairs},
        "gt1_gt2_correlation": {
            "orthogonal": float(corr_orig),
            "violated": float(corr_viol),
            "delta": float(corr_viol - corr_orig),
        },
    }
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), ensure_ascii=False, indent=2)
    print(f"\n✓ saved {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
