"""R1-violation multi-prefix ablation.

3 prefixes per category × 33 sub-topics × 50 random policies = ~5K cells per prefix.
Total: 3 × 5K = 15K cells (이미 5K 있고, 추가 10K = ~$0.3 Gemini)
"""
from __future__ import annotations
import json, math, os, sys, time, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(REPO / ".env")
DATA = REPO / "data"
OUT_DIR = REPO / "experiments/r1_ablation_multi"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# 3 prefixes per category (different demographic dimensions)
PREFIXES = {
    "welfare":    ["노인", "장애인", "한부모"],
    "employment": ["청년", "노인", "장애인"],
    "education":  ["다자녀", "한부모", "다문화"],
    "housing":    ["청년", "신혼부부", "저소득"],
    "health":     ["장애인", "노인", "임산부"],
    "culture":    ["노인", "장애인", "다문화"],
    "living":     ["기초수급", "노인", "장애인"],
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
    import random
    random.seed(42)
    print("Loading...", flush=True)
    policies = json.load(open(REPO / "data/policies.json"))
    bok = [p for p in policies if "복지로" in p.get("source", "") or "bokjiro" in p.get("url", "")]
    pol_dict = {p["policy_id"]: p for p in bok}
    pool = json.load(open(DATA / "query_pool_v3.json"))

    # Stratified policy sample (50 random) — same seed as before
    pol_sample = random.sample([p["policy_id"] for p in bok], 50)

    # Construct multi-prefix violated queries
    cells_to_label = []
    queries_orig_per_subtopic = {}
    cats = ['welfare', 'employment', 'education', 'housing', 'health', 'culture', 'living']
    for c in cats:
        for sti, st in enumerate(pool[c]['subtopics']):
            orig_q = st['queries'][0]
            sub_id = f"{c}_{sti:02d}_0"
            queries_orig_per_subtopic[sub_id] = orig_q
            for prefix in PREFIXES[c]:
                violated_q = f"{prefix} {orig_q}"
                violated_qid = f"{c}_{sti:02d}_0__{prefix}"
                for pid in pol_sample:
                    cells_to_label.append((violated_qid, violated_q, pid))

    print(f"Total violated cells: {len(cells_to_label):,}")
    print(f"  = 33 subtopics × 3 prefixes × 50 policies = {33*3*50}")

    # Resume
    out_path = OUT_DIR / "multi_prefix_scores.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                d = json.loads(line)
                done.add((d["qid"], d["pid"]))
            except: pass
    todo = [(q, qt, p) for q, qt, p in cells_to_label if (q, p) not in done]
    print(f"Resume: done={len(done)}, todo={len(todo)}")

    # Test 5 first
    if len(done) < 5:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        with open(out_path, "a") as f:
            for q, qt, p in todo[:5]:
                pol = pol_dict[p]
                s = score_pair(client, qt, pol["name"], pol.get("summary"))
                f.write(json.dumps({"qid": q, "pid": p, "score": s}, ensure_ascii=False) + "\n")
                done.add((q, p))
        print("Test 5: passed")
        todo = [(q, qt, p) for q, qt, p in cells_to_label if (q, p) not in done]

    # Full
    print(f"Full: {len(todo)} pairs × 20 threads")
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
                    print(f"  [{n_done[0]}/{len(todo)}] rate={rate:.1f}/s")
    out_f.close()
    print(f"  done {time.time()-t0:.1f}s")

    # Analyze per prefix
    scores_by_prefix = defaultdict(list)
    for line in open(out_path):
        try:
            d = json.loads(line)
            qid = d["qid"]
            if "__" in qid and d.get("score") is not None:
                prefix = qid.split("__", 1)[1]
                scores_by_prefix[prefix].append(d["score"])
        except: pass

    # Original orthogonal subset (same cells, original queries)
    orig_scores = {}
    for line in open(REPO / "data/gt2_openai_final.jsonl"):
        d = json.loads(line)
        parts = d["custom_id"].split("__")
        if len(parts) == 2 and d.get("score") is not None:
            orig_scores[(parts[0], parts[1])] = d["score"]

    # For each prefix, compare to original
    print(f"\n=== Multi-prefix R1 ablation results ===")
    print(f"{'Prefix':<12s} {'n':>6s} {'avg':>6s} {'%≥1':>6s} {'%=2':>6s}  {'orig_avg':>9s} {'orig_%≥1':>9s}")

    # Build per-prefix subsample comparison
    summary = {}
    for line in open(out_path):
        try:
            d = json.loads(line)
            qid = d["qid"]
            if "__" not in qid: continue
            base_qid, prefix = qid.split("__", 1)
            if d.get("score") is None: continue
            summary.setdefault(prefix, {"violated": [], "orig": []})
            summary[prefix]["violated"].append((base_qid, d["pid"], d["score"]))
        except: pass

    for prefix, data in summary.items():
        viol = [s for _, _, s in data["violated"]]
        # match orig
        orig = []
        for bq, pid, _ in data["violated"]:
            o = orig_scores.get((bq, pid))
            if o is not None:
                orig.append(o)
        if not viol or not orig:
            continue
        n = len(viol)
        avg_v = sum(viol)/n
        gte1_v = sum(1 for s in viol if s >= 1)/n*100
        eq2_v = sum(1 for s in viol if s == 2)/n*100
        avg_o = sum(orig)/len(orig)
        gte1_o = sum(1 for s in orig if s >= 1)/len(orig)*100
        print(f"  {prefix:<10s} {n:>6d} {avg_v:>6.3f} {gte1_v:>5.1f}% {eq2_v:>5.1f}%   {avg_o:>9.3f} {gte1_o:>8.1f}%")

    # Aggregate across all prefixes
    all_viol = []
    all_orig = []
    for prefix, data in summary.items():
        for bq, pid, s in data["violated"]:
            all_viol.append(s)
            o = orig_scores.get((bq, pid))
            if o is not None:
                all_orig.append(o)
    print(f"\n  Overall (all 3 prefixes pooled):")
    if all_viol and all_orig:
        gte1_v_all = sum(1 for s in all_viol if s >= 1)/len(all_viol)*100
        gte1_o_all = sum(1 for s in all_orig if s >= 1)/len(all_orig)*100
        print(f"    Original orthogonal: %≥1 = {gte1_o_all:.1f}% (n={len(all_orig):,})")
        print(f"    R1-violated (3 prefix avg): %≥1 = {gte1_v_all:.1f}% (n={len(all_viol):,})")
        print(f"    Δ = {gte1_v_all - gte1_o_all:+.1f}pp")

    # Save
    json.dump({
        "n_prefixes_per_category": 3,
        "n_subtopics": 33,
        "n_policies_sampled": 50,
        "n_total_cells": sum(len(d["violated"]) for d in summary.values()),
        "per_prefix": {p: {"n": len(d["violated"]), "%≥1": sum(1 for _, _, s in d["violated"] if s>=1)/len(d["violated"])*100 if d["violated"] else 0} for p, d in summary.items()},
    }, open(OUT_DIR / "summary.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
