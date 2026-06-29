"""GT-3 = GT-1 ∩ GT-2: per (persona, query), 자격×주제관련 모두 만족하는 정책 집합.

6 매트릭스:
  GT-1 (region) / GT-1 (no_region) / GT-2 (single) /
  GT-3 strict (region, GT-2≥2) / GT-3 lenient (region, GT-2≥1) /
  GT-3 strict (no_region, GT-2≥2) / GT-3 lenient (no_region, GT-2≥1)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
GT2_OAI = REPO / "data/gt2_openai_final.jsonl"
QUERY_POOL = DATA / "query_pool_v3.json"


def load_gt1(path):
    """{persona_id: [policy_ids]}"""
    return json.load(open(path))


def load_gt2():
    """{(query_id, policy_id): score}"""
    gt2 = {}
    for line in open(GT2_OAI):
        d = json.loads(line)
        cid = d["custom_id"]
        # custom_id format: "{cat}_{st_idx:02d}_{q_idx}__{policy_id}"
        parts = cid.split("__")
        if len(parts) != 2:
            continue
        gt2[(parts[0], parts[1])] = d["score"]
    return gt2


def query_id_list():
    """Returns list of (query_id, query_text, category, subtopic) in order matching scores."""
    pool = json.load(open(QUERY_POOL))
    cats = ['welfare', 'employment', 'education', 'housing', 'health', 'culture', 'living']
    out = []
    for c in cats:
        for sti, st in enumerate(pool[c]['subtopics']):
            for qi, q in enumerate(st['queries']):
                qid = f"{c}_{sti:02d}_{qi}"
                out.append({"query_id": qid, "text": q, "category": c, "subtopic": st['subtopic']})
    return out


def build_gt3(gt1: dict, gt2: dict, queries: list, threshold: int) -> dict:
    """
    GT-3[persona_id][query_id] = list of policy_ids satisfying:
      - policy in gt1[persona_id] (eligibility)
      - gt2[(query_id, policy)] >= threshold (topic match)
    """
    gt3 = {}
    qids = [q["query_id"] for q in queries]
    for pid_str, eligible_pols in gt1.items():
        elig_set = set(eligible_pols)
        gt3[pid_str] = {}
        for qid in qids:
            matched = []
            for pol in elig_set:
                score = gt2.get((qid, pol))
                if score is not None and score >= threshold:
                    matched.append(pol)
            gt3[pid_str][qid] = matched
    return gt3


def stats(gt3):
    """target set sizes 통계."""
    sizes = []
    n_empty = 0
    for pid, qmap in gt3.items():
        for qid, lst in qmap.items():
            sizes.append(len(lst))
            if len(lst) == 0:
                n_empty += 1
    import statistics
    return {
        "n_persona_query_pairs": len(sizes),
        "n_empty_target_sets": n_empty,
        "pct_empty": round(n_empty / len(sizes) * 100, 2),
        "target_size_mean": round(statistics.mean(sizes), 2),
        "target_size_median": int(statistics.median(sizes)),
        "target_size_max": max(sizes),
    }


def main():
    gt1_region = load_gt1(DATA / "ground_truth_v4.json")
    gt1_no_region = load_gt1(DATA / "ground_truth_v4_no_region.json")
    gt2 = load_gt2()
    queries = query_id_list()
    print(f"GT-1 region: {len(gt1_region)} personas")
    print(f"GT-2 (q,p) pairs: {len(gt2):,}")
    print(f"Queries: {len(queries)}\n")

    for region_setting in ["region", "no_region"]:
        gt1 = gt1_region if region_setting == "region" else gt1_no_region
        for thresh, name in [(1, "lenient"), (2, "strict")]:
            gt3 = build_gt3(gt1, gt2, queries, thresh)
            s = stats(gt3)
            tag = f"gt3_{region_setting}_{name}"
            out = DATA / f"{tag}.json"
            json.dump(gt3, open(out, "w", encoding="utf-8"), ensure_ascii=False)
            sout = DATA / f"{tag}_stats.json"
            json.dump(s, open(sout, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"=== {tag} (GT-2 score >= {thresh}) ===")
            print(f"  (persona×query) pairs: {s['n_persona_query_pairs']:,}")
            print(f"  empty target sets: {s['n_empty_target_sets']:,} ({s['pct_empty']}%)")
            print(f"  target size: mean={s['target_size_mean']}, median={s['target_size_median']}, max={s['target_size_max']}")
            print(f"  saved {out.name}\n")


if __name__ == "__main__":
    main()
