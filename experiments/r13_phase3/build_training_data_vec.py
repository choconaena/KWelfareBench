"""학습 데이터 생성 (벡터화 버전).

핵심 최적화:
  - labels → numpy matrix (4937 × 53)
  - persona 충족 여부 → 53-dim vector
  - revealed mask → 어느 tag-attribute가 알려졌는지
  - soft_score = vectorized log P sum

각 페르소나에 대해 turn 0..MAX_TURNS-1까지 oracle next-best attr 측정.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eval"))
sys.path.insert(0, str(REPO / "experiments/r13_phase2"))
sys.path.insert(0, str(REPO / "experiments/r13_phase3"))

from baselines.base import persona_query, policy_text  # noqa: E402
from compute_ground_truth_v3 import (  # noqa: E402
    SPECIAL_TAG_MAP,
    persona_education_tag,
    persona_employment_tag,
    persona_household_tags,
    persona_required_special_tags,
)
from soft_eligibility import compute_marginals  # noqa: E402

POLICIES_PATH = REPO / "data/processed/policies.json"
PERSONAS_PATH = REPO / "docs/papers/ai4good/data/personas_v2.json"
GT_PATH = REPO / "docs/papers/ai4good/data/ground_truth_v3.json"
LABELS_PATH = REPO / "experiments/r13_phase1/llm_labeling_full/labels.json"
EMB_CACHE = REPO / "experiments/r13_phase3/policy_emb.npy"
EMB_IDS = REPO / "experiments/r13_phase3/policy_ids.json"
OUT_PATH = REPO / "experiments/r13_phase3/training_data.json"

ATTRS = [
    "special_targets", "age", "sido", "household_types", "income_level",
    "disability", "education", "employment", "gender", "sigungu",
]
MAX_TURNS = 5
TOP_K = 10
CAND_POOL = 100  # soft top-N → dense rerank


def is_bokjiro(p):
    return "복지로" in p.get("source", "") or "bokjiro" in p.get("url", "")


# tag → 어느 attribute에 의존하는지 (학습/inference용)
TAG_TO_ATTR = {}


def build_tag_attr_map(tag_list):
    m = {}
    for t in tag_list:
        if t.startswith("personal.gender"):
            m[t] = "gender"
        elif t.startswith("personal.disability"):
            m[t] = "disability"
        elif t.startswith("economic.income"):
            m[t] = "income_level"
        elif t.startswith("economic.employment"):
            m[t] = "employment"
        elif t.startswith("household.type"):
            m[t] = "household_types"
        elif (t.startswith("household.pregnancy") or t.startswith("household.birth_order")
              or t.startswith("social.veteran") or t.startswith("social.immigrant")
              or t.startswith("social.vulnerable") or t.startswith("social.violence_victim")):
            m[t] = "special_targets"
        elif t.startswith("social.education"):
            m[t] = "education"
        else:
            m[t] = None  # policy.category, residence, age, region — 별도 처리
    return m


def persona_satisfy_vector(persona, tag_list):
    """페르소나가 각 tag를 충족하는지 binary vector."""
    sat = np.zeros(len(tag_list))
    g = persona.get("gender", "")
    has_dis = persona.get("disability") == "있음"
    il = persona.get("income_level", "")
    inc_detail = set(persona.get("income_detail", []) or [])
    is_basic = "기초" in il or "수급" in il or any("기초" in d or "수급" in d for d in inc_detail)
    is_secondary = "차상위" in il or any("차상위" in d for d in inc_detail)
    is_medical = any("의료급여" in d for d in inc_detail)
    et = persona_employment_tag(persona)
    hh_set = persona_household_tags(persona)
    edu = persona_education_tag(persona)
    spec_set = persona_required_special_tags(persona)

    for i, t in enumerate(tag_list):
        if t == "personal.gender.female_only":
            sat[i] = 1.0 if g == "여성" else 0.0
        elif t == "personal.gender.male_only":
            sat[i] = 1.0 if g == "남성" else 0.0
        elif t.startswith("personal.disability"):
            sat[i] = 1.0 if has_dis else 0.0
        elif t == "economic.income.basic_recipient":
            sat[i] = 1.0 if is_basic else 0.0
        elif t == "economic.income.secondary":
            sat[i] = 1.0 if (is_basic or is_secondary) else 0.0
        elif t == "economic.income.medical_aid":
            sat[i] = 1.0 if is_medical else 0.0
        elif t.startswith("economic.employment"):
            sat[i] = 1.0 if et == t else 0.0
        elif t.startswith("household.type"):
            sat[i] = 1.0 if t in hh_set else 0.0
        elif t.startswith("social.education"):
            sat[i] = 1.0 if edu == t else 0.0
        elif t in SPECIAL_TAG_MAP.values():
            sat[i] = 1.0 if t in spec_set else 0.0
    return sat


def main():
    print("Loading data...")
    with open(POLICIES_PATH) as f:
        policies = [p for p in json.load(f) if is_bokjiro(p)]
    with open(PERSONAS_PATH) as f:
        personas = json.load(f)
    with open(GT_PATH) as f:
        gt = json.load(f)
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    print(f"  policies: {len(policies)}, personas: {len(personas)}")

    # marginal
    marginals = compute_marginals(personas)

    # tag list (binary tags only, exclude meta/numeric/region/category)
    sample_lab = next(iter(labels.values()))
    tag_list = [t for t in sample_lab.keys()
                if not t.startswith("_")
                and not t.startswith("policy.category")
                and not t.startswith("personal.age")
                and not t.startswith("social.residence")
                and t != "economic.income.median_threshold"]
    n_tags = len(tag_list)
    print(f"  binary tags: {n_tags}")

    # labels matrix: policies × tags  (-1, 0, 1)
    policy_ids = [p["policy_id"] for p in policies]
    L = np.zeros((len(policies), n_tags), dtype=np.int8)
    for i, pid in enumerate(policy_ids):
        lab = labels[pid]
        for j, t in enumerate(tag_list):
            L[i, j] = lab.get(t, 0)

    # tag → attribute mapping
    tag_to_attr = build_tag_attr_map(tag_list)
    # 각 tag가 어느 attr에 의존하는지 → mask matrix
    tag_attr_idx = np.array([ATTRS.index(tag_to_attr[t]) if tag_to_attr[t] in ATTRS else -1
                              for t in tag_list])

    # marginal vector (each tag's prior P)
    M = np.array([marginals.get(t, 0.5) for t in tag_list])

    # numeric: age_min, age_max, region 별도 array
    age_min = np.array([labels[pid].get("personal.age.age_min") if labels[pid].get("personal.age.age_min") is not None else -1 for pid in policy_ids])
    age_max = np.array([labels[pid].get("personal.age.age_max") if labels[pid].get("personal.age.age_max") is not None else 999 for pid in policy_ids])
    pol_level = [p.get("region", {}).get("level", "") for p in policies]
    pol_sido = [p.get("region", {}).get("sido", "") for p in policies]
    pol_sigungu = [p.get("region", {}).get("sigungu", "") for p in policies]

    # embeddings
    print("Loading policy embeddings...")
    policy_emb = np.load(EMB_CACHE)
    with open(EMB_IDS) as f:
        cached_ids = json.load(f)
    assert cached_ids == policy_ids, "policy_ids mismatch"
    print(f"  policy_emb: {policy_emb.shape}")

    print("Encoding persona queries...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    queries = [persona_query(p) for p in personas]
    persona_q_emb = model.encode(queries, convert_to_numpy=True)
    persona_q_emb = persona_q_emb / (np.linalg.norm(persona_q_emb, axis=1, keepdims=True) + 1e-9)

    # persona × policy similarity (180 × 4937)
    print(f"Computing query-policy sim matrix...")
    SIM = persona_q_emb @ policy_emb.T  # (180, 4937)
    print(f"  SIM shape: {SIM.shape}")

    # persona satisfy matrices: 180 × n_tags
    print("Building persona satisfy vectors...")
    SAT = np.array([persona_satisfy_vector(p, tag_list) for p in personas])  # (180, n_tags)

    # Vectorized soft_score function
    def soft_scores_for_persona(persona_idx, revealed_mask):
        """
        persona_idx: 0..179
        revealed_mask: 10-dim 0/1 (어느 attribute 알려졌는지)
        returns: (n_policies,) log score
        """
        # tag별 attr known 여부
        attr_known = np.zeros(n_tags)
        for j in range(n_tags):
            ai = tag_attr_idx[j]
            if ai >= 0 and revealed_mask[ai] > 0:
                attr_known[j] = 1.0

        # P(satisfy tag) for this persona
        sat = SAT[persona_idx]
        # if known: P = sat (0 or 1); if unknown: P = marginal
        P = attr_known * sat + (1 - attr_known) * M  # (n_tags,)
        # clip
        P_clip = np.clip(P, 1e-6, 1.0)
        # 1-P for negative case
        oneP_clip = np.clip(1.0 - P, 1e-6, 1.0)

        # tag별: val=0 → contrib 0, val=1 → log(P), val=-1 → log(1-P)
        contrib = np.where(L > 0, np.log(P_clip)[None, :], 0.0)
        contrib += np.where(L < 0, np.log(oneP_clip)[None, :], 0.0)
        # contrib: (n_policies, n_tags)
        scores = contrib.sum(axis=1)  # (n_policies,)

        # age check
        persona_age = personas[persona_idx].get("age")
        if revealed_mask[ATTRS.index("age")] > 0 and persona_age is not None:
            scores += np.where(persona_age < age_min, np.log(1e-6), 0.0)
            scores += np.where(persona_age > age_max, np.log(1e-6), 0.0)

        # region
        p_sido = personas[persona_idx].get("sido")
        p_sigungu = personas[persona_idx].get("sigungu")
        if revealed_mask[ATTRS.index("sido")] > 0 and p_sido:
            for i, lev in enumerate(pol_level):
                if lev in ("시도", "시군구") and pol_sido[i] and pol_sido[i] != p_sido:
                    scores[i] += np.log(1e-6)
        if revealed_mask[ATTRS.index("sigungu")] > 0 and p_sigungu:
            for i, lev in enumerate(pol_level):
                if lev == "시군구" and pol_sigungu[i] and pol_sigungu[i] != p_sigungu:
                    scores[i] += np.log(1e-6)

        return scores

    def retrieve_topk(persona_idx, revealed_mask, k=TOP_K):
        scores = soft_scores_for_persona(persona_idx, revealed_mask)
        # 후보 풀 (soft top-CAND_POOL)
        cand_idx = np.argsort(-scores)[:CAND_POOL]
        # dense rerank
        dense_score = SIM[persona_idx, cand_idx]
        rerank = cand_idx[np.argsort(-dense_score)]
        return [policy_ids[i] for i in rerank[:k]]

    def recall_at_k(retrieved, eligible_set, k=TOP_K):
        if not eligible_set:
            return 0.0
        return len(set(retrieved[:k]) & eligible_set) / len(eligible_set)

    print("\nGenerating training data (vectorized greedy oracle)...")
    records = []
    t0 = time.time()
    for pi, persona in enumerate(personas):
        eligible_set = set(gt.get(persona["persona_id"], []))
        revealed_mask = np.zeros(len(ATTRS))

        for turn in range(MAX_TURNS):
            cur_retrieved = retrieve_topk(pi, revealed_mask)
            cur_recall = recall_at_k(cur_retrieved, eligible_set)

            cand_attrs = [a for a in ATTRS if revealed_mask[ATTRS.index(a)] == 0]
            if not cand_attrs:
                break

            attr_recalls = {}
            for attr in cand_attrs:
                trial_mask = revealed_mask.copy()
                trial_mask[ATTRS.index(attr)] = 1
                trial_retrieved = retrieve_topk(pi, trial_mask)
                attr_recalls[attr] = recall_at_k(trial_retrieved, eligible_set)

            best_attr = max(attr_recalls, key=attr_recalls.get)
            best_recall = attr_recalls[best_attr]

            records.append({
                "persona_id": persona["persona_id"],
                "group": persona.get("group"),
                "turn": turn,
                "revealed": [ATTRS[i] for i, m in enumerate(revealed_mask) if m > 0],
                "candidates": cand_attrs,
                "next_best_attr": best_attr,
                "recall_before": float(cur_recall),
                "recall_after": float(best_recall),
                "delta": float(best_recall - cur_recall),
                "attr_recalls": {k: float(v) for k, v in attr_recalls.items()},
            })
            revealed_mask[ATTRS.index(best_attr)] = 1

        if (pi + 1) % 30 == 0 or pi == len(personas) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (pi + 1) * (len(personas) - pi - 1)
            print(f"  [{pi+1}/{len(personas)}] {elapsed:.0f}s, ETA {eta:.0f}s", flush=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT_PATH} ({len(records)} records, {time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
