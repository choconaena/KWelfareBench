"""IG (Information Gain) dynamic baseline 추가 — Heuristic / IG / ML / Oracle 4-way 비교.

IG dynamic selector:
  - 매 turn마다 모든 candidate attribute의 information gain 계산
  - IG = entropy(eligible_pool_before) - E_attr[entropy(eligible_pool_after)]
  - Proxy: 단순 후보 정책 set의 expected size reduction이 가장 큰 attribute 선택
  - 페르소나별 동적이지만 학습 X (deterministic from current state)

36 test 페르소나 only (leak-free).
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eval"))
sys.path.insert(0, str(REPO / "experiments/r13_phase2"))
sys.path.insert(0, str(REPO / "experiments/r13_phase3"))

from baselines.base import persona_query  # noqa: E402
from compute_ground_truth_v3 import (  # noqa: E402
    SPECIAL_TAG_MAP,
    persona_education_tag,
    persona_employment_tag,
    persona_household_tags,
    persona_required_special_tags,
)
from soft_eligibility import compute_marginals  # noqa: E402
from train_selector import (  # noqa: E402
    ATTR2IDX,
    ATTRS,
    persona_features,
    revealed_features,
)
from build_training_data_vec import (  # noqa: E402
    build_tag_attr_map,
    is_bokjiro,
    persona_satisfy_vector,
)

POLICIES_PATH = REPO / "data/policies.json"
PERSONAS_PATH = REPO / "experiments/r13_phase3/personas_v2.json"
GT_PATH = REPO / "experiments/r13_phase3/ground_truth_v3.json"
LABELS_PATH = REPO / "experiments/r13_phase3/labels.json"
EMB_CACHE = REPO / "experiments/r13_phase3/policy_emb.npy"
EMB_IDS = REPO / "experiments/r13_phase3/policy_ids.json"
TRAIN_PATH = REPO / "experiments/r13_phase3/training_data.json"
MODEL_PATH = REPO / "experiments/r13_phase3/selector_model.pkl"
OUT_PATH = REPO / "experiments/r13_phase3/eval_4way.json"

KS = [5, 10, 20]
MAX_TURNS = 5
CAND_POOL = 100

HEURISTIC_PRIORITY = [
    "special_targets", "age", "sido", "household_types", "income_level",
    "disability", "education", "employment", "gender", "sigungu",
]


def main():
    with open(POLICIES_PATH) as f:
        policies = [p for p in json.load(f) if is_bokjiro(p)]
    with open(PERSONAS_PATH) as f:
        all_personas = json.load(f)
    with open(GT_PATH) as f:
        gt = json.load(f)
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    with open(MODEL_PATH, "rb") as f:
        sel = pickle.load(f)
    clf = sel["model"]

    persona_ids_all = [p["persona_id"] for p in all_personas]
    train_ids, test_ids = train_test_split(persona_ids_all, test_size=0.2, random_state=42)
    test_set = set(test_ids)
    test_personas = [p for p in all_personas if p["persona_id"] in test_set]
    train_personas = [p for p in all_personas if p["persona_id"] not in test_set]
    print(f"전체 {len(all_personas)}, test {len(test_personas)}, train {len(train_personas)}")

    marginals = compute_marginals(train_personas)
    sample_lab = next(iter(labels.values()))
    tag_list = [t for t in sample_lab.keys()
                if not t.startswith("_") and not t.startswith("policy.category")
                and not t.startswith("personal.age") and not t.startswith("social.residence")
                and t != "economic.income.median_threshold"]
    n_tags = len(tag_list)

    policy_ids = [p["policy_id"] for p in policies]
    L = np.zeros((len(policies), n_tags), dtype=np.int8)
    for i, pid in enumerate(policy_ids):
        for j, t in enumerate(tag_list):
            L[i, j] = labels[pid].get(t, 0)
    tag_to_attr = build_tag_attr_map(tag_list)
    tag_attr_idx = np.array([ATTRS.index(tag_to_attr[t]) if tag_to_attr[t] in ATTRS else -1
                              for t in tag_list])
    M = np.array([marginals.get(t, 0.5) for t in tag_list])

    age_min = np.array([labels[pid].get("personal.age.age_min") if labels[pid].get("personal.age.age_min") is not None else -1 for pid in policy_ids])
    age_max = np.array([labels[pid].get("personal.age.age_max") if labels[pid].get("personal.age.age_max") is not None else 999 for pid in policy_ids])
    pol_level = [p.get("region", {}).get("level", "") for p in policies]
    pol_sido = [p.get("region", {}).get("sido", "") for p in policies]
    pol_sigungu = [p.get("region", {}).get("sigungu", "") for p in policies]

    policy_emb = np.load(EMB_CACHE)
    with open(EMB_IDS) as f:
        cached = json.load(f)
    assert cached == policy_ids

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    queries = [persona_query(p) for p in test_personas]
    persona_q_emb = model.encode(queries, convert_to_numpy=True)
    persona_q_emb = persona_q_emb / (np.linalg.norm(persona_q_emb, axis=1, keepdims=True) + 1e-9)
    SIM = persona_q_emb @ policy_emb.T

    SAT = np.array([persona_satisfy_vector(p, tag_list) for p in test_personas])

    def soft_scores(pi, mask):
        attr_known = np.zeros(n_tags)
        for j in range(n_tags):
            ai = tag_attr_idx[j]
            if ai >= 0 and mask[ai] > 0:
                attr_known[j] = 1.0
        sat = SAT[pi]
        P = attr_known * sat + (1 - attr_known) * M
        P_clip = np.clip(P, 1e-6, 1.0)
        oneP_clip = np.clip(1.0 - P, 1e-6, 1.0)
        contrib = np.where(L > 0, np.log(P_clip)[None, :], 0.0)
        contrib += np.where(L < 0, np.log(oneP_clip)[None, :], 0.0)
        scores = contrib.sum(axis=1)
        age = test_personas[pi].get("age")
        if mask[ATTRS.index("age")] > 0 and age is not None:
            scores += np.where(age < age_min, np.log(1e-6), 0.0)
            scores += np.where(age > age_max, np.log(1e-6), 0.0)
        p_sido = test_personas[pi].get("sido")
        p_sigungu = test_personas[pi].get("sigungu")
        if mask[ATTRS.index("sido")] > 0 and p_sido:
            for i, lev in enumerate(pol_level):
                if lev in ("시도", "시군구") and pol_sido[i] and pol_sido[i] != p_sido:
                    scores[i] += np.log(1e-6)
        if mask[ATTRS.index("sigungu")] > 0 and p_sigungu:
            for i, lev in enumerate(pol_level):
                if lev == "시군구" and pol_sigungu[i] and pol_sigungu[i] != p_sigungu:
                    scores[i] += np.log(1e-6)
        return scores

    def retrieve_topk(pi, mask, k):
        s = soft_scores(pi, mask)
        cand = np.argsort(-s)[:CAND_POOL]
        rerank = cand[np.argsort(-SIM[pi, cand])]
        return [policy_ids[i] for i in rerank[:k]]

    def recall(retrieved, eligible_set, k):
        if not eligible_set:
            return 0.0
        return len(set(retrieved[:k]) & eligible_set) / len(eligible_set)

    def select_heuristic(persona, revealed):
        for a in HEURISTIC_PRIORITY:
            if a not in revealed:
                return a
        return None

    def select_trained(persona, revealed):
        feat = persona_features(persona) + revealed_features(sorted(revealed))
        feat_arr = np.array([feat])
        cand_idx = [ATTR2IDX[a] for a in ATTRS if a not in revealed]
        if not cand_idx:
            return None
        proba = clf.predict_proba(feat_arr)[0]
        return ATTRS[max(cand_idx, key=lambda i: proba[i])]

    # IG dynamic baseline:
    # 매 turn마다 모든 candidate attribute가 expected eligible-pool-size를 얼마나 줄이는지 측정
    # IG = log2(N_before) - log2(E[N_after])  (uniform prior)
    # Proxy: candidate attribute reveal했을 때 soft score의 entropy 감소
    def select_ig(pi, persona, revealed, mask):
        # 현재 pool: top-CAND_POOL 정책의 soft score
        cur_scores = soft_scores(pi, mask)
        cur_top = np.argsort(-cur_scores)[:CAND_POOL]
        n_cur = len(cur_top)

        candidates = [a for a in ATTRS if a not in revealed]
        if not candidates:
            return None

        best_attr = None
        best_ig = -1
        for attr in candidates:
            trial_mask = mask.copy()
            trial_mask[ATTRS.index(attr)] = 1
            trial_scores = soft_scores(pi, trial_mask)
            trial_top = np.argsort(-trial_scores)[:CAND_POOL]
            # IG proxy: log(현재 pool 평균 score) - log(after pool 평균 score)
            # 더 정확: pool 안에 있는 비율 차이로 entropy
            shrink = (cur_scores[cur_top].sum() - trial_scores[cur_top].sum())
            ig = shrink  # 감소량이 클수록 정보이득 큼
            if ig > best_ig:
                best_ig = ig
                best_attr = attr
        return best_attr

    with open(TRAIN_PATH) as f:
        train_records = json.load(f)
    oracle_lookup = {(r["persona_id"], tuple(sorted(r["revealed"]))): r["next_best_attr"]
                     for r in train_records}

    def select_oracle(persona, revealed):
        return oracle_lookup.get((persona["persona_id"], tuple(sorted(revealed))))

    strategies = ["heuristic", "ig", "trained", "oracle"]
    results = {s: {turn: [] for turn in range(MAX_TURNS + 1)} for s in strategies}

    for pi, persona in enumerate(test_personas):
        eligible_set = set(gt.get(persona["persona_id"], []))
        for sname in strategies:
            mask = np.zeros(len(ATTRS))
            revealed = set()
            for turn in range(MAX_TURNS + 1):
                ret = retrieve_topk(pi, mask, max(KS))
                rec = {f"recall@{k}": recall(ret, eligible_set, k) for k in KS}
                results[sname][turn].append(rec)
                if turn == MAX_TURNS:
                    break
                if sname == "heuristic":
                    next_attr = select_heuristic(persona, revealed)
                elif sname == "ig":
                    next_attr = select_ig(pi, persona, revealed, mask)
                elif sname == "trained":
                    next_attr = select_trained(persona, revealed)
                else:
                    next_attr = select_oracle(persona, revealed)
                if next_attr is None or next_attr in revealed:
                    break
                revealed.add(next_attr)
                mask[ATTRS.index(next_attr)] = 1

    summary = {"n_test_personas": len(test_personas)}
    for sname in strategies:
        summary[sname] = {}
        for turn in range(MAX_TURNS + 1):
            recs = results[sname][turn]
            for k in KS:
                vals = [r[f"recall@{k}"] for r in recs]
                summary[sname][f"turn{turn}_recall@{k}"] = round(sum(vals) / len(vals), 4) if vals else 0.0

    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'turn':<6s} {'metric':<12s} {'heuristic':>10s} {'IG':>10s} {'trained':>10s} {'oracle':>10s}")
    for turn in range(MAX_TURNS + 1):
        for k in [10]:
            row = " ".join(f"{summary[s][f'turn{turn}_recall@{k}']:>10.4f}" for s in strategies)
            print(f"{turn:<6d} recall@{k:<6d} {row}")
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
