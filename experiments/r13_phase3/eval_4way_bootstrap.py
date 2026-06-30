"""4-way eval 재실행 + per-persona recall + bootstrap 95% CI + paired Wilcoxon.

eval_with_ig_baseline.py를 기반으로 per-persona 결과를 보존하고
bootstrap CI와 paired Wilcoxon p-value를 계산.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
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
OUT_PER = REPO / "experiments/r13_phase3/eval_4way_per_persona.json"
OUT_BOOT = REPO / "experiments/r13_phase3/eval_4way_bootstrap.json"

K = 10
MAX_TURNS = 5
CAND_POOL = 100
N_BOOT = 10000

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

    def retrieve_topk(pi, mask, k=K):
        s = soft_scores(pi, mask)
        cand = np.argsort(-s)[:CAND_POOL]
        rerank = cand[np.argsort(-SIM[pi, cand])]
        return [policy_ids[i] for i in rerank[:k]]

    def recall(retrieved, eligible_set, k=K):
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

    def select_ig(pi, persona, revealed, mask):
        cur_scores = soft_scores(pi, mask)
        cur_top = np.argsort(-cur_scores)[:CAND_POOL]
        candidates = [a for a in ATTRS if a not in revealed]
        if not candidates:
            return None
        best_attr = None
        best_ig = -1
        for attr in candidates:
            trial_mask = mask.copy()
            trial_mask[ATTRS.index(attr)] = 1
            trial_scores = soft_scores(pi, trial_mask)
            shrink = (cur_scores[cur_top].sum() - trial_scores[cur_top].sum())
            if shrink > best_ig:
                best_ig = shrink
                best_attr = attr
        return best_attr

    with open(TRAIN_PATH) as f:
        train_records = json.load(f)
    oracle_lookup = {(r["persona_id"], tuple(sorted(r["revealed"]))): r["next_best_attr"]
                     for r in train_records}

    def select_oracle(persona, revealed):
        return oracle_lookup.get((persona["persona_id"], tuple(sorted(revealed))))

    strategies = ["heuristic", "ig", "trained", "oracle"]
    per_persona = {s: {t: [] for t in range(MAX_TURNS + 1)} for s in strategies}

    for pi, persona in enumerate(test_personas):
        eligible_set = set(gt.get(persona["persona_id"], []))
        for sname in strategies:
            mask = np.zeros(len(ATTRS))
            revealed = set()
            for turn in range(MAX_TURNS + 1):
                ret = retrieve_topk(pi, mask)
                r = recall(ret, eligible_set)
                per_persona[sname][turn].append(r)
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

    # save per persona
    with open(OUT_PER, "w") as f:
        json.dump({s: {str(t): per_persona[s][t] for t in per_persona[s]}
                   for s in strategies}, f, indent=2)

    # bootstrap + Wilcoxon
    rng = np.random.default_rng(42)

    def boot_ci(vals):
        arr = np.array(vals)
        boots = np.empty(N_BOOT)
        n = len(arr)
        for i in range(N_BOOT):
            boots[i] = arr[rng.integers(0, n, n)].mean()
        return float(arr.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

    out = {"n_test": len(test_personas), "k": K, "n_boot": N_BOOT}
    for s in strategies:
        out[s] = {}
        for t in range(MAX_TURNS + 1):
            mean, lo, hi = boot_ci(per_persona[s][t])
            out[s][f"turn{t}"] = {"mean": round(mean, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}

    out["wilcoxon"] = {}
    pairs = [("trained", "heuristic"), ("trained", "ig"), ("ig", "heuristic")]
    for a, b in pairs:
        out["wilcoxon"][f"{a}_vs_{b}"] = {}
        for t in range(MAX_TURNS + 1):
            va, vb = per_persona[a][t], per_persona[b][t]
            if all(x == y for x, y in zip(va, vb)):
                out["wilcoxon"][f"{a}_vs_{b}"][f"turn{t}"] = {"p_value": 1.0, "note": "all equal"}
                continue
            try:
                stat, p = wilcoxon(va, vb, zero_method="zsplit")
                out["wilcoxon"][f"{a}_vs_{b}"][f"turn{t}"] = {"statistic": round(float(stat), 4), "p_value": round(float(p), 4)}
            except Exception as e:
                out["wilcoxon"][f"{a}_vs_{b}"][f"turn{t}"] = {"error": str(e)}

    with open(OUT_BOOT, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== Bootstrap 95% CI on Recall@10 (n={len(test_personas)}) ===")
    print(f"{'turn':<6s} {'heuristic':>22s} {'IG':>22s} {'trained':>22s} {'oracle':>22s}")
    for t in range(MAX_TURNS + 1):
        row = []
        for s in strategies:
            r = out[s][f"turn{t}"]
            row.append(f"{r['mean']:.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}]")
        print(f"{t:<6d} " + " ".join(f"{x:>22s}" for x in row))

    print("\n=== Paired Wilcoxon p-values ===")
    for pair, by_turn in out["wilcoxon"].items():
        print(f"\n  {pair}:")
        for tk, v in by_turn.items():
            if "p_value" in v:
                sig = "**" if v["p_value"] < 0.01 else ("*" if v["p_value"] < 0.05 else "")
                print(f"    {tk}: p={v['p_value']:.4f} {sig}")
            else:
                print(f"    {tk}: {v}")

    print(f"\n저장: {OUT_BOOT}")


if __name__ == "__main__":
    main()
