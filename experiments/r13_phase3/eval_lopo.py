"""LOPO (Leave-One-Persona-Out) cross-validation on the 144 train personas.

Reviewer concern: n=36 test set is too small to give per-persona variance bounds.
LOPO over the 144-persona TRAIN set produces a fair, leak-free generalisation curve.

For each persona p in 144 train set:
  1. Fit RandomForestClassifier on training_data records EXCLUDING all rows for p
  2. Run the same conversational eval as eval_leakfree.py on persona p (turns 0-5)
  3. Record Recall@5/10/20 per turn

Aggregate: mean ± 95% bootstrap CI of Recall@10 over 144 personas at each turn.
Compare against heuristic priority and oracle (same protocol).

Output: experiments/r13_phase3/eval_lopo.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eval"))
sys.path.insert(0, str(REPO / "experiments/r13_phase2"))
sys.path.insert(0, str(REPO / "experiments/r13_phase3"))

from baselines.base import persona_query  # noqa: E402
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
OUT_PATH = REPO / "experiments/r13_phase3/eval_lopo.json"

KS = [5, 10, 20]
MAX_TURNS = 5
CAND_POOL = 100
N_ESTIMATORS_LOPO = 200  # downgrade if too slow

HEURISTIC_PRIORITY = [
    "special_targets", "age", "sido", "household_types", "income_level",
    "disability", "education", "employment", "gender", "sigungu",
]


def make_X_y(records, persona_dict):
    X, y, pids = [], [], []
    for r in records:
        p = persona_dict[r["persona_id"]]
        feat = persona_features(p) + revealed_features(r["revealed"])
        X.append(feat)
        y.append(ATTR2IDX[r["next_best_attr"]])
        pids.append(r["persona_id"])
    return np.array(X), np.array(y), pids


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = arr[idx].mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def paired_bootstrap_ci(a, b, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    diff = a - b
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(diff.mean()), lo, hi


def main():
    t_start = time.time()

    with open(POLICIES_PATH) as f:
        policies = [p for p in json.load(f) if is_bokjiro(p)]
    with open(PERSONAS_PATH) as f:
        all_personas = json.load(f)
    with open(GT_PATH) as f:
        gt = json.load(f)
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    with open(TRAIN_PATH) as f:
        train_records = json.load(f)

    # 동일 split (random_state=42, 80/20 페르소나 단위)
    persona_ids_all = [p["persona_id"] for p in all_personas]
    train_ids, test_ids = train_test_split(persona_ids_all, test_size=0.2, random_state=42)
    train_set = set(train_ids)
    train_personas_only = [p for p in all_personas if p["persona_id"] in train_set]
    print(f"전체 {len(all_personas)}, train {len(train_personas_only)}, test {len(test_ids)}")
    print(f"LOPO target: {len(train_personas_only)} personas (excluded from official 36 test)")

    # Marginals: train set only (leak-free)
    marginals = compute_marginals(train_personas_only)

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

    # Encode queries for the 144 train personas (LOPO targets)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    queries = [persona_query(p) for p in train_personas_only]
    persona_q_emb = model.encode(queries, convert_to_numpy=True, show_progress_bar=False)
    persona_q_emb = persona_q_emb / (np.linalg.norm(persona_q_emb, axis=1, keepdims=True) + 1e-9)
    SIM = persona_q_emb @ policy_emb.T

    SAT = np.array([persona_satisfy_vector(p, tag_list) for p in train_personas_only])

    def soft_scores(pi, mask, personas_local):
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
        age = personas_local[pi].get("age")
        if mask[ATTRS.index("age")] > 0 and age is not None:
            scores += np.where(age < age_min, np.log(1e-6), 0.0)
            scores += np.where(age > age_max, np.log(1e-6), 0.0)
        p_sido = personas_local[pi].get("sido")
        p_sigungu = personas_local[pi].get("sigungu")
        if mask[ATTRS.index("sido")] > 0 and p_sido:
            for i, lev in enumerate(pol_level):
                if lev in ("시도", "시군구") and pol_sido[i] and pol_sido[i] != p_sido:
                    scores[i] += np.log(1e-6)
        if mask[ATTRS.index("sigungu")] > 0 and p_sigungu:
            for i, lev in enumerate(pol_level):
                if lev == "시군구" and pol_sigungu[i] and pol_sigungu[i] != p_sigungu:
                    scores[i] += np.log(1e-6)
        return scores

    def retrieve_topk(pi, mask, k, personas_local):
        s = soft_scores(pi, mask, personas_local)
        cand = np.argsort(-s)[:CAND_POOL]
        rerank = cand[np.argsort(-SIM[pi, cand])]
        return [policy_ids[i] for i in rerank[:k]]

    def recall(retrieved, eligible_set, k):
        if not eligible_set:
            return 0.0
        return len(set(retrieved[:k]) & eligible_set) / len(eligible_set)

    # Filter training records to those in the 144 train set (don't use 36 test rows in any LOPO fold)
    persona_dict = {p["persona_id"]: p for p in all_personas}
    train_records_only = [r for r in train_records if r["persona_id"] in train_set]
    print(f"Train records (within 144 train pool): {len(train_records_only)}")

    X_full, y_full, rec_pids = make_X_y(train_records_only, persona_dict)
    rec_pids_arr = np.array(rec_pids)

    # Oracle lookup: still uses all training_data records for legitimate (persona_id, revealed) keys
    oracle_lookup = {(r["persona_id"], tuple(sorted(r["revealed"]))): r["next_best_attr"]
                     for r in train_records}

    # Run LOPO over 144 train personas
    strategies = ["heuristic", "trained", "oracle"]
    # results[strategy][turn] -> list of {recall@k: ...} per persona
    results = {s: {turn: [] for turn in range(MAX_TURNS + 1)} for s in strategies}
    # Per-persona recall@10 records (paired): {strategy: {turn: list of recall@10 ordered by persona idx}}
    paired_recall10 = {s: {turn: [] for turn in range(MAX_TURNS + 1)} for s in strategies}

    print(f"\n=== LOPO start (n={len(train_personas_only)}, n_estimators={N_ESTIMATORS_LOPO}) ===")
    for pi, persona in enumerate(train_personas_only):
        pid = persona["persona_id"]
        # Build LOPO training fold: all rows EXCEPT those with persona_id == pid
        mask_keep = rec_pids_arr != pid
        X_tr = X_full[mask_keep]
        y_tr = y_full[mask_keep]

        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS_LOPO, max_depth=10, random_state=42, n_jobs=-1
        )
        clf.fit(X_tr, y_tr)

        eligible_set = set(gt.get(pid, []))

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
            # proba may be shorter than len(ATTRS) if some classes were absent during fit
            classes = clf.classes_
            score = {int(c): float(proba[i]) for i, c in enumerate(classes)}
            return ATTRS[max(cand_idx, key=lambda i: score.get(i, 0.0))]

        def select_oracle(persona, revealed):
            return oracle_lookup.get((persona["persona_id"], tuple(sorted(revealed))))

        for sname, sfn in [("heuristic", select_heuristic),
                           ("trained", select_trained),
                           ("oracle", select_oracle)]:
            mask_v = np.zeros(len(ATTRS))
            revealed = set()
            for turn in range(MAX_TURNS + 1):
                ret = retrieve_topk(pi, mask_v, max(KS), train_personas_only)
                rec = {f"recall@{k}": recall(ret, eligible_set, k) for k in KS}
                results[sname][turn].append(rec)
                paired_recall10[sname][turn].append(rec["recall@10"])
                if turn == MAX_TURNS:
                    break
                next_attr = sfn(persona, revealed)
                if next_attr is None or next_attr in revealed:
                    # Stay put — append same numbers for remaining turns
                    for t_remain in range(turn + 1, MAX_TURNS + 1):
                        ret_r = retrieve_topk(pi, mask_v, max(KS), train_personas_only)
                        rec_r = {f"recall@{k}": recall(ret_r, eligible_set, k) for k in KS}
                        results[sname][t_remain].append(rec_r)
                        paired_recall10[sname][t_remain].append(rec_r["recall@10"])
                    break
                revealed.add(next_attr)
                mask_v[ATTRS.index(next_attr)] = 1

        if (pi + 1) % 10 == 0 or pi == 0:
            elapsed = time.time() - t_start
            rate = (pi + 1) / elapsed
            eta = (len(train_personas_only) - (pi + 1)) / max(rate, 1e-9)
            print(f"  [{pi+1:3d}/{len(train_personas_only)}] elapsed={elapsed:.1f}s, eta={eta:.1f}s "
                  f"(turn5 recall@10 trained mean so far={np.mean(paired_recall10['trained'][MAX_TURNS]):.4f})")

    # Aggregate: bootstrap CI of mean recall@k per strategy per turn
    aggregated = {"n_personas": len(train_personas_only),
                  "n_estimators": N_ESTIMATORS_LOPO,
                  "ks": KS,
                  "max_turns": MAX_TURNS}
    for sname in strategies:
        per_turn_mean = []
        per_turn_lo = []
        per_turn_hi = []
        per_turn_recall10_lists = {}
        for turn in range(MAX_TURNS + 1):
            recs = results[sname][turn]
            r10 = [r["recall@10"] for r in recs]
            mean, lo, hi = bootstrap_ci(r10)
            per_turn_mean.append(mean)
            per_turn_lo.append(lo)
            per_turn_hi.append(hi)
            per_turn_recall10_lists[f"turn{turn}"] = r10
        # Also record means for recall@5 and recall@20
        per_turn_other = {}
        for k in KS:
            per_turn_other[f"recall@{k}_mean"] = []
            for turn in range(MAX_TURNS + 1):
                vals = [r[f"recall@{k}"] for r in results[sname][turn]]
                per_turn_other[f"recall@{k}_mean"].append(round(float(np.mean(vals)), 4))
        aggregated[f"{sname}_lopo"] = {
            "recall@10_per_turn": [round(x, 4) for x in per_turn_mean],
            "ci95_lo": [round(x, 4) for x in per_turn_lo],
            "ci95_hi": [round(x, 4) for x in per_turn_hi],
            "n_personas": len(train_personas_only),
            **{k: v for k, v in per_turn_other.items()},
            "per_persona_recall@10": per_turn_recall10_lists,
        }

    # Paired bootstrap CI: trained vs heuristic at turn 5
    a = paired_recall10["trained"][MAX_TURNS]
    b = paired_recall10["heuristic"][MAX_TURNS]
    diff_mean, diff_lo, diff_hi = paired_bootstrap_ci(a, b)
    aggregated["diff_at_turn5"] = {
        "trained_minus_heuristic": round(diff_mean, 4),
        "ci95": [round(diff_lo, 4), round(diff_hi, 4)],
    }

    # Trained vs oracle gap at turn 5
    o = paired_recall10["oracle"][MAX_TURNS]
    diff_to_oracle, lo_o, hi_o = paired_bootstrap_ci(a, o)
    aggregated["diff_at_turn5_trained_minus_oracle"] = {
        "mean": round(diff_to_oracle, 4),
        "ci95": [round(lo_o, 4), round(hi_o, 4)],
    }

    with open(OUT_PATH, "w") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT_PATH}")
    print(f"\n=== LOPO Recall@10 (mean [95% bootstrap CI]) ===")
    for sname in strategies:
        row = aggregated[f"{sname}_lopo"]
        print(f"\n  {sname}:")
        for turn in range(MAX_TURNS + 1):
            print(f"    turn {turn}: {row['recall@10_per_turn'][turn]:.4f} "
                  f"[{row['ci95_lo'][turn]:.4f}, {row['ci95_hi'][turn]:.4f}]")

    # 1-line final summary
    t5 = aggregated["trained_lopo"]["recall@10_per_turn"][MAX_TURNS]
    t5_lo = aggregated["trained_lopo"]["ci95_lo"][MAX_TURNS]
    t5_hi = aggregated["trained_lopo"]["ci95_hi"][MAX_TURNS]
    d_mean = aggregated["diff_at_turn5"]["trained_minus_heuristic"]
    d_lo, d_hi = aggregated["diff_at_turn5"]["ci95"]
    n = aggregated["trained_lopo"]["n_personas"]
    elapsed_total = time.time() - t_start
    print(f"\nTotal compute time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(
        f"\nLOPO Recall@10 turn5 = {t5:.3f} [95% CI {t5_lo:.3f}-{t5_hi:.3f}] "
        f"over n={n} train personas; trained vs heuristic delta = "
        f"{d_mean:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}] (paired bootstrap)"
    )


if __name__ == "__main__":
    main()
