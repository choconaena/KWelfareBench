"""Clarification-strategy baselines: ε-greedy on IG and Thompson sampling on IG.

Reviewer concern: paper claims CLARINET wasn't reproduced due to compute budget,
but a simpler clarification baseline like ε-greedy or Thompson sampling on
attribute information gain (IG) should be added.

Strategies (evaluated on the leak-free 36 test personas):
  1. trained: existing RandomForest selector (selector_model.pkl)
  2. epsilon_greedy_ig (ε=0.1): with prob 1-ε take argmax IG, with prob ε pick random
  3. thompson_ig: Beta-Bernoulli posterior over normalised IG, per-persona, per-turn updates
  4. heuristic: HEURISTIC_PRIORITY ordering
  5. random: uniform random selection over unrevealed attributes
  6. oracle: lookup (persona_id, revealed) -> next_best_attr from training_data

IG definition (matching eval_with_ig_baseline.py):
  IG(attr) = sum_{i in cur_top_K}( cur_score_i - score_i_after_revealing_attr )
  i.e. how much the current top-pool's total log-likelihood drops when this attribute
  is revealed (hypothetically agreed-with-persona). This is the same proxy used in
  the existing IG selector, so the comparison is internally consistent.

For ε-greedy: pick argmax IG with prob 1-ε else uniform among unrevealed attributes.
For Thompson: each attribute starts at Beta(1,1). At each turn, sample from each
posterior, pick argmax sampled value (restricting to unrevealed). Reveal it,
observe normalised IG of the revealed attribute as the reward, update posterior:
  α_a += r, β_a += (1 - r), where r = IG_normalised in [0, 1].

Output: experiments/r13_phase3/eval_clarification_baselines.json
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
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

POLICIES_PATH = REPO / "data/processed/policies.json"
PERSONAS_PATH = REPO / "docs/papers/ai4good/data/personas_v2.json"
GT_PATH = REPO / "docs/papers/ai4good/data/ground_truth_v3.json"
LABELS_PATH = REPO / "experiments/r13_phase1/llm_labeling_full/labels.json"
EMB_CACHE = REPO / "experiments/r13_phase3/policy_emb.npy"
EMB_IDS = REPO / "experiments/r13_phase3/policy_ids.json"
TRAIN_PATH = REPO / "experiments/r13_phase3/training_data.json"
MODEL_PATH = REPO / "experiments/r13_phase3/selector_model.pkl"
OUT_PATH = REPO / "experiments/r13_phase3/eval_clarification_baselines.json"

KS = [5, 10, 20]
MAX_TURNS = 5
CAND_POOL = 100
EPSILON = 0.1
SEED = 42

HEURISTIC_PRIORITY = [
    "special_targets", "age", "sido", "household_types", "income_level",
    "disability", "education", "employment", "gender", "sigungu",
]


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
    rng = np.random.default_rng(SEED)

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

    # Same 80/20 split as eval_leakfree.py / train_selector.py
    persona_ids_all = [p["persona_id"] for p in all_personas]
    train_ids, test_ids = train_test_split(persona_ids_all, test_size=0.2, random_state=42)
    test_set = set(test_ids)
    test_personas = [p for p in all_personas if p["persona_id"] in test_set]
    train_personas = [p for p in all_personas if p["persona_id"] not in test_set]
    print(f"전체 {len(all_personas)}, train {len(train_personas)}, test {len(test_personas)}")

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
    persona_q_emb = model.encode(queries, convert_to_numpy=True, show_progress_bar=False)
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

    # Compute IG for each candidate attribute given current state.
    # Returns dict {attr: ig_value (raw)} for unrevealed attrs.
    def compute_ig_table(pi, mask, revealed):
        cur_scores = soft_scores(pi, mask)
        cur_top = np.argsort(-cur_scores)[:CAND_POOL]
        ig = {}
        candidates = [a for a in ATTRS if a not in revealed]
        for attr in candidates:
            trial_mask = mask.copy()
            trial_mask[ATTRS.index(attr)] = 1
            trial_scores = soft_scores(pi, trial_mask)
            shrink = (cur_scores[cur_top].sum() - trial_scores[cur_top].sum())
            ig[attr] = float(shrink)
        return ig

    def select_heuristic(persona, revealed, **_):
        for a in HEURISTIC_PRIORITY:
            if a not in revealed:
                return a
        return None

    def select_trained(persona, revealed, **_):
        feat = persona_features(persona) + revealed_features(sorted(revealed))
        feat_arr = np.array([feat])
        cand_idx = [ATTR2IDX[a] for a in ATTRS if a not in revealed]
        if not cand_idx:
            return None
        proba = clf.predict_proba(feat_arr)[0]
        return ATTRS[max(cand_idx, key=lambda i: proba[i])]

    def select_random(persona, revealed, *, rng_local, **_):
        candidates = [a for a in ATTRS if a not in revealed]
        if not candidates:
            return None
        return candidates[int(rng_local.integers(0, len(candidates)))]

    def select_eps_greedy_ig(persona, revealed, *, rng_local, pi, mask, **_):
        candidates = [a for a in ATTRS if a not in revealed]
        if not candidates:
            return None
        if rng_local.random() < EPSILON:
            return candidates[int(rng_local.integers(0, len(candidates)))]
        ig = compute_ig_table(pi, mask, revealed)
        return max(candidates, key=lambda a: ig.get(a, 0.0))

    def select_thompson_ig(persona, revealed, *, rng_local, pi, mask, posterior, **_):
        candidates = [a for a in ATTRS if a not in revealed]
        if not candidates:
            return None
        # Sample from each candidate's Beta posterior
        sampled = {}
        for a in candidates:
            alpha, beta = posterior[a]
            sampled[a] = float(rng_local.beta(alpha, beta))
        return max(candidates, key=lambda a: sampled[a])

    with open(TRAIN_PATH) as f:
        train_records = json.load(f)
    oracle_lookup = {(r["persona_id"], tuple(sorted(r["revealed"]))): r["next_best_attr"]
                     for r in train_records}

    def select_oracle(persona, revealed, **_):
        return oracle_lookup.get((persona["persona_id"], tuple(sorted(revealed))))

    strategies = ["heuristic", "random", "epsilon_greedy_ig", "thompson_ig", "trained", "oracle"]
    selectors = {
        "heuristic": select_heuristic,
        "random": select_random,
        "epsilon_greedy_ig": select_eps_greedy_ig,
        "thompson_ig": select_thompson_ig,
        "trained": select_trained,
        "oracle": select_oracle,
    }

    results = {s: {turn: [] for turn in range(MAX_TURNS + 1)} for s in strategies}
    paired_recall10 = {s: [] for s in strategies}  # for paired bootstrap, only turn 5

    for pi, persona in enumerate(test_personas):
        eligible_set = set(gt.get(persona["persona_id"], []))
        for sname in strategies:
            mask_v = np.zeros(len(ATTRS))
            revealed = set()
            # Per-strategy independent RNG seeded per (strategy, persona) for reproducibility
            rng_local = np.random.default_rng(SEED + pi * 100 + hash(sname) % 1000)
            posterior = {a: [1.0, 1.0] for a in ATTRS}  # Beta(α, β) prior
            ig_norm_max = 1e-9  # running max for IG normalisation per persona

            for turn in range(MAX_TURNS + 1):
                ret = retrieve_topk(pi, mask_v, max(KS))
                rec = {f"recall@{k}": recall(ret, eligible_set, k) for k in KS}
                results[sname][turn].append(rec)
                if turn == MAX_TURNS:
                    paired_recall10[sname].append(rec["recall@10"])
                    break
                next_attr = selectors[sname](
                    persona, revealed,
                    rng_local=rng_local, pi=pi, mask=mask_v, posterior=posterior,
                )
                if next_attr is None or next_attr in revealed:
                    # Lock state — fill remaining turns with the same metric
                    for t_remain in range(turn + 1, MAX_TURNS + 1):
                        ret_r = retrieve_topk(pi, mask_v, max(KS))
                        rec_r = {f"recall@{k}": recall(ret_r, eligible_set, k) for k in KS}
                        results[sname][t_remain].append(rec_r)
                        if t_remain == MAX_TURNS:
                            paired_recall10[sname].append(rec_r["recall@10"])
                    break

                # For Thompson: observe reward = normalised IG of the chosen attr, update posterior
                if sname == "thompson_ig":
                    ig_full = compute_ig_table(pi, mask_v, revealed)
                    # Normalise by running max IG observed for this persona
                    if ig_full:
                        cur_max = max(ig_full.values())
                        if cur_max > ig_norm_max:
                            ig_norm_max = cur_max
                    raw_ig = ig_full.get(next_attr, 0.0)
                    r_norm = max(0.0, min(1.0, raw_ig / ig_norm_max if ig_norm_max > 1e-9 else 0.0))
                    posterior[next_attr][0] += r_norm
                    posterior[next_attr][1] += (1.0 - r_norm)

                revealed.add(next_attr)
                mask_v[ATTRS.index(next_attr)] = 1

    # Aggregate
    summary = {
        "n_test_personas": len(test_personas),
        "epsilon": EPSILON,
        "seed": SEED,
        "max_turns": MAX_TURNS,
    }
    for sname in strategies:
        per_turn = {}
        for turn in range(MAX_TURNS + 1):
            recs = results[sname][turn]
            for k in KS:
                vals = [r[f"recall@{k}"] for r in recs]
                per_turn[f"turn{turn}_recall@{k}"] = round(float(np.mean(vals)), 4)
        # Bootstrap CI for turn-5 recall@10
        r10_t5 = [r["recall@10"] for r in results[sname][MAX_TURNS]]
        mean5, lo5, hi5 = bootstrap_ci(r10_t5)
        per_turn["turn5_recall@10_mean"] = round(mean5, 4)
        per_turn["turn5_recall@10_ci95"] = [round(lo5, 4), round(hi5, 4)]
        per_turn["turn5_recall@10_per_persona"] = r10_t5
        summary[sname] = per_turn

    # Pairwise paired bootstrap: trained vs each other, at turn 5
    summary["paired_diff_at_turn5_vs_trained"] = {}
    base = paired_recall10["trained"]
    for sname in strategies:
        if sname == "trained":
            continue
        diff_mean, lo, hi = paired_bootstrap_ci(base, paired_recall10[sname])
        summary["paired_diff_at_turn5_vs_trained"][sname] = {
            "trained_minus": round(diff_mean, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        }

    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Recall@10 per turn ===")
    print(f"{'turn':<5s}" + "".join(f"{s:>20s}" for s in strategies))
    for turn in range(MAX_TURNS + 1):
        row = "".join(f"{summary[s][f'turn{turn}_recall@10']:>20.4f}" for s in strategies)
        print(f"{turn:<5d}{row}")

    print(f"\n=== Turn-5 Recall@10 with 95% bootstrap CI ===")
    for sname in strategies:
        row = summary[sname]
        print(f"  {sname:<22s} mean={row['turn5_recall@10_mean']:.4f} "
              f"CI=[{row['turn5_recall@10_ci95'][0]:.4f}, {row['turn5_recall@10_ci95'][1]:.4f}]")

    print(f"\n=== Paired diff vs trained at turn 5 (positive = trained wins) ===")
    for sname, d in summary["paired_diff_at_turn5_vs_trained"].items():
        print(f"  trained - {sname:<22s} = {d['trained_minus']:+.4f} CI=[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")

    elapsed = time.time() - t_start
    print(f"\nTotal compute time: {elapsed:.1f}s")
    print(f"\n저장: {OUT_PATH}")

    # 1-line summary
    print(
        f"\nRecall@10 turn5: trained={summary['trained']['turn5_recall@10']:.3f}, "
        f"ε-greedy(IG)={summary['epsilon_greedy_ig']['turn5_recall@10']:.3f}, "
        f"Thompson(IG)={summary['thompson_ig']['turn5_recall@10']:.3f}, "
        f"heuristic={summary['heuristic']['turn5_recall@10']:.3f}, "
        f"oracle={summary['oracle']['turn5_recall@10']:.3f}"
    )


if __name__ == "__main__":
    main()
