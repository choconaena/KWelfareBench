"""Question selector ML 학습 데이터 생성.

각 (persona, current_revealed_set) state에서 다음에 어느 attribute을 elicit하면
recall@10이 가장 많이 증가하는지 측정 → supervised 학습 데이터.

전략:
  - 페르소나 180명 × 가능한 revealed_set 일부 sample (경로 대신)
  - turn-by-turn 점진적 추가 (greedy oracle):
    turn t에서 모든 가능한 next attribute을 시도 → 가장 큰 delta 선택 → 학습 record
  - 결과: (persona_features, revealed_state) → next_attribute_id 분류 데이터

attribute set:
  ATTRS = ["special_targets", "age", "sido", "household_types", "income_level",
           "disability", "education", "employment", "gender", "sigungu"]
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eval"))
sys.path.insert(0, str(REPO / "experiments/r13_phase2"))
sys.path.insert(0, str(REPO / "experiments/r13_phase3"))

from baselines.our_hybrid_retriever import OurHybridRetriever  # noqa: E402
from baselines.base import policy_text  # noqa: E402
from soft_eligibility import compute_marginals, soft_score  # noqa: E402

POLICIES_PATH = REPO / "data/processed/policies.json"
PERSONAS_PATH = REPO / "docs/papers/ai4good/data/personas_v2.json"
GT_PATH = REPO / "docs/papers/ai4good/data/ground_truth_v3.json"
LABELS_PATH = REPO / "experiments/r13_phase1/llm_labeling_full/labels.json"
OUT_PATH = REPO / "experiments/r13_phase3/training_data.json"
EMB_CACHE = REPO / "experiments/r13_phase3/policy_emb.npy"
EMB_IDS = REPO / "experiments/r13_phase3/policy_ids.json"

ATTRS = [
    "special_targets", "age", "sido", "household_types", "income_level",
    "disability", "education", "employment", "gender", "sigungu",
]
MAX_TURNS = 5
TOP_K = 10


def is_bokjiro(p):
    return "복지로" in p.get("source", "") or "bokjiro" in p.get("url", "")


def build_partial_persona(full: dict, revealed: set) -> dict:
    partial = {
        "persona_id": full["persona_id"],
        "group": full.get("group"),
        "query": full.get("query", ""),
        "age": None,
        "gender": "",
        "sido": None,
        "sigungu": None,
        "income_level": "",
        "income_detail": [],
        "disability": "없음",
        "household_types": [],
        "employment": "",
        "special_targets": [],
        "education": None,
    }
    for attr in revealed:
        partial[attr] = full.get(attr)
        if attr == "income_level":
            partial["income_detail"] = full.get("income_detail", [])
    return partial


def soft_retrieve(persona, marginals, revealed, labels, policy_region, policy_emb, policy_ids, model, top_k):
    """soft eligibility + dense rerank로 top_k retrieval."""
    # 1) soft score
    scores = []
    for pid, lab in labels.items():
        s = soft_score(persona, lab, marginals, revealed, policy_region.get(pid))
        scores.append((pid, s))
    # 상위 candidate 100개
    scores.sort(key=lambda x: -x[1])
    cand = [pid for pid, _ in scores[:max(top_k * 5, 100)]]
    if not cand:
        return []
    # 2) dense rerank — query 기준
    from baselines.base import persona_query
    q = persona_query(persona)
    q_emb = model.encode([q], convert_to_numpy=True)[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    id2idx = {pid: i for i, pid in enumerate(policy_ids)}
    reranked = sorted(cand, key=lambda pid: -float(policy_emb[id2idx[pid]] @ q_emb))
    return reranked[:top_k]


def recall_at_k(retrieved, eligible_set, k):
    if not eligible_set:
        return 0.0
    return len(set(retrieved[:k]) & eligible_set) / len(eligible_set)


def main():
    random.seed(42)
    np.random.seed(42)

    print("Loading data...")
    with open(POLICIES_PATH) as f:
        policies = [p for p in json.load(f) if is_bokjiro(p)]
    with open(PERSONAS_PATH) as f:
        personas = json.load(f)
    with open(GT_PATH) as f:
        gt = json.load(f)
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    policy_region = {p["policy_id"]: p.get("region", {}) for p in policies}
    print(f"  policies: {len(policies)}, personas: {len(personas)}, labels: {len(labels)}")

    print("Computing marginals...")
    marginals = compute_marginals(personas)
    print(f"  marginals: {len(marginals)} tags")

    print("Encoding policy embeddings (or load cache)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    if EMB_CACHE.exists() and EMB_IDS.exists():
        policy_emb = np.load(EMB_CACHE)
        with open(EMB_IDS) as f:
            policy_ids = json.load(f)
        print(f"  loaded cache: {policy_emb.shape}")
    else:
        policy_ids = [p["policy_id"] for p in policies]
        texts = [policy_text(p) for p in policies]
        policy_emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        norms = np.linalg.norm(policy_emb, axis=1, keepdims=True) + 1e-9
        policy_emb = policy_emb / norms
        np.save(EMB_CACHE, policy_emb)
        with open(EMB_IDS, "w") as f:
            json.dump(policy_ids, f)
        print(f"  encoded: {policy_emb.shape}")

    print(f"\nGenerating training data — greedy oracle next-attribute selection...")
    records = []  # 각 record: {persona_features, revealed_state, next_best_attr, recall_after}

    t0 = time.time()
    for pi, persona in enumerate(personas):
        eligible_set = set(gt.get(persona["persona_id"], []))
        revealed = set()

        for turn in range(MAX_TURNS):
            # 현재 recall
            cur_retrieved = soft_retrieve(
                build_partial_persona(persona, revealed), marginals, revealed,
                labels, policy_region, policy_emb, policy_ids, model, TOP_K
            )
            cur_recall = recall_at_k(cur_retrieved, eligible_set, TOP_K)

            # 각 candidate next attr 시도
            candidates = [a for a in ATTRS if a not in revealed]
            if not candidates:
                break
            best_attr = None
            best_recall = cur_recall
            attr_recalls = {}
            for attr in candidates:
                trial_revealed = revealed | {attr}
                trial_partial = build_partial_persona(persona, trial_revealed)
                trial_retrieved = soft_retrieve(
                    trial_partial, marginals, trial_revealed,
                    labels, policy_region, policy_emb, policy_ids, model, TOP_K
                )
                trial_recall = recall_at_k(trial_retrieved, eligible_set, TOP_K)
                attr_recalls[attr] = trial_recall
                if trial_recall > best_recall:
                    best_recall = trial_recall
                    best_attr = attr

            if best_attr is None:
                # 아무것도 개선 안 되면, 가장 큰 trial_recall 선택 (greedy)
                best_attr = max(attr_recalls, key=attr_recalls.get)
                best_recall = attr_recalls[best_attr]

            records.append({
                "persona_id": persona["persona_id"],
                "group": persona.get("group"),
                "turn": turn,
                "revealed": sorted(revealed),
                "candidates": candidates,
                "next_best_attr": best_attr,
                "recall_before": cur_recall,
                "recall_after": best_recall,
                "delta": best_recall - cur_recall,
                "attr_recalls": attr_recalls,
            })
            revealed.add(best_attr)

        if (pi + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{pi+1}/{len(personas)}] {elapsed:.0f}s elapsed, ~{elapsed/(pi+1)*len(personas):.0f}s total est.", flush=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT_PATH} ({len(records)} records)")


if __name__ == "__main__":
    main()
