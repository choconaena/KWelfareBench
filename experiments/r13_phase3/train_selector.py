"""Question Selector ML 학습 — supervised classifier.

입력: persona feature vector + revealed state (10-bit)
출력: 다음에 elicit할 attribute (10-class)

학습 데이터: training_data.json (greedy oracle로 생성)
모델: scikit-learn RandomForestClassifier
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
TRAIN_PATH = REPO / "experiments/r13_phase3/training_data.json"
MODEL_PATH = REPO / "experiments/r13_phase3/selector_model.pkl"
PERSONAS_PATH = REPO / "docs/papers/ai4good/data/personas_v2.json"

ATTRS = [
    "special_targets", "age", "sido", "household_types", "income_level",
    "disability", "education", "employment", "gender", "sigungu",
]
ATTR2IDX = {a: i for i, a in enumerate(ATTRS)}

GROUPS = ["disabled", "single_parent", "senior", "youth", "general", "intersectional"]


def persona_features(persona: dict) -> list[float]:
    """페르소나 → numerical feature vector (모델 입력용)."""
    f = []
    # group one-hot
    for g in GROUPS:
        f.append(1.0 if persona.get("group") == g else 0.0)
    # age (normalized)
    age = persona.get("age", 0) or 0
    f.append(age / 100.0)
    # gender
    g = persona.get("gender", "")
    f.append(1.0 if g == "여성" else 0.0)
    f.append(1.0 if g == "남성" else 0.0)
    # disability
    f.append(1.0 if persona.get("disability") == "있음" else 0.0)
    # income_level: basic / secondary / etc
    il = persona.get("income_level", "")
    f.append(1.0 if "기초" in il or "수급" in il else 0.0)
    f.append(1.0 if "차상위" in il else 0.0)
    # household
    hh = persona.get("household_types") or []
    for kw in ["1인가구", "한부모", "다자녀", "신혼부부", "다문화", "조손"]:
        f.append(1.0 if any(kw in h for h in hh) else 0.0)
    # special
    sp = persona.get("special_targets") or []
    f.append(1.0 if sp else 0.0)
    return f


def revealed_features(revealed: list[str]) -> list[float]:
    """revealed set → 10-bit one-hot."""
    rev = set(revealed)
    return [1.0 if a in rev else 0.0 for a in ATTRS]


def make_X_y(records, personas):
    persona_dict = {p["persona_id"]: p for p in personas}
    X = []
    y = []
    for r in records:
        p = persona_dict[r["persona_id"]]
        feat = persona_features(p) + revealed_features(r["revealed"])
        X.append(feat)
        y.append(ATTR2IDX[r["next_best_attr"]])
    return np.array(X), np.array(y)


def main():
    with open(TRAIN_PATH) as f:
        records = json.load(f)
    with open(PERSONAS_PATH) as f:
        personas = json.load(f)
    print(f"records: {len(records)}, personas: {len(personas)}")

    X, y = make_X_y(records, personas)
    print(f"X shape: {X.shape}, y shape: {y.shape}, classes: {len(np.unique(y))}")

    # 80/20 split (페르소나 단위로 split하면 leak 없음)
    persona_ids = list({r["persona_id"] for r in records})
    train_pids, test_pids = train_test_split(persona_ids, test_size=0.2, random_state=42)
    train_pids_set, test_pids_set = set(train_pids), set(test_pids)

    train_idx = [i for i, r in enumerate(records) if r["persona_id"] in train_pids_set]
    test_idx = [i for i, r in enumerate(records) if r["persona_id"] in test_pids_set]
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    print(f"train: {len(X_tr)}, test: {len(X_te)}")

    # RF classifier
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")

    # baseline: 가장 빈번한 next_best_attr (전체 train data 기준)
    from collections import Counter
    most_common = Counter(y_tr.tolist()).most_common(1)[0][0]
    baseline_acc = (y_te == most_common).mean()
    print(f"Majority baseline acc: {baseline_acc:.4f}")
    print(f"\nClassification report:")
    target_names = [ATTRS[i] for i in sorted(np.unique(y_te))]
    print(classification_report(y_te, y_pred, target_names=target_names, zero_division=0))

    # 저장
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "ATTRS": ATTRS, "GROUPS": GROUPS}, f)
    print(f"\n저장: {MODEL_PATH}")
    print(f"  test acc: {acc:.4f}, majority baseline: {baseline_acc:.4f}, lift: {(acc-baseline_acc)*100:.1f}pp")


if __name__ == "__main__":
    main()
