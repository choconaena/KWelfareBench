"""GT-1 v3 (단순화): personas_v3.tag_values × labels.json 직접 비교.

Phase 3 outcome: 매핑 코드를 build_personas_v3.py로 이전 → eligible() 25줄로 축소.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "data/policy_tags_labels.json"
PERSONAS = REPO / "data/personas_v3.json"
POLICIES = REPO / "data/policies.json"
OUT_GT = REPO / "data/ground_truth_v4.json"
OUT_STATS = REPO / "data/ground_truth_v4_stats.json"

# 정책 schema의 binary requirement 카테고리
EMP_SUBS = ("unemployed", "self_employed", "startup", "agriculture_fishery",
            "small_business_sme", "public_servant_military", "industrial_accident")
HH_SUBS = ("single", "single_parent", "multi_child", "newlywed",
           "multicultural", "grandparent", "youth_minor")
EDU_SUBS = ("elementary_to_high", "university", "out_of_school", "ged")
SPECIAL_TAGS = (
    "social.veteran.war_veteran", "social.veteran.national_merit",
    "social.veteran.independence", "social.veteran.veteran_family",
    "social.immigrant.foreigner_general", "social.immigrant.north_korean_defector",
    "social.immigrant.late_arrival_child",
    "social.vulnerable.vulnerable_general", "social.vulnerable.crisis_household",
    "social.vulnerable.homeless", "social.vulnerable.solo_elderly",
    "social.vulnerable.foster_or_protected_child", "social.vulnerable.adoption",
    "social.violence_victim.violence",
    "household.pregnancy.pregnant_postpartum", "household.pregnancy.infertility",
    "household.pregnancy.unmarried_mother", "household.pregnancy.birth",
)


def eligible(pv: dict, lab: dict, pol_region: dict, ignore_region: bool = False) -> tuple[bool, str | None]:
    """단순화된 자격 판정. pv = persona.tag_values."""
    # 1. Binary "any of these required" tags: gender, disability, income, employment, household, education
    for tag, fail_dim in [
        ("personal.gender.female_only", "gender"),
        ("personal.gender.male_only", "gender"),
        ("personal.disability.required_any", "disability"),
        *[(f"personal.disability.{s}", "disability") for s in ("severe", "developmental", "visual", "hearing")],
        ("economic.income.basic_recipient", "income"),
        ("economic.income.secondary", "income"),
        ("economic.income.medical_aid", "income"),
        *[(f"economic.employment.{s}", "employment") for s in EMP_SUBS],
        *[(f"household.type.{s}", "household") for s in HH_SUBS],
        *[(f"social.education.{s}", "education") for s in EDU_SUBS],
    ]:
        if lab.get(tag) == 1 and pv.get(tag) != 1:
            return False, fail_dim

    # 2. Age numeric
    age = pv.get("personal.age.value")
    if age is not None:
        amin = lab.get("personal.age.age_min")
        amax = lab.get("personal.age.age_max")
        if (amin is not None and age < amin) or (amax is not None and age > amax):
            return False, "age"

    # 3. Income median threshold
    threshold = lab.get("economic.income.median_threshold")
    if threshold is not None and pv.get("economic.income.median_pct_estimate", 999) > threshold:
        return False, "income"

    # 4. Special tags (positive matching: persona must have at least one of policy's required specials)
    pol_specials = {tag for tag in SPECIAL_TAGS if lab.get(tag) == 1}
    if pol_specials and not any(pv.get(t) == 1 for t in pol_specials):
        return False, "special"

    # 5. Region
    if not ignore_region:
        p_sido = pv.get("region.sido")
        p_sigungu = pv.get("region.sigungu")
        pol_level = pol_region.get("level")
        if pol_level == "시도" and pol_region.get("sido") and pol_region.get("sido") != p_sido:
            return False, "sido"
        if pol_level == "시군구":
            if pol_region.get("sido") and pol_region.get("sido") != p_sido:
                return False, "sido"
            if pol_region.get("sigungu") and pol_region.get("sigungu") != p_sigungu:
                return False, "sigungu"

    return True, None


def main():
    labels = json.load(open(LABELS))
    personas = json.load(open(PERSONAS))
    all_policies = {p["policy_id"]: p for p in json.load(open(POLICIES))}
    pol_region = {pid: all_policies[pid].get("region", {}) for pid in labels if pid in all_policies}

    print(f"Personas: {len(personas)}, Policies: {len(labels)}")
    print(f"Total cells: {len(personas) * len(labels):,}\n")

    for ignore_region in [False, True]:
        gt = {}
        fail_counter = Counter()
        n_eligible = []
        for persona in personas:
            pv = persona["tag_values"]
            pid_list = []
            for pid, lab in labels.items():
                ok, fdim = eligible(pv, lab, pol_region.get(pid, {}), ignore_region=ignore_region)
                if ok:
                    pid_list.append(pid)
                elif fdim:
                    fail_counter[fdim] += 1
            gt[persona["persona_id"]] = pid_list
            n_eligible.append(len(pid_list))

        # Stats
        policy_count = Counter()
        for lst in gt.values():
            for pid in lst:
                policy_count[pid] += 1
        n_dead = sum(1 for pid in labels if policy_count.get(pid, 0) == 0)

        suffix = "_no_region" if ignore_region else ""
        out_gt = REPO / f"data/ground_truth_v4{suffix}.json"
        out_stats = REPO / f"data/ground_truth_v4_stats{suffix}.json"

        stats = {
            "ignore_region": ignore_region,
            "n_personas": len(personas),
            "n_policies": len(labels),
            "n_eligible_per_persona": {
                "mean": round(statistics.mean(n_eligible), 1),
                "median": int(statistics.median(n_eligible)),
                "min": min(n_eligible), "max": max(n_eligible),
            },
            "n_dead_policies": n_dead,
            "pct_dead": round(n_dead / len(labels) * 100, 2),
            "fail_dimension": dict(fail_counter),
        }

        json.dump(gt, open(out_gt, "w", encoding="utf-8"), ensure_ascii=False)
        json.dump(stats, open(out_stats, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"=== ignore_region={ignore_region} ===")
        print(f"  mean eligible/persona: {stats['n_eligible_per_persona']['mean']}")
        print(f"  median: {stats['n_eligible_per_persona']['median']}, range: {stats['n_eligible_per_persona']['min']}-{stats['n_eligible_per_persona']['max']}")
        print(f"  dead policies: {n_dead}/{len(labels)} ({stats['pct_dead']}%)")
        print(f"  saved: {out_gt.name}, {out_stats.name}\n")


if __name__ == "__main__":
    main()
