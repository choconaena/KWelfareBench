"""GT v3 — 새 labels.json (LLM-tagged 4937) × personas_v2 (180명, 옛 schema 구조).

페르소나 실제 구조:
  persona_id, group, age, gender(한국어), sido, sigungu, income_level, income_detail(list),
  disability("있음"/"없음"), household_types(list), employment(str), special_targets(list),
  education(str|null), negative_declarations(list)

→ 새 53-tag labels.json과 매핑하는 eligible() 함수
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "experiments/r13_phase1/llm_labeling_full/labels.json"
PERSONAS = REPO / "docs/papers/ai4good/data/personas_v2.json"
POLICIES_PATH = REPO / "data/processed/policies.json"
OUT_GT = REPO / "docs/papers/ai4good/data/ground_truth_v3.json"
OUT_STATS = REPO / "docs/papers/ai4good/data/ground_truth_v3_stats.json"

# special_targets 한국어 → schema tag 매핑
SPECIAL_TAG_MAP = {
    "참전유공자": "social.veteran.war_veteran",
    "국가유공자": "social.veteran.national_merit",
    "독립유공자": "social.veteran.independence",
    "유공자가족": "social.veteran.veteran_family",
    "외국인": "social.immigrant.foreigner_general",
    "결혼이민자": "social.immigrant.foreigner_general",
    "북한이탈주민": "social.immigrant.north_korean_defector",
    "탈북민": "social.immigrant.north_korean_defector",
    "중도입국자녀": "social.immigrant.late_arrival_child",
    "취약계층": "social.vulnerable.vulnerable_general",
    "위기가구": "social.vulnerable.crisis_household",
    "노숙인": "social.vulnerable.homeless",
    "독거노인": "social.vulnerable.solo_elderly",
    "보호아동": "social.vulnerable.foster_or_protected_child",
    "위탁아동": "social.vulnerable.foster_or_protected_child",
    "입양": "social.vulnerable.adoption",
    "가정폭력": "social.violence_victim.violence",
    "성폭력": "social.violence_victim.violence",
    "임산부": "household.pregnancy.pregnant_postpartum",
    "산모": "household.pregnancy.pregnant_postpartum",
    "난임": "household.pregnancy.infertility",
    "미혼모": "household.pregnancy.unmarried_mother",
    "출산": "household.pregnancy.birth",
}

# household_types 한국어 → tag
HH_TAG_MAP = {
    "1인가구": "household.type.single",
    "한부모": "household.type.single_parent",
    "다자녀": "household.type.multi_child",
    "신혼부부": "household.type.newlywed",
    "다문화": "household.type.multicultural",
    "조손": "household.type.grandparent",
    "소년소녀": "household.type.youth_minor",
}

# employment 한국어 → tag
EMP_TAG_MAP = {
    "미취업": "economic.employment.unemployed",
    "실업": "economic.employment.unemployed",
    "구직": "economic.employment.unemployed",
    "자영업": "economic.employment.self_employed",
    "창업": "economic.employment.startup",
    "농업": "economic.employment.agriculture_fishery",
    "어업": "economic.employment.agriculture_fishery",
    "농어업": "economic.employment.agriculture_fishery",
    "소상공인": "economic.employment.small_business_sme",
    "중소기업": "economic.employment.small_business_sme",
    "공무원": "economic.employment.public_servant_military",
    "군인": "economic.employment.public_servant_military",
    "산재": "economic.employment.industrial_accident",
}

# education 한국어 → tag
EDU_TAG_MAP = {
    "초등학생": "social.education.elementary_to_high",
    "중학생": "social.education.elementary_to_high",
    "고등학생": "social.education.elementary_to_high",
    "대학생": "social.education.university",
    "대학원생": "social.education.university",
    "학교밖": "social.education.out_of_school",
    "검정고시": "social.education.ged",
}


def persona_required_special_tags(persona: dict) -> set:
    """페르소나가 충족하는 special tag 집합."""
    s = set()
    for kw in persona.get("special_targets", []) or []:
        for k, tag in SPECIAL_TAG_MAP.items():
            if k in kw:
                s.add(tag)
    return s


def persona_household_tags(persona: dict) -> set:
    s = set()
    for ht in persona.get("household_types", []) or []:
        for k, tag in HH_TAG_MAP.items():
            if k in ht:
                s.add(tag)
    return s


def persona_employment_tag(persona: dict) -> str | None:
    emp = persona.get("employment", "") or ""
    for k, tag in EMP_TAG_MAP.items():
        if k in emp:
            return tag
    return None


def persona_education_tag(persona: dict) -> str | None:
    edu = persona.get("education", "") or ""
    if not edu:
        return None
    for k, tag in EDU_TAG_MAP.items():
        if k in edu:
            return tag
    return None


def eligible(persona: dict, lab: dict, pol_region: dict) -> tuple[bool, str | None]:
    # 1) gender (한국어)
    g = persona.get("gender", "")
    if lab.get("personal.gender.female_only") == 1 and g != "여성":
        return False, "gender"
    if lab.get("personal.gender.male_only") == 1 and g != "남성":
        return False, "gender"

    # 2) age
    age = persona.get("age")
    amin, amax = lab.get("personal.age.age_min"), lab.get("personal.age.age_max")
    if age is not None:
        if amin is not None and age < amin:
            return False, "age"
        if amax is not None and age > amax:
            return False, "age"

    # 3) disability
    has_dis = persona.get("disability") == "있음"
    if lab.get("personal.disability.required_any") == 1 and not has_dis:
        return False, "disability"
    # 세부 (severe/dev/visual/hearing) — 페르소나 detail 없으면 has_dis로 약하게 통과
    for sub in ("severe", "developmental", "visual", "hearing"):
        if lab.get(f"personal.disability.{sub}") == 1 and not has_dis:
            return False, "disability"

    # 4) income
    income_level = persona.get("income_level", "")
    income_detail = set(persona.get("income_detail", []) or [])
    is_basic = "기초" in income_level or "수급" in income_level or any("기초" in d or "수급" in d for d in income_detail)
    is_secondary = "차상위" in income_level or any("차상위" in d for d in income_detail)
    is_medical = any("의료급여" in d for d in income_detail)

    if lab.get("economic.income.basic_recipient") == 1 and not is_basic:
        return False, "income"
    if lab.get("economic.income.secondary") == 1 and not (is_basic or is_secondary):
        return False, "income"
    if lab.get("economic.income.medical_aid") == 1 and not is_medical:
        return False, "income"
    # median_threshold: 페르소나 income_level별 추정
    threshold = lab.get("economic.income.median_threshold")
    if threshold is not None:
        # 단순 매핑: 기초수급=30, 차상위=50, 일반=120 (보수적)
        if is_basic:
            est = 30
        elif is_secondary:
            est = 50
        elif "저소득" in income_level:
            est = 80
        else:
            est = 120
        if est > threshold:
            return False, "income"

    # 5) employment
    emp_tag = persona_employment_tag(persona)
    for tag_suffix in ("unemployed", "self_employed", "startup", "agriculture_fishery",
                        "small_business_sme", "public_servant_military", "industrial_accident"):
        full_tag = f"economic.employment.{tag_suffix}"
        if lab.get(full_tag) == 1 and emp_tag != full_tag:
            return False, "employment"

    # 6) household
    hh_tags = persona_household_tags(persona)
    for tag_suffix in ("single", "single_parent", "multi_child", "newlywed",
                        "multicultural", "grandparent", "youth_minor"):
        full_tag = f"household.type.{tag_suffix}"
        if lab.get(full_tag) == 1 and full_tag not in hh_tags:
            return False, "household"

    # 7) education
    edu_tag = persona_education_tag(persona)
    for tag_suffix in ("elementary_to_high", "university", "out_of_school", "ged"):
        full_tag = f"social.education.{tag_suffix}"
        if lab.get(full_tag) == 1 and edu_tag != full_tag:
            return False, "education"

    # 8) special (positive matching)
    persona_specials = persona_required_special_tags(persona)
    pol_required_specials = set()
    for tag in list(SPECIAL_TAG_MAP.values()):
        if lab.get(tag) == 1:
            pol_required_specials.add(tag)
    if pol_required_specials and not (persona_specials & pol_required_specials):
        return False, "special"

    # 9) region
    p_sido = persona.get("sido")
    p_sigungu = persona.get("sigungu")
    pol_level = pol_region.get("level")
    pol_sido = pol_region.get("sido")
    pol_sigungu = pol_region.get("sigungu")

    if pol_level == "시도" and pol_sido and p_sido and pol_sido != p_sido:
        return False, "sido"
    if pol_level == "시군구":
        if pol_sido and p_sido and pol_sido != p_sido:
            return False, "sido"
        if pol_sigungu and p_sigungu and pol_sigungu != p_sigungu:
            return False, "sigungu"

    return True, None


def main():
    print("=" * 60)
    print("GT v3 재계산 — labels.json × personas_v2.json")
    print("=" * 60)

    with open(LABELS) as f:
        labels = json.load(f)
    with open(PERSONAS) as f:
        personas = json.load(f)
    with open(POLICIES_PATH) as f:
        all_policies = {p["policy_id"]: p for p in json.load(f)}

    print(f"\n정책: {len(labels)}, 페르소나: {len(personas)}")
    print(f"총 cell 수: {len(labels) * len(personas):,}\n")

    # 정책별 region 미리 추출
    policy_region = {pid: all_policies[pid].get("region", {}) for pid in labels if pid in all_policies}

    # GT 계산
    gt = {}
    fail_counter = Counter()
    n_eligible_per_persona = []

    for persona in personas:
        pid_list = []
        for pid, lab in labels.items():
            ok, fail_dim = eligible(persona, lab, policy_region.get(pid, {}))
            if ok:
                pid_list.append(pid)
            elif fail_dim:
                fail_counter[fail_dim] += 1
        gt[persona["persona_id"]] = pid_list
        n_eligible_per_persona.append(len(pid_list))

    # 통계
    stats = {
        "total_personas": len(personas),
        "total_policies": len(labels),
        "data_source": f"Bokjiro {len(labels)} (sync labeling 2026-04, 53-tag schema)",
        "n_eligible_distribution": {
            "min": min(n_eligible_per_persona),
            "max": max(n_eligible_per_persona),
            "mean": statistics.mean(n_eligible_per_persona),
            "median": statistics.median(n_eligible_per_persona),
            "stdev": statistics.stdev(n_eligible_per_persona) if len(n_eligible_per_persona) > 1 else 0,
        },
        "failure_dimension_counter": dict(fail_counter),
    }

    # 그룹별
    by_group = defaultdict(list)
    for p, n in zip(personas, n_eligible_per_persona):
        by_group[p.get("group", "unknown")].append(n)
    stats["per_group_mean_eligible"] = {g: statistics.mean(v) for g, v in by_group.items()}
    stats["per_group_n_personas"] = {g: len(v) for g, v in by_group.items()}

    # 정책별
    policy_eligible_count = Counter()
    for pid_list in gt.values():
        for pid in pid_list:
            policy_eligible_count[pid] += 1
    n_dead = sum(1 for pid in labels if policy_eligible_count.get(pid, 0) == 0)
    stats["dead_policies"] = {
        "n_dead": n_dead,
        "pct_dead": round(n_dead / len(labels) * 100, 2),
    }
    stats["policy_eligible_distribution"] = {
        "max": max(policy_eligible_count.values()) if policy_eligible_count else 0,
        "mean": round(statistics.mean(policy_eligible_count.values()), 2) if policy_eligible_count else 0,
        "median": int(statistics.median(policy_eligible_count.values())) if policy_eligible_count else 0,
    }

    # 저장
    with open(OUT_GT, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False)
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=== 통계 요약 ===")
    print(f"페르소나당 자격 정책: mean={stats['n_eligible_distribution']['mean']:.1f}, median={stats['n_eligible_distribution']['median']:.0f}, range={stats['n_eligible_distribution']['min']}~{stats['n_eligible_distribution']['max']}")
    print(f"\n실패 dimension top 5:")
    for k, v in sorted(fail_counter.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k:>15s}: {v:>10,} ({v/(len(labels)*len(personas))*100:.1f}%)")
    print(f"\n그룹별 평균 자격 정책 수:")
    for g, m in sorted(stats["per_group_mean_eligible"].items(), key=lambda x: -x[1]):
        print(f"  {g:>15s} (n={stats['per_group_n_personas'][g]}): {m:.1f}")
    print(f"\nDead policies (자격자 0명): {n_dead}/{len(labels)} ({n_dead/len(labels)*100:.1f}%)")
    print(f"\n저장: {OUT_GT}")
    print(f"저장: {OUT_STATS}")


if __name__ == "__main__":
    main()
