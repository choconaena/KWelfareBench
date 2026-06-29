"""personas_v2.json → personas_v3.json: tag_values 필드 추가 (정책 schema와 통일).

Persona의 한국어 attributes → 69 binary + 3 numeric + region NL 필드로 사전 인코딩.
GT-1 함수 단순화 (100줄 → 25줄).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PERSONAS_V2 = REPO / "data/personas_v2.json"
PERSONAS_V3 = REPO / "data/personas_v3.json"
REGIONS = REPO / "regions.json"

# 한국어 → schema tag 매핑 (compute_ground_truth_v3.py에서 가져옴)
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

HH_TAG_MAP = {
    "1인가구": "household.type.single",
    "한부모": "household.type.single_parent",
    "다자녀": "household.type.multi_child",
    "신혼부부": "household.type.newlywed",
    "다문화": "household.type.multicultural",
    "조손": "household.type.grandparent",
    "소년소녀": "household.type.youth_minor",
}

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

EDU_TAG_MAP = {
    "초등학생": "social.education.elementary_to_high",
    "중학생": "social.education.elementary_to_high",
    "고등학생": "social.education.elementary_to_high",
    "대학생": "social.education.university",
    "대학원생": "social.education.university",
    "학교밖": "social.education.out_of_school",
    "검정고시": "social.education.ged",
}


def map_first(text, mapping):
    """text가 mapping의 어느 key를 포함하면 해당 tag 반환. 없으면 None."""
    if not text:
        return None
    for k, tag in mapping.items():
        if k in text:
            return tag
    return None


def map_all(items, mapping):
    """list of items → set of matched tags."""
    tags = set()
    for item in (items or []):
        for k, tag in mapping.items():
            if k in item:
                tags.add(tag)
    return tags


def estimate_income_pct_median(income_level: str, income_detail: list) -> int:
    """중위소득 % 추정. 기초=30, 차상위=50, 저소득=80, 일반=120."""
    detail_set = set(income_detail or [])
    is_basic = "기초" in income_level or "수급" in income_level or any(
        "기초" in d or "수급" in d for d in detail_set
    )
    is_secondary = "차상위" in income_level or any("차상위" in d for d in detail_set)
    if is_basic:
        return 30
    if is_secondary:
        return 50
    if "저소득" in income_level:
        return 80
    return 120


def build_tag_values(persona: dict) -> dict:
    """Persona 한국어 fields → 69 binary + 3 numeric + region 통합 tag_values 필드."""
    tv = {}

    # === Personal: gender ===
    gender = persona.get("gender", "")
    tv["personal.gender.female_only"] = 1 if gender == "여성" else 0
    tv["personal.gender.male_only"] = 1 if gender == "남성" else 0

    # === Personal: disability ===
    has_dis = persona.get("disability") == "있음"
    tv["personal.disability.required_any"] = 1 if has_dis else 0
    # severe/dev/visual/hearing — persona detail 없음 → has_dis로 weak coverage
    for sub in ("severe", "developmental", "visual", "hearing"):
        tv[f"personal.disability.{sub}"] = 1 if has_dis else 0

    # === Personal: health ===
    # personas_v2에는 health 세부 없음 → 모두 0 (보수적)
    for sub in ("chronic_or_severe", "rare_or_cancer", "mental_health",
                "dementia", "diabetes", "hypertension", "depression"):
        tv[f"personal.health.{sub}"] = 0

    # === Personal: age (numeric) ===
    tv["personal.age.value"] = persona.get("age")  # 페르소나의 실제 나이

    # === Economic: income ===
    income_level = persona.get("income_level", "")
    income_detail = persona.get("income_detail", []) or []
    detail_set = set(income_detail)
    is_basic = "기초" in income_level or "수급" in income_level or any(
        "기초" in d or "수급" in d for d in detail_set
    )
    is_secondary = "차상위" in income_level or any("차상위" in d for d in detail_set)
    is_medical = any("의료급여" in d for d in detail_set)
    tv["economic.income.basic_recipient"] = 1 if is_basic else 0
    tv["economic.income.secondary"] = 1 if (is_basic or is_secondary) else 0
    tv["economic.income.medical_aid"] = 1 if is_medical else 0
    tv["economic.income.median_pct_estimate"] = estimate_income_pct_median(income_level, income_detail)

    # === Economic: employment ===
    emp_tag = map_first(persona.get("employment", ""), EMP_TAG_MAP)
    for sub in ("unemployed", "self_employed", "startup", "agriculture_fishery",
                "small_business_sme", "public_servant_military", "industrial_accident"):
        full = f"economic.employment.{sub}"
        tv[full] = 1 if emp_tag == full else 0

    # === Economic: housing ===
    # personas_v2에 housing 세부 없음 → 모두 0
    for sub in ("no_house", "jeonse", "monthly_rent", "rental"):
        tv[f"economic.housing.{sub}"] = 0

    # === Household: type ===
    hh_tags = map_all(persona.get("household_types", []), HH_TAG_MAP)
    for sub in ("single", "single_parent", "multi_child", "newlywed",
                "multicultural", "grandparent", "youth_minor"):
        full = f"household.type.{sub}"
        tv[full] = 1 if full in hh_tags else 0

    # === Household: pregnancy + birth_order ===
    special_tags = map_all(persona.get("special_targets", []), SPECIAL_TAG_MAP)
    for sub in ("pregnant_postpartum", "infertility", "unmarried_mother", "birth"):
        full = f"household.pregnancy.{sub}"
        tv[full] = 1 if full in special_tags else 0
    # birth_order: personas_v2에 없음 → 0
    for sub in ("first", "second", "third_plus"):
        tv[f"household.birth_order.{sub}"] = 0

    # === Social: veteran/immigrant/vulnerable/violence ===
    for tag in [
        "social.veteran.war_veteran", "social.veteran.national_merit",
        "social.veteran.independence", "social.veteran.veteran_family",
        "social.immigrant.foreigner_general", "social.immigrant.north_korean_defector",
        "social.immigrant.late_arrival_child",
        "social.vulnerable.vulnerable_general", "social.vulnerable.crisis_household",
        "social.vulnerable.homeless", "social.vulnerable.solo_elderly",
        "social.vulnerable.foster_or_protected_child", "social.vulnerable.adoption",
        "social.violence_victim.violence",
    ]:
        tv[tag] = 1 if tag in special_tags else 0

    # === Social: education ===
    edu_tag = map_first(persona.get("education", ""), EDU_TAG_MAP)
    for sub in ("elementary_to_high", "university", "out_of_school", "ged"):
        full = f"social.education.{sub}"
        tv[full] = 1 if edu_tag == full else 0

    # === Social: residence ===
    # 페르소나에 거주기간 정보 없음 → 모두 1 (만족 가정, 권한 검토 필요시 0)
    tv["social.residence.resident_required"] = 1
    tv["social.residence.long_term_resident"] = 1

    # === Region (NL field) ===
    sido = persona.get("sido", "")
    sigungu = persona.get("sigungu", "")
    tv["region.sido"] = sido
    tv["region.sigungu"] = sigungu
    tv["region.nl"] = f"{sido} {sigungu}".strip()

    return tv


def main():
    personas = json.load(open(PERSONAS_V2))
    print(f"Personas: {len(personas)}")

    for p in personas:
        p["tag_values"] = build_tag_values(p)

    # Verify 1st persona
    first = personas[0]
    print(f"\nFirst persona tag_values keys: {len(first['tag_values'])}")
    binary_count = sum(1 for v in first["tag_values"].values() if v in (0, 1))
    print(f"  Binary (0/1) tags: {binary_count}")
    print(f"  Numeric/string tags: {len(first['tag_values']) - binary_count}")

    # Save
    json.dump(personas, open(PERSONAS_V3, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n✓ saved {PERSONAS_V3}")

    # Save schema reference (paper §3에 박을 수 있도록)
    schema_doc = {
        "n_binary_tags": binary_count,
        "n_numeric_tags": 2,  # personal.age.value, economic.income.median_pct_estimate
        "n_region_fields": 3,  # sido, sigungu, nl
        "binary_tag_keys": [k for k, v in first["tag_values"].items() if v in (0, 1)],
        "numeric_tag_keys": [k for k, v in first["tag_values"].items() if isinstance(v, (int, float)) and v not in (0, 1)],
        "region_keys": ["region.sido", "region.sigungu", "region.nl"],
        "mapping_provenance": {
            "special_targets_map": SPECIAL_TAG_MAP,
            "household_types_map": HH_TAG_MAP,
            "employment_map": EMP_TAG_MAP,
            "education_map": EDU_TAG_MAP,
        }
    }
    schema_path = REPO / "data/persona_schema_v3.json"
    json.dump(schema_doc, open(schema_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"✓ saved {schema_path}")


if __name__ == "__main__":
    main()
