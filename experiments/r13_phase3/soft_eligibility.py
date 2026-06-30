"""Soft eligibility scoring — 부분 정보 환경에서의 자격 확률 계산.

설계:
  - 페르소나의 attribute가 알려진 경우: hard match
  - 알려지지 않은 (None/empty) 경우: marginal P(satisfy tag) 사용
  - 정책별 score = sum over tags of: log P(persona satisfies tag | known/marginal)
                  + log P(region match | known/marginal)
  - product 대신 log-sum (numerical stability)

Marginal P는 180 페르소나 전체에서 추정 (empirical prior).
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eval"))

from compute_ground_truth_v3 import (  # noqa: E402
    HH_TAG_MAP,
    SPECIAL_TAG_MAP,
    persona_education_tag,
    persona_employment_tag,
    persona_household_tags,
    persona_required_special_tags,
)


def compute_marginals(personas: list[dict]) -> dict:
    """180 페르소나에서 각 tag-condition 충족 확률 추정 (empirical prior)."""
    n = len(personas)
    marginal = {}

    # personal.gender
    g_counts = Counter(p.get("gender", "") for p in personas)
    marginal["personal.gender.female_only"] = g_counts.get("여성", 0) / n
    marginal["personal.gender.male_only"] = g_counts.get("남성", 0) / n

    # disability
    d_count = sum(1 for p in personas if p.get("disability") == "있음")
    for sub in ("required_any", "severe", "developmental", "visual", "hearing"):
        marginal[f"personal.disability.{sub}"] = d_count / n

    # income
    inc_basic = sum(
        1 for p in personas
        if "기초" in p.get("income_level", "") or "수급" in p.get("income_level", "")
        or any("기초" in d or "수급" in d for d in p.get("income_detail", []) or [])
    )
    inc_secondary = sum(
        1 for p in personas
        if "차상위" in p.get("income_level", "")
        or any("차상위" in d for d in p.get("income_detail", []) or [])
    )
    marginal["economic.income.basic_recipient"] = inc_basic / n
    marginal["economic.income.secondary"] = (inc_basic + inc_secondary) / n
    marginal["economic.income.medical_aid"] = sum(
        1 for p in personas if any("의료급여" in d for d in p.get("income_detail", []) or [])
    ) / n

    # employment
    emp_marginal = Counter()
    for p in personas:
        et = persona_employment_tag(p)
        if et:
            emp_marginal[et] += 1
    for tag in ("unemployed", "self_employed", "startup", "agriculture_fishery",
                 "small_business_sme", "public_servant_military", "industrial_accident"):
        full = f"economic.employment.{tag}"
        marginal[full] = emp_marginal.get(full, 0) / n

    # household
    hh_marginal = Counter()
    for p in personas:
        for t in persona_household_tags(p):
            hh_marginal[t] += 1
    for tag in ("single", "single_parent", "multi_child", "newlywed",
                 "multicultural", "grandparent", "youth_minor"):
        full = f"household.type.{tag}"
        marginal[full] = hh_marginal.get(full, 0) / n

    # education
    edu_marginal = Counter()
    for p in personas:
        e = persona_education_tag(p)
        if e:
            edu_marginal[e] += 1
    for tag in ("elementary_to_high", "university", "out_of_school", "ged"):
        full = f"social.education.{tag}"
        marginal[full] = edu_marginal.get(full, 0) / n

    # special_targets — 페르소나가 충족하는 비율
    spec_counter = Counter()
    for p in personas:
        for t in persona_required_special_tags(p):
            spec_counter[t] += 1
    for tag in SPECIAL_TAG_MAP.values():
        marginal[tag] = spec_counter.get(tag, 0) / n

    return marginal


def soft_satisfy_prob(persona: dict, lab_value: int, tag: str, marginals: dict, attr_known: bool) -> float:
    """페르소나가 정책의 tag 조건을 충족할 확률.

    lab_value: 정책의 tag 값 (-1, 0, 1)
    attr_known: 페르소나가 이 attribute을 노출했는지
    """
    if lab_value == 0:
        return 1.0  # 정책이 무관 → 항상 통과

    if attr_known:
        # hard match
        # gender
        if tag == "personal.gender.female_only":
            return 1.0 if persona.get("gender") == "여성" else 0.0
        if tag == "personal.gender.male_only":
            return 1.0 if persona.get("gender") == "남성" else 0.0
        if tag.startswith("personal.disability"):
            return 1.0 if persona.get("disability") == "있음" else 0.0
        if tag == "economic.income.basic_recipient":
            il = persona.get("income_level", "")
            return 1.0 if "기초" in il or "수급" in il else 0.0
        if tag == "economic.income.secondary":
            il = persona.get("income_level", "")
            return 1.0 if "기초" in il or "수급" in il or "차상위" in il else 0.0
        if tag == "economic.income.medical_aid":
            return 1.0 if any("의료급여" in d for d in persona.get("income_detail", []) or []) else 0.0
        if tag.startswith("economic.employment"):
            return 1.0 if persona_employment_tag(persona) == tag else 0.0
        if tag.startswith("household.type"):
            return 1.0 if tag in persona_household_tags(persona) else 0.0
        if tag.startswith("social.education"):
            return 1.0 if persona_education_tag(persona) == tag else 0.0
        if tag in SPECIAL_TAG_MAP.values():
            return 1.0 if tag in persona_required_special_tags(persona) else 0.0
        return 0.5  # unknown
    else:
        # marginal probability
        return marginals.get(tag, 0.5)


def soft_score(persona: dict, lab: dict, marginals: dict, revealed_attrs: set,
                pol_region: dict | None = None) -> float:
    """페르소나 × 정책 soft eligibility log-score (높을수록 자격 확률 높음).

    revealed_attrs: 페르소나의 노출된 attribute names (예: {"age", "gender"})
    """
    # tag-attribute 매핑 (어떤 tag이 어떤 페르소나 attribute에 의존하는지)
    tag_attr_map = {
        "personal.gender": "gender",
        "personal.disability": "disability",
        "economic.income": "income_level",
        "economic.employment": "employment",
        "household.type": "household_types",
        "household.pregnancy": "special_targets",
        "household.birth_order": "special_targets",
        "social.veteran": "special_targets",
        "social.immigrant": "special_targets",
        "social.vulnerable": "special_targets",
        "social.violence_victim": "special_targets",
        "social.education": "education",
    }

    log_score = 0.0
    for tag, val in lab.items():
        if tag.startswith("_") or tag.startswith("policy.category") or tag.startswith("personal.age") or tag.startswith("social.residence") or tag == "economic.income.median_threshold":
            continue
        if val == 0:
            continue  # 무관

        # 어떤 attribute에 매핑되는지
        attr = None
        for prefix, a in tag_attr_map.items():
            if tag.startswith(prefix):
                attr = a
                break

        attr_known = attr in revealed_attrs if attr else False
        p = soft_satisfy_prob(persona, val, tag, marginals, attr_known)
        # 자격 박탈 (-1) 처리: 페르소나가 그 조건이면 자격 X → P(자격 박탈 회피) = 1 - P(satisfy)
        if val == -1:
            p = 1.0 - p
        # log-score, 0 방지
        log_score += math.log(max(p, 1e-6))

    # age numeric
    if "age" in revealed_attrs and persona.get("age") is not None:
        amin = lab.get("personal.age.age_min")
        amax = lab.get("personal.age.age_max")
        age = persona["age"]
        if amin is not None and age < amin:
            log_score += math.log(1e-6)
        if amax is not None and age > amax:
            log_score += math.log(1e-6)

    # region
    if pol_region:
        pol_level = pol_region.get("level")
        pol_sido = pol_region.get("sido")
        pol_sigungu = pol_region.get("sigungu")
        if "sido" in revealed_attrs and persona.get("sido"):
            if pol_level == "시도" and pol_sido and pol_sido != persona["sido"]:
                log_score += math.log(1e-6)
            if pol_level == "시군구" and pol_sido and pol_sido != persona["sido"]:
                log_score += math.log(1e-6)
        if "sigungu" in revealed_attrs and persona.get("sigungu"):
            if pol_level == "시군구" and pol_sigungu and pol_sigungu != persona["sigungu"]:
                log_score += math.log(1e-6)

    return log_score
