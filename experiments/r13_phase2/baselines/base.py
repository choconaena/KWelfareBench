"""BaseRetriever 추상 인터페이스 + 공통 텍스트 유틸."""
from __future__ import annotations

from abc import ABC, abstractmethod


def policy_text(p: dict) -> str:
    """정책 → 검색용 텍스트 (이름 + 요약 + 자격 + 혜택)."""
    return " ".join(
        filter(
            None,
            [
                p.get("name", ""),
                p.get("summary", ""),
                p.get("description", ""),
                p.get("eligibility", ""),
                p.get("benefits", ""),
            ],
        )
    )


def persona_query(persona: dict) -> str:
    """페르소나 → 검색 query (자유어 + 속성 키워드)."""
    parts = [persona.get("query", "")]
    if persona.get("disability") == "있음":
        parts.append("장애인")
    income = persona.get("income_level", "")
    if "기초" in income or "수급" in income:
        parts.append("기초생활수급자")
    if "차상위" in income:
        parts.append("차상위계층")
    if persona.get("age") and persona["age"] >= 65:
        parts.append("노인")
    elif persona.get("age") and persona["age"] <= 39:
        parts.append("청년")
    parts.extend(persona.get("household_types", []) or [])
    parts.extend(persona.get("special_targets", []) or [])
    sido = persona.get("sido")
    if sido:
        parts.append(sido)
    return " ".join(filter(None, parts))


class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, policies: list[dict]) -> None:
        """전체 정책 인덱싱."""
        ...

    @abstractmethod
    def retrieve(self, persona: dict, top_k: int) -> list[str]:
        """페르소나 → top_k 정책 ID list."""
        ...
