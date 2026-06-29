"""Common interface for all baseline retrievers."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PolicyResult:
    policy_id: str
    score: float = 0.0
    rank: int = 0
    audit_path: Optional[List[str]] = field(default=None)  # graph traversal path


class Retriever(ABC):
    """All baselines implement this. Returns ranked list of policy_ids."""
    name: str = "base"

    @abstractmethod
    def retrieve(self, query: str, persona: dict, k: int = 10) -> List[PolicyResult]:
        ...
