"""B4: Graph-augmented retrieval (Eligibility Graph traversal + dense fusion).

Pipeline:
  1. Persona → entity-link to graph nodes (tag, region)
  2. Multi-hop walk over has_tag, contains_region, implies_tag (positive)
  3. Filter via excludes_tag (negative)
  4. Score policies by:
     a. graph match score (number of matched eligibility predicates)
     b. fused with dense similarity from query
  5. Audit trail = the graph path that connected each policy to user

Hop limits and ablation flags configurable.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import List, Set

import chromadb
import networkx as nx
from sentence_transformers import SentenceTransformer

from .base import Retriever, PolicyResult


REPO = Path(__file__).resolve().parents[3]
GRAPH_PKL = REPO / "data/graph/eligibility_graph.pkl"
CHROMA_DB_PATH = str(Path(__file__).resolve().parents[2] / "data/vectors/chroma_db")
COLLECTION_NAME = "demo_welfare_policies"


# Persona attribute → tag node mapping
def persona_to_tag_nodes(persona: dict) -> List[str]:
    """Return list of tag node ids that represent the persona's attributes."""
    nodes = []
    # gender
    g = persona.get("gender")
    if g:
        nodes.append(f"tag:gender:{g}")
    # disability
    if persona.get("disability") == "있음":
        nodes.append("tag:disability:필수")
    # income_detail
    for inc in persona.get("income_detail", []) or []:
        if inc and inc != "상관없음":
            nodes.append(f"tag:income_detail:{inc}")
    # household
    for hh in persona.get("household_types", []) or []:
        nodes.append(f"tag:household:{hh}")
    # employment
    if persona.get("employment"):
        nodes.append(f"tag:employment:{persona['employment']}")
    # special
    for sp in persona.get("special_targets", []) or []:
        nodes.append(f"tag:special:{sp}")
    # education
    if persona.get("education"):
        nodes.append(f"tag:education:{persona['education']}")
    return nodes


def persona_to_region_nodes(persona: dict) -> List[str]:
    """Return list of region node ids the persona is part of."""
    sido = persona.get("sido")
    sigungu = persona.get("sigungu")
    nodes = ["region:전국"]
    if sido:
        nodes.append(f"region:시도:{sido}")
        if sigungu:
            nodes.append(f"region:시군구:{sido}/{sigungu}")
    return nodes


class GraphAugmentedRetriever(Retriever):
    name = "Graph"

    def __init__(self,
                 use_implies: bool = True,
                 use_excludes: bool = True,
                 use_region_hierarchy: bool = True,
                 max_hop_implies: int = 2,
                 fusion_alpha: float = 0.5):
        with open(GRAPH_PKL, "rb") as f:
            self.g: nx.MultiDiGraph = pickle.load(f)
        self.use_implies = use_implies
        self.use_excludes = use_excludes
        self.use_region_hierarchy = use_region_hierarchy
        self.max_hop_implies = max_hop_implies
        self.fusion_alpha = fusion_alpha

        # Dense retriever embedded
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.model = SentenceTransformer("BAAI/bge-m3")

        # Index: tag_node -> set(policy nodes that have this tag)
        self._tag_to_policies = {}
        for u, v, edata in self.g.edges(data=True):
            if edata.get("edge_type") == "has_tag":
                # u=policy, v=tag
                self._tag_to_policies.setdefault(v, set()).add(u)

        # Index: region_node -> set(policy nodes applying to this region)
        self._region_to_policies = {}
        for u, v, edata in self.g.edges(data=True):
            if edata.get("edge_type") == "applies_to_region":
                self._region_to_policies.setdefault(v, set()).add(u)

    def _expand_tags_via_implies(self, tag_nodes: Set[str]) -> Set[str]:
        """For each persona tag, follow implies_tag edges within hop limit."""
        if not self.use_implies:
            return set(tag_nodes)
        expanded = set(tag_nodes)
        frontier = set(tag_nodes)
        for _ in range(self.max_hop_implies):
            new_frontier = set()
            for node in frontier:
                if node not in self.g:
                    continue
                for _, succ, edata in self.g.out_edges(node, data=True):
                    if edata.get("edge_type") == "implies_tag" and succ not in expanded:
                        new_frontier.add(succ)
            expanded |= new_frontier
            if not new_frontier:
                break
            frontier = new_frontier
        return expanded

    def _expand_regions_via_hierarchy(self, region_nodes: Set[str]) -> Set[str]:
        """For each persona region, also include 전국 (already in) — no further expansion needed
        since policies already apply to specific level, and we want region MATCH not coverage."""
        return set(region_nodes)  # persona regions already cover {전국, sido, sigungu}

    def _excluded_policies(self, persona: dict) -> Set[str]:
        """Find policies that are excluded for this persona via excludes_tag.
        If policy has tag T and persona has tag T' where excludes(T', T), then policy is filtered.
        """
        if not self.use_excludes:
            return set()
        persona_tag_nodes = persona_to_tag_nodes(persona)
        excluded_target_tags = set()
        for ptn in persona_tag_nodes:
            if ptn not in self.g:
                continue
            for _, succ, edata in self.g.out_edges(ptn, data=True):
                if edata.get("edge_type") == "excludes_tag":
                    excluded_target_tags.add(succ)
        # Find policies that have any excluded tag
        excluded_policies = set()
        for et in excluded_target_tags:
            excluded_policies |= self._tag_to_policies.get(et, set())
        return excluded_policies

    def _eligible_policies_via_graph(self, persona: dict) -> dict:
        """Return {policy_id: graph_match_score}.

        Match score = (# matching positive predicates) for this policy.
        """
        # 1. Persona tag nodes (with implies expansion)
        persona_tags = set(persona_to_tag_nodes(persona))
        expanded_tags = self._expand_tags_via_implies(persona_tags)

        # 2. Persona region nodes
        persona_regions = set(persona_to_region_nodes(persona))

        # 3. Excluded policies
        excluded = self._excluded_policies(persona)

        # 4. Score each policy
        # Region match: policy's applies_to_region must be one of persona_regions
        score_map = {}
        # First: policies in valid regions
        candidate_policies = set()
        for rn in persona_regions:
            candidate_policies |= self._region_to_policies.get(rn, set())

        for pid in candidate_policies:
            if pid in excluded:
                continue
            # Count positive tag matches: out-edges of this policy of type has_tag,
            # whose target is in expanded_tags
            match_count = 0
            tag_names_match = []
            # Check if policy's required tags are all satisfied
            policy_tags_required = set()
            for _, t_node, edata in self.g.out_edges(pid, data=True):
                if edata.get("edge_type") == "has_tag":
                    policy_tags_required.add(t_node)
            # Match
            for pt in policy_tags_required:
                if pt in expanded_tags:
                    match_count += 1
                    tag_names_match.append(pt)
            # Soft scoring: matches per total tags
            # If policy has 0 tags, score = 0.5 (neutral)
            denom = len(policy_tags_required) if policy_tags_required else 1
            graph_score = (match_count + 0.5) / (denom + 1.0)  # smoothed
            score_map[pid] = (graph_score, tag_names_match, policy_tags_required)
        return score_map

    def retrieve(self, query: str, persona: dict, k: int = 10) -> List[PolicyResult]:
        # 1. Graph-based candidate scoring
        graph_scores = self._eligible_policies_via_graph(persona)

        # 2. Dense candidate scoring (top-100 from query)
        emb = self.model.encode([query])[0].tolist()
        dense_res = self.collection.query(
            query_embeddings=[emb], n_results=100, include=["distances"]
        )
        dense_ids = dense_res["ids"][0]
        dense_dists = dense_res["distances"][0]
        dense_score = {pid: float(1.0 - d) for pid, d in zip(dense_ids, dense_dists)}

        # 3. Fuse: only consider policies in graph-eligible candidate set,
        # rank by α*dense + (1-α)*graph_match
        alpha = self.fusion_alpha
        fused = []
        for pid, (g_score, matched_tags, all_required) in graph_scores.items():
            d_score = dense_score.get(pid, 0.0)
            final = alpha * d_score + (1 - alpha) * g_score
            audit = []
            if matched_tags:
                audit.append(f"matched_tags={[t.split(':',2)[2] for t in matched_tags]}")
            audit.append(f"region_match=ok")
            fused.append((pid, final, audit))

        fused.sort(key=lambda x: x[1], reverse=True)
        out = []
        for rank, (pid, score, audit) in enumerate(fused[:k], 1):
            out.append(PolicyResult(
                policy_id=pid, score=float(score), rank=rank, audit_path=audit,
            ))
        return out
