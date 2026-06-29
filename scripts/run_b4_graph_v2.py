"""B4 Graph-augmented retrieval — schema-aligned with personas_v3.

Graph nodes:
  - policy IDs (no prefix): connected via has_tag, applies_to_region
  - tag:* nodes (319): policy requirements
  - region:* nodes (246): policy region

Persona v3 → graph node mapping:
  persona.gender → tag:gender:남성/여성
  persona.disability=있음 → tag:disability:필수
  persona.income_detail → tag:income_detail:기초수급/차상위/...
  persona.household_types → tag:household:1인가구/...
  persona.employment → tag:employment:미취업/...
  persona.special_targets → tag:special:탈북/유공자/...
  persona.education → tag:education:대학생/...
  persona.sido/sigungu → region:시도:.../region:시군구:.../region:전국 (universal)

Eligibility:
  policy P is eligible for persona iff:
    - All of P's has_tag requirements are met by persona's tags (intersection covers)
    - P's region (applies_to_region) overlaps persona's region nodes
    - No excludes_tag is in persona's tags
"""
from __future__ import annotations
import json, math, sys, time, pickle
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from sentence_transformers import SentenceTransformer
import networkx as nx

DATA = REPO / "data"
GRAPH_PKL = REPO / "data/graph/eligibility_graph.pkl"


def persona_tag_nodes(persona):
    """Persona v3 → set of tag node IDs in graph."""
    tags = set()
    g = persona.get("gender")
    if g: tags.add(f"tag:gender:{g}")
    if persona.get("disability") == "있음":
        tags.add("tag:disability:필수")
    for inc in persona.get("income_detail", []) or []:
        if inc and inc != "상관없음":
            tags.add(f"tag:income_detail:{inc}")
    # 항상 universal income_detail도 매칭
    tags.add("tag:income_detail:상관없음")
    for hh in persona.get("household_types", []) or []:
        tags.add(f"tag:household:{hh}")
    if persona.get("employment"):
        tags.add(f"tag:employment:{persona['employment']}")
    for sp in persona.get("special_targets", []) or []:
        tags.add(f"tag:special:{sp}")
    if persona.get("education"):
        tags.add(f"tag:education:{persona['education']}")
    return tags


def persona_region_nodes(persona):
    nodes = {"region:전국"}
    sido = persona.get("sido", "")
    sigungu = persona.get("sigungu", "")
    if sido:
        nodes.add(f"region:시도:{sido}")
        if sigungu:
            nodes.add(f"region:시군구:{sido}/{sigungu}")
    return nodes


def graph_eligible_for_persona(g, persona):
    """Return set of policy_ids eligible for this persona via graph traversal."""
    p_tags = persona_tag_nodes(persona)
    p_regions = persona_region_nodes(persona)
    # Iterate all policy nodes
    eligible = set()
    for node in g.nodes():
        if node.startswith(("tag:", "region:", "cat:")):
            continue
        # node is a policy ID
        # Get policy's required tags + regions
        req_tags = set()
        excl_tags = set()
        pol_regions = set()
        for u, v, d in g.edges(node, data=True):
            et = d.get("edge_type")
            if et == "has_tag":
                req_tags.add(v)
            elif et == "excludes_tag":
                excl_tags.add(v)
            elif et == "applies_to_region":
                pol_regions.add(v)
        # Excluded tags?
        if p_tags & excl_tags:
            continue
        # Each required tag — persona must have at least one (any) match per dimension
        # But here we use flat AND: every req_tag must be in p_tags OR be a 상관없음 wildcard
        # Simplified: if all req_tags ⊆ p_tags+wildcards, eligible
        # Group req_tags by dimension (e.g., income, household)
        # For each dim, persona must satisfy at least one in that dim
        dims = defaultdict(set)
        for tag in req_tags:
            parts = tag.split(":", 2)
            if len(parts) >= 2:
                dim = parts[1]
                dims[dim].add(tag)
        ok = True
        for dim, dim_tags in dims.items():
            if dim_tags & p_tags:
                continue  # match
            ok = False
            break
        if not ok:
            continue
        # Region overlap
        if pol_regions and not (p_regions & pol_regions):
            continue
        eligible.add(node)
    return eligible


def main():
    print("Loading...", flush=True)
    policies = json.load(open(REPO / "data/policies.json"))
    bok = [p for p in policies if "복지로" in p.get("source", "") or "bokjiro" in p.get("url", "")]
    pool = json.load(open(DATA / "query_pool_v3.json"))
    queries = []
    cats = ['welfare', 'employment', 'education', 'housing', 'health', 'culture', 'living']
    for c in cats:
        for sti, st in enumerate(pool[c]['subtopics']):
            for qi, q in enumerate(st['queries']):
                queries.append({"qid": f"{c}_{sti:02d}_{qi}", "text": q})
    personas = json.load(open(DATA / "personas_v3.json"))
    gt3_strict = json.load(open(DATA / "gt3_region_strict.json"))
    gt3_lenient = json.load(open(DATA / "gt3_region_lenient.json"))
    g = pickle.load(open(GRAPH_PKL, "rb"))
    print(f"  policies={len(bok)}, queries={len(queries)}, personas={len(personas)}, graph nodes={g.number_of_nodes()}")

    # Compute graph eligibility per persona
    print("\nComputing graph eligibility per persona...", flush=True)
    t0 = time.time()
    persona_eligible = {}
    for i, persona in enumerate(personas):
        elig = graph_eligible_for_persona(g, persona)
        persona_eligible[persona["persona_id"]] = elig
    avg = sum(len(v) for v in persona_eligible.values()) / len(persona_eligible)
    print(f"  done {time.time()-t0:.1f}s, avg eligible/persona = {avg:.1f}")

    # Dense embeddings (for query rerank)
    print("\nDense embeddings...", flush=True)
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    corpus = [" ".join([p.get("name","") or "", p.get("summary","") or "", (p.get("description") or "")[:300]]) for p in bok]
    pol_embs = model.encode(corpus, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    q_embs = model.encode([q["text"] for q in queries], batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    sims = q_embs @ pol_embs.T
    pid_to_idx = {p["policy_id"]: i for i, p in enumerate(bok)}

    # Evaluate on GT-3
    results = {}
    for setting_name, gt3 in [("strict", gt3_strict), ("lenient", gt3_lenient)]:
        print(f"\n=== B4 (Graph) GT-3 {setting_name} ===", flush=True)
        out = defaultdict(list)
        for pid, qmap in gt3.items():
            elig_set = persona_eligible.get(pid, set())
            for qid, target_list in qmap.items():
                if not target_list:
                    continue
                target = set(target_list)
                qi = next((i for i, q in enumerate(queries) if q["qid"] == qid), None)
                if qi is None:
                    continue
                # Rank graph-eligible by Dense sim to query
                eligible_list = [p for p in elig_set if p in pid_to_idx]
                if not eligible_list:
                    ranked = []
                else:
                    elig_idxs = [pid_to_idx[p] for p in eligible_list]
                    elig_sims = sims[qi][elig_idxs]
                    sorted_idxs = np.argsort(-elig_sims)
                    ranked = [eligible_list[i] for i in sorted_idxs]
                k = 10
                topk = ranked[:k]
                hits = sum(1 for p in topk if p in target)
                p_at_k = hits / k
                r_at_k = hits / len(target)
                dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(topk) if p in target)
                idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(target))))
                ndcg = dcg / idcg if idcg > 0 else 0
                out["p@10"].append(p_at_k)
                out["r@10"].append(r_at_k)
                out["ndcg@10"].append(ndcg)
        avg_m = {k: sum(v) / len(v) if v else 0 for k, v in out.items()}
        results[f"gt3_{setting_name}"] = avg_m
        print(f"  P@10={avg_m['p@10']:.4f}  R@10={avg_m['r@10']:.4f}  NDCG@10={avg_m['ndcg@10']:.4f}", flush=True)

    out_path = REPO / "experiments/b4_graph_v2_gt3_results.json"
    json.dump(results, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
