"""Query pool 생성 (옵션 C+D): 카테고리 → sub-topic → 자연어 query.

Step 1: 7 정책 카테고리(welfare/employment/education/housing/health/culture/living)
        각각의 정책 sample 30개씩 LLM에 보여주고 sub-topic 12-15개 추출.
Step 2: 각 sub-topic 당 자연어 query 1-2개 생성 (페르소나 명시 정보 비노출).
Step 3: 결과 query_pool_candidates.json에 저장 → 사용자+Claude 큐레이션 후 최종 query_pool.json.

비용 추정: gpt-5.4-mini, ~$0.10 / 시간 ~5분
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")

POLICIES_PATH = REPO / "data/policies.json"
LABELS_PATH = REPO / "data/policy_tags_labels.json"
OUT_PATH = REPO / "data/query_pool_candidates.json"

MODEL = "gpt-5.4-mini"
SAMPLE_PER_CATEGORY = 30  # 각 카테고리에서 정책 sample
TOPICS_PER_CATEGORY = 14  # sub-topic 갯수 (목표)
QUERIES_PER_TOPIC = 2     # 각 sub-topic 당 query 갯수

CATEGORIES = ["welfare", "employment", "education", "housing", "health", "culture", "living"]
CATEGORY_KO = {
    "welfare": "사회복지/생활지원",
    "employment": "고용/취업/창업",
    "education": "교육/학자금/장학",
    "housing": "주거/주택/임대",
    "health": "건강/의료/돌봄",
    "culture": "문화/여가/체험",
    "living": "생활/긴급지원/에너지",
}

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def is_bokjiro(p):
    return "복지로" in p.get("source", "") or "bokjiro" in p.get("url", "")


def extract_subtopics(category_ko: str, policies_sample: list[dict]) -> list[str]:
    """LLM에게 정책 sample 보고 카테고리 안의 sub-topic 추출 시키기."""
    pol_lines = []
    for i, p in enumerate(policies_sample, 1):
        name = p.get("name", "")[:60]
        summary = (p.get("summary") or "")[:100]
        pol_lines.append(f"[{i}] {name} | {summary}")
    pol_block = "\n".join(pol_lines)

    prompt = f"""당신은 한국 복지정책 분석 전문가입니다.

다음은 ``{category_ko}'' 카테고리에 속하는 정책 {len(policies_sample)}개입니다:

{pol_block}

이 정책들이 다루는 *주요 관심사 sub-topic*을 {TOPICS_PER_CATEGORY}개 추출하세요.

규칙:
- 사용자가 검색할 때 떠올릴 만한 *관심사 단위*로 묶기 (예: ``청년 월세'', ``노인 일자리'', ``난임 시술'')
- 너무 좁지도 너무 넓지도 않게
- 카테고리 안에서 가능한 다양하게
- *지역명, 나이 숫자, 소득 수준* 같은 페르소나 명시 정보 포함 금지

출력 형식 (한 줄에 sub-topic 하나, 다른 텍스트 X):
sub-topic1
sub-topic2
..."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    content = resp.choices[0].message.content.strip()
    topics = [line.strip().lstrip("0123456789. -·•") for line in content.split("\n") if line.strip()]
    topics = [t for t in topics if 2 <= len(t) <= 50]  # 너무 짧거나 긴 것 제외
    return topics


def generate_queries(category_ko: str, subtopic: str) -> list[str]:
    """sub-topic 하나에서 자연어 query QUERIES_PER_TOPIC개 생성."""
    prompt = f"""당신은 한국 복지정책 검색 사용자 시뮬레이터입니다.

카테고리: {category_ko}
관심사 sub-topic: ``{subtopic}''

이 sub-topic을 검색하려는 일반 시민의 자연스러운 query를 {QUERIES_PER_TOPIC}개 만드세요.

규칙:
- 짧고 자연스럽게 (15-35자)
- 일반 시민 어투 (전문 용어 X)
- *지역명·나이 숫자·구체적 소득 수준 같은 페르소나 명시 정보 포함 금지*
- ``서울'' ``65세'' ``기초수급자'' 같은 단어 X
- 같은 sub-topic 안에서 표현·관점 다양하게

출력 형식 (한 줄에 query 하나, 다른 텍스트 X):
query1
query2"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    content = resp.choices[0].message.content.strip()
    queries = [line.strip().lstrip("0123456789. -·•").strip("\"'") for line in content.split("\n") if line.strip()]
    queries = [q for q in queries if 5 <= len(q) <= 80]
    return queries[:QUERIES_PER_TOPIC]


def main():
    random.seed(42)

    with open(POLICIES_PATH) as f:
        all_policies = json.load(f)
    with open(LABELS_PATH) as f:
        labels = json.load(f)

    bok = [p for p in all_policies if is_bokjiro(p)]
    pol_dict = {p["policy_id"]: p for p in bok}

    # 정책을 7 카테고리로 분류 (LLM이 매긴 라벨 기준)
    by_cat: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for pid, lab in labels.items():
        if pid not in pol_dict:
            continue
        for c in CATEGORIES:
            if lab.get(f"policy.category.{c}") == 1:
                by_cat[c].append(pol_dict[pid])

    print("=== 카테고리별 정책 갯수 ===")
    for c in CATEGORIES:
        print(f"  {c} ({CATEGORY_KO[c]}): {len(by_cat[c])}")
    print()

    # Step 1: sub-topic 추출 + Step 2: query 생성
    out = {"_meta": {"model": MODEL, "topics_per_cat": TOPICS_PER_CATEGORY, "queries_per_topic": QUERIES_PER_TOPIC}}
    for c in CATEGORIES:
        ko = CATEGORY_KO[c]
        sample = random.sample(by_cat[c], min(SAMPLE_PER_CATEGORY, len(by_cat[c])))
        print(f"[{ko}] sub-topic 추출 (sample {len(sample)})...")
        topics = extract_subtopics(ko, sample)
        print(f"  → {len(topics)}개 추출:")
        for t in topics:
            print(f"     · {t}")

        cat_data = []
        for t in topics:
            queries = generate_queries(ko, t)
            cat_data.append({"subtopic": t, "queries": queries})
            for q in queries:
                print(f"     [{t}] → {q}")
        out[c] = {"category_ko": ko, "subtopics": cat_data}
        print()

    # 저장
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 요약
    total_topics = sum(len(out[c]["subtopics"]) for c in CATEGORIES)
    total_queries = sum(len(s["queries"]) for c in CATEGORIES for s in out[c]["subtopics"])
    print(f"=== 결과 요약 ===")
    print(f"  총 sub-topic: {total_topics}")
    print(f"  총 query 후보: {total_queries}")
    print(f"  저장: {OUT_PATH}")
    print(f"\n다음 단계: 사용자 + Claude 큐레이션 → 최종 query_pool.json")


if __name__ == "__main__":
    main()
