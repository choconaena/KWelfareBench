"""GT-2 graded prompt 검증: 20건 sample (수동 expected grade × LLM 채점).

목적:
  - prompt 명확한지 / 0/1/2 구분 잘 되는지 확인
  - persona-orthogonal query (지역명/나이 X) 가 정책과 매칭 잘 되는지
  - "수혜자 = 본인" 가정 명시가 LLM에게 제대로 전달되는지

20건 구성: clear-2 8건 + partial-1 6건 + clear-0 6건
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")

POLICIES_PATH = REPO / "data/policies.json"

MODEL = "gpt-5.4-nano"  # 결정됨 (test_nano_vs_mini.py 결과)
PRICE_IN = 0.20
PRICE_OUT = 1.25

GRADING_INSTRUCTION = """당신은 한국 복지정책 추천 평가자입니다.
사용자의 query와 정책 한 건이 *주제·관심사 측면에서* 얼마나 관련 있는지 평가하세요.

**중요 규칙**:
- 자격(eligibility) 평가 X. 오직 query의 주제·관심사가 정책의 주제·내용과 부합하는지만 봅니다.
- *수혜자 = 본인* 가정. 사용자 본인이 수혜자라고 가정하고 평가 (가족 대신 검색 X).

3단계 graded relevance:
- 0 (매우 무관): query 주제와 정책 주제가 거의 관련 없음
  예: "월세 지원" query × "보훈수당" 정책 → 0
- 1 (어느 정도 관련): 같은 큰 카테고리이거나 부분적 관련
  예: "월세 지원" query × "신혼부부 전세자금" 정책 → 1 (둘 다 주거지만 월세 vs 전세)
- 2 (매우 관련): query가 직접적으로 이 정책의 주제를 가리킴
  예: "월세 지원" query × "청년 월세 지원" 정책 → 2

판단 기준:
- query의 *주제 단어*에 집중 (주거/교육/취업/돌봄/의료 등)
- 정책 *이름* + *요약* 보고 카테고리 매칭
- 자격 조건은 무시 (대상자 차이 무관)
- 같은 카테고리 안에서도 직접 매칭 vs 부분 매칭 구분"""


# 20건 test pair (handpicked with expected grade)
TEST_PAIRS = [
    # === clear 2 (직접 매칭) — 8건 ===
    ("월세·전월세 지원 신청 방법", "bokjiro_WLF00005112", 2,
     "주거/주거비 부담 경감 + 청년 월세 지원사업 → 직접"),
    ("학비·교육비 지원 어떻게 받나요", "bokjiro_WLF00000054", 2,
     "교육/학비·교육비 + 보훈장학금 → 직접"),
    ("교통약자 콜택시 이용 방법", "bokjiro_WLF00002567", 2,
     "생활/교통·이동 + 장애인 콜택시 → 직접"),
    ("건강검진 무료로 받을 수 있나요", "bokjiro_WLF00005004", 2,
     "건강/건강검진 + 치매검사비 → 직접 (검진)"),
    ("긴급 생계비 부족할 때 지원받는 법", "bokjiro_WLF00003179", 2,
     "생활/긴급 생계 + 긴급복지 의료지원 → 직접"),
    ("보훈수당 신청 방법", "bokjiro_WLF00004200", 2,
     "생활/보훈수당 + 국가보훈관리(수당) → 직접"),
    ("어린이집 비용 도움받는 법", "bokjiro_WLF00001140", 2,
     "교육/보육료 + 방과후보육료 → 직접"),
    ("정신건강 상담 무료로 받는 법", "bokjiro_WLF00002781", 2,
     "건강/정신건강 + 자살예방 심리치유 → 직접"),

    # === partial 1 (같은 카테고리, 부분 관련) — 6건 ===
    ("월세·전월세 지원 신청 방법", "bokjiro_WLF00001851", 1,
     "주거 카테고리 같지만 월세 vs 전세자금 (다른 결)"),
    ("직업훈련·재취업 지원 알아보기", "bokjiro_WLF00004654", 1,
     "고용 카테고리지만 직업훈련 vs 인턴제 (다른 결)"),
    ("건강검진 무료로 받을 수 있나요", "bokjiro_WLF00001361", 1,
     "건강 카테고리지만 검진 vs 산후조리 (다른 의료영역)"),
    ("출산·양육 지원 어떻게 받나요", "bokjiro_WLF00003246", 1,
     "출산·양육 카테고리지만 양육 vs 임신·출산 의료비 (인접)"),
    ("학비·교육비 지원 어떻게 받나요", "bokjiro_WLF00005997", 1,
     "교육 카테고리지만 학비 vs 농어촌유학 (다른 결)"),
    ("문화 바우처·여가 프로그램 신청 방법", "bokjiro_WLF00001186", 1,
     "문화/여가 카테고리지만 문화바우처 vs 가족센터 (인접)"),

    # === clear 0 (무관) — 6건 ===
    ("월세·전월세 지원 신청 방법", "bokjiro_WLF00004200", 0,
     "주거 query × 보훈수당 → 무관"),
    ("학비·교육비 지원 어떻게 받나요", "bokjiro_WLF00006127", 0,
     "교육 query × 임대보증금 → 무관"),
    ("건강검진 무료로 받을 수 있나요", "bokjiro_WLF00005665", 0,
     "건강 query × 중증장애인 공공일자리 → 무관"),
    ("교통약자 콜택시 이용 방법", "bokjiro_WLF00000054", 0,
     "교통·이동 query × 보훈장학금 → 무관"),
    ("출산·양육 지원 어떻게 받나요", "bokjiro_WLF00003206", 0,
     "출산·양육 query × 북한이탈 의료비 → 무관"),
    ("문화 바우처·여가 프로그램 신청 방법", "bokjiro_WLF00001099", 0,
     "문화 query × 농업인 건강보험료 → 무관"),
]


def build_prompt(query: str, policy_name: str, policy_summary: str) -> str:
    return (
        f"{GRADING_INSTRUCTION}\n\n"
        f"---\n"
        f"사용자 query: \"{query}\"\n"
        f"정책 이름: \"{policy_name}\"\n"
        f"정책 요약: \"{policy_summary[:200]}\"\n"
        f"---\n"
        f"등급 (0/1/2 한 자리만, 다른 텍스트 X):"
    )


def parse_score(content: str):
    for ch in content:
        if ch in "012":
            return int(ch)
    return None


def main():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    pols = json.load(open(POLICIES_PATH))
    pol_dict = {p["policy_id"]: p for p in pols}

    print(f"=== GT-2 grading test: 20 pairs × {MODEL} ===\n")
    print(f"{'#':>3s} {'expected':>9s} {'LLM':>4s} {'match':>6s}  {'query':<35s} | {'policy':<40s} | comment")
    print("-" * 160)

    results = []
    tin_total = tout_total = 0
    correct = 0
    one_off = 0  # adjacent (0↔1, 1↔2 = 1 차이)
    t0 = time.time()

    for i, (query, pid, expected, comment) in enumerate(TEST_PAIRS, 1):
        if pid not in pol_dict:
            print(f"  [{i:>2d}] SKIP: {pid} not found")
            continue
        pol = pol_dict[pid]
        prompt = build_prompt(query, pol["name"], pol.get("summary") or "")
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            tin = resp.usage.prompt_tokens
            tout = resp.usage.completion_tokens
            tin_total += tin
            tout_total += tout
            score = parse_score(content)
            match = "OK" if score == expected else (
                "~ 1차이" if score is not None and abs(score - expected) == 1 else "❌"
            )
            if score == expected:
                correct += 1
            elif score is not None and abs(score - expected) == 1:
                one_off += 1
            print(f"  [{i:>2d}]  exp={expected:>2d}   {score!s:>4s}  {match:>6s}  "
                  f"{query[:35]:<35s} | {pol['name'][:40]:<40s} | {comment[:50]}")
            results.append({"i": i, "query": query, "policy": pol["name"], "expected": expected,
                            "llm": score, "comment": comment, "raw": content})
        except Exception as e:
            print(f"  [{i:>2d}] FAIL: {e}")
            results.append({"i": i, "fail": str(e)})

    elapsed = time.time() - t0
    cost = tin_total / 1_000_000 * PRICE_IN + tout_total / 1_000_000 * PRICE_OUT
    n_total = len([r for r in results if "llm" in r])
    print()
    print("=" * 60)
    print(f"  정확 일치: {correct}/{n_total} = {correct/n_total*100:.1f}%")
    print(f"  1 차이 (adjacent):  {one_off}/{n_total} = {one_off/n_total*100:.1f}%")
    print(f"  완전 빗나감 (≥2 차이): {n_total - correct - one_off}")
    print(f"  토큰: in={tin_total}, out={tout_total} | 비용 ${cost:.6f}")
    print(f"  시간: {elapsed:.1f}s")
    print(f"  → 326K cells extrapolation: ${cost / n_total * 326_000:.2f}")
    print(f"  → batch 50% (~24h): ${cost / n_total * 326_000 / 2:.2f}")

    # 결과 저장
    out_path = REPO / "experiments/gt2_test_20.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    json.dump({"model": MODEL, "n": n_total, "correct": correct, "one_off": one_off,
               "results": results}, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"\n  저장: {out_path}")


if __name__ == "__main__":
    main()
