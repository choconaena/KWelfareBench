"""Gemini 2.5 Flash-Lite sync parallel: 326K cells with retry.

Tier 1 RPM ~1000 → 326K / 1000 = 5.4h. ThreadPool 30-thread + 429 backoff.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")

POLICIES_PATH = REPO / "data/policies.json"
QUERY_POOL_PATH = REPO / "data/query_pool_v3.json"
OUT_DIR = REPO / "experiments/gt2_full_gemini"
OUT_DIR.mkdir(exist_ok=True, parents=True)
OUT_PATH = OUT_DIR / "scores_sync.jsonl"

MODEL = "gemini-2.5-flash-lite"
N_THREADS = 30
SAVE_EVERY = 500
MAX_RETRIES = 5


GRADING_INSTRUCTION = """당신은 한국 복지정책 추천 평가자입니다.
사용자 query와 정책 한 건이 *주제·관심사 측면에서* 얼마나 관련 있는지 평가하세요.

**규칙**
1. 자격(eligibility)은 평가 X. 오직 query의 주제·관심사가 정책의 주제·내용과 부합하는지만 평가.
2. *수혜자 = 본인* 가정. 사용자 본인이 수혜자라고 가정 (가족 대신 검색 X).
3. 정책 이름의 *prefix*(예: 지역명 "경기도", 대상 prefix "보훈/긴급복지/북한이탈/장애인")는 무시하고, **정책의 핵심 지원 내용**으로 매칭.

**3단계 graded relevance**
- 0 (매우 무관): query 주제와 정책 주제가 거의 관련 없음
- 1 (어느 정도 관련): 같은 큰 카테고리이거나 부분적/인접 관련
- 2 (매우 관련): query가 *직접 가리키는 주제*의 정책 (해당 종류 포함)

**Few-shot 예시**
- query="월세 지원" / policy="청년 월세 지원사업" → 2 (직접)
- query="월세 지원" / policy="신혼부부 전세자금 대출이자" → 1 (주거 카테고리, 인접)
- query="월세 지원" / policy="국가보훈관리(보훈수당)" → 0 (무관)
- query="학비·교육비 지원" / policy="국가보훈대상자 보훈장학금" → 2 (보훈 prefix 무시; 장학금 = 학비 지원)
- query="건강검진" / policy="치매검사비 지원" → 2 (검진 한 종류)
- query="건강검진" / policy="경기도 산후조리비 지원" → 0 (검진 ≠ 산후조리)
- query="정신건강 상담" / policy="자살예방 심리치유 지원" → 2 (자살예방 prefix 무시; 정신건강 상담 한 종류)
- query="긴급 생계비" / policy="긴급복지 의료지원" → 2 (긴급복지 = 직접)
- query="문화 바우처" / policy="가족센터 운영" → 0 (가족교류 ≠ 바우처)"""


def build_prompt(query, policy_name, policy_summary):
    return (f"{GRADING_INSTRUCTION}\n\n---\n사용자 query: \"{query}\"\n"
            f"정책 이름: \"{policy_name}\"\n정책 요약: \"{policy_summary[:200]}\"\n---\n"
            f"등급 (0/1/2 한 자리만):")


def load_pairs():
    pols = json.load(open(POLICIES_PATH))
    bok = [p for p in pols if "복지로" in p.get("source", "") or "bokjiro" in p.get("url", "")]
    pool = json.load(open(QUERY_POOL_PATH))
    pairs = []
    cats = ['welfare', 'employment', 'education', 'housing', 'health', 'culture', 'living']
    for c in cats:
        for sti, st in enumerate(pool[c]['subtopics']):
            for qi, q in enumerate(st['queries']):
                qid = f"{c}_{sti:02d}_{qi}"
                for p in bok:
                    pairs.append({
                        "custom_id": f"{qid}__{p['policy_id']}",
                        "query": q,
                        "policy_name": p['name'],
                        "policy_summary": p.get('summary') or '',
                    })
    return pairs


def call_with_retry(client, prompt):
    """429 backoff + 재시도."""
    delay = 2.0
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=4),
            )
            content = (resp.text or "").strip()
            score = None
            for ch in content:
                if ch in "012":
                    score = int(ch)
                    break
            ti = getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
            to = getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
            return score, ti, to, None
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                time.sleep(delay)
                delay = min(delay * 2, 30)
            elif "500" in err_str or "503" in err_str or "timeout" in err_str.lower():
                time.sleep(delay)
                delay = min(delay * 1.5, 15)
            else:
                # non-retryable
                break
    return None, 0, 0, str(last_err)


def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    pairs = load_pairs()
    print(f"Total pairs: {len(pairs):,}", flush=True)

    # Resume support
    done_ids = set()
    if OUT_PATH.exists():
        for line in open(OUT_PATH):
            try:
                d = json.loads(line)
                if d.get("score") is not None:  # skip prior failures, allow re-attempt
                    done_ids.add(d["custom_id"])
            except Exception:
                pass
        print(f"Resume: {len(done_ids):,} already done", flush=True)

    todo = [p for p in pairs if p["custom_id"] not in done_ids]
    print(f"Todo: {len(todo):,} (× {N_THREADS} threads)", flush=True)

    out_f = open(OUT_PATH, "a", encoding="utf-8")
    write_lock = threading.Lock()
    t0 = time.time()
    n_done = [0]
    n_fail = [0]
    last_save = [time.time()]

    def task(p):
        prompt = build_prompt(p["query"], p["policy_name"], p["policy_summary"])
        score, ti, to, err = call_with_retry(client, prompt)
        return p["custom_id"], score, ti, to, err

    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futures = [ex.submit(task, p) for p in todo]
        for fut in as_completed(futures):
            try:
                cid, score, ti, to, err = fut.result()
            except Exception as e:
                cid, score, ti, to, err = "?", None, 0, 0, str(e)
            with write_lock:
                out_f.write(json.dumps({"custom_id": cid, "score": score, "tokens_in": ti, "tokens_out": to, "error": err}, ensure_ascii=False) + "\n")
                n_done[0] += 1
                if err:
                    n_fail[0] += 1
                if n_done[0] % SAVE_EVERY == 0:
                    out_f.flush()
                    elapsed = time.time() - t0
                    rate = n_done[0] / elapsed
                    eta = (len(todo) - n_done[0]) / rate if rate > 0 else 0
                    print(f"  [{n_done[0]:>6d}/{len(todo)}] rate={rate:.1f}/s ETA={eta/60:.1f}min fail={n_fail[0]}", flush=True)

    out_f.close()
    elapsed = time.time() - t0
    print(f"\nDone: {n_done[0]:,} in {elapsed/60:.1f}min, failures={n_fail[0]}", flush=True)


if __name__ == "__main__":
    main()
