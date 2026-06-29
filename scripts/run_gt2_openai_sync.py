"""OpenAI nano sync parallel for failed batch chunks.

40M enqueued token 한도라 5 chunks (chunk 2-6) batch에서 실패. sync로 직접 호출.
input JSONL은 이미 만들어져 있음 → 그대로 읽어서 sync 호출.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")

OUT_DIR = REPO / "experiments/gt2_full"
OUT_PATH = OUT_DIR / "scores_sync_openai.jsonl"

N_THREADS = 50
SAVE_EVERY = 500
MAX_RETRIES = 3
FAILED_CHUNKS = [2, 3, 4, 5, 6]


def parse_score(content):
    if not content:
        return None
    for ch in content:
        if ch in "012":
            return int(ch)
    return None


def call_with_retry(client, body):
    delay = 1.0
    last_err = None
    for _ in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(**body)
            content = resp.choices[0].message.content or ""
            return parse_score(content), resp.usage.prompt_tokens, resp.usage.completion_tokens, None
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                time.sleep(delay)
                delay = min(delay * 2, 30)
            elif "500" in err_str or "503" in err_str or "timeout" in err_str.lower():
                time.sleep(delay)
                delay = min(delay * 1.5, 10)
            else:
                break
    return None, 0, 0, str(last_err)


def main():
    client = OpenAI()

    # Load all requests from failed chunks
    print(f"Loading failed chunks {FAILED_CHUNKS}...", flush=True)
    requests = []
    for ci in FAILED_CHUNKS:
        path = OUT_DIR / f"batch_input_{ci:02d}.jsonl"
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                requests.append({"custom_id": d["custom_id"], "body": d["body"]})
    print(f"Total requests: {len(requests):,}", flush=True)

    # resume
    done_ids = set()
    if OUT_PATH.exists():
        for line in open(OUT_PATH):
            try:
                d = json.loads(line)
                if d.get("score") is not None:
                    done_ids.add(d["custom_id"])
            except Exception:
                pass
        print(f"Resume: {len(done_ids):,} already done", flush=True)

    todo = [r for r in requests if r["custom_id"] not in done_ids]
    print(f"Todo: {len(todo):,} (× {N_THREADS} threads)", flush=True)

    out_f = open(OUT_PATH, "a", encoding="utf-8")
    write_lock = threading.Lock()
    t0 = time.time()
    n_done = [0]
    n_fail = [0]

    def task(r):
        score, ti, to, err = call_with_retry(client, r["body"])
        return r["custom_id"], score, ti, to, err

    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futures = [ex.submit(task, r) for r in todo]
        for fut in as_completed(futures):
            cid, score, ti, to, err = fut.result()
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
    print(f"\nDone: {n_done[0]:,} in {elapsed/60:.1f}min, fail={n_fail[0]}", flush=True)


if __name__ == "__main__":
    main()
