"""別台電腦的連線與速度檢查（2026-10-08）

1. 連線延遲：連續 20 次 /health，看中位數與最大值（資料品質規則 4 的基準）
2. 速度：暖機 1 次，再連續送 3 個請求，看服務時間與第一個 token 的時間

用法：python check_remote.py http://192.168.0.164:8080 [max_tokens] [prompt 檔名]
"""
import json
import os
import statistics as st
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
base = sys.argv[1]
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 128
pfile = sys.argv[3] if len(sys.argv) > 3 else "medium.txt"
prompt = open(os.path.join(HERE, "prompts", pfile), encoding="utf-8").read()

rtt = []
for _ in range(20):
    t = time.perf_counter()
    urllib.request.urlopen(base + "/health", timeout=10).read()
    rtt.append((time.perf_counter() - t) * 1000)
print(f"連線延遲（20 次）：中位數 {st.median(rtt):.1f} ms，最大 {max(rtt):.1f} ms")


def req(i):
    payload = {"model": "local", "messages": [{"role": "user", "content": f"[check {i}] " + prompt}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": True, "ignore_eos": True}
    r = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(), method="POST",
                               headers={"Content-Type": "application/json"})
    t0, tf, n = time.perf_counter(), None, 0
    with urllib.request.urlopen(r, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                try:
                    d = (json.loads(line[5:]).get("choices") or [{}])[0].get("delta") or {}
                except json.JSONDecodeError:
                    continue
                if d.get("content"):
                    tf = tf or time.perf_counter()
                    n += 1
    return time.perf_counter() - t0, (tf - t0) if tf else None, n


req(0)
for i in range(1, 4):
    s, ttft, n = req(i)
    print(f"請求 {i}：服務時間 {s:.3f} 秒，第一個 token {ttft:.3f} 秒，輸出 {n} tokens（{pfile}，max {max_tokens}）")
