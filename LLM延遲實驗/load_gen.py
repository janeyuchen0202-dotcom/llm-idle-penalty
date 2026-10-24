"""open-loop Poisson 負載產生器（Day 10）

對 OpenAI 相容端點（llama-server 等）送出請求，記錄每個請求的時間戳到 CSV。

重點：**open-loop**
  送出時間在實驗開始前就全部排好，時間到就送，**不等前一個請求的回應**。
  如果改成「收到回應才送下一個」（closed-loop），伺服器變慢時送出速度會自動變慢，
  排隊延遲就被掩蓋了（coordinated omission）。

只用 Python 標準函式庫。

範例：
  python load_gen.py --rate 0.5 --n 200 --prompt-file prompts/medium.txt --out out/rho50.csv
注意：URL 用 127.0.0.1，不要用 localhost（見 --url 的說明）。
  python load_gen.py --url http://localhost:9000/v1/chat/completions --rate 0.25 --n 200 --out out/fake.csv
"""
import argparse
import csv
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

# CSV 欄位
FIELDS = ["req_id", "prompt_type", "prompt_chars",
          "t_scheduled", "t_send", "t_first_token", "t_done",
          "ttft", "total", "out_tokens", "status", "error"]

'''
這裡下面在設定預備發送的訊號數量以及時間
'''
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    # 一定要用 127.0.0.1 而不是 localhost：llama-server 只綁 IPv4，
    # Windows 解析 localhost 會先試 IPv6 (::1)，失敗後才退回 IPv4，每次多約 2 秒。
    p.add_argument("--model", default="local")           # llama-server 不檢查名稱
    p.add_argument("--api-key", default="sk-no-key-required")
    p.add_argument("--rate", type=float, required=True, help="到達率 λ（每秒請求數）")
    p.add_argument("--n", type=int, required=True, help="總請求數")
    p.add_argument("--prompt-file", action="append", default=None,
                   help="prompt 檔。可重複指定：--prompt-file short.txt --prompt-file long.txt")
    p.add_argument("--mix", default=None,
                   help="各 prompt 檔的比例，逗號分隔，例如 0.5,0.5（預設平均分配）")
    p.add_argument("--max-tokens", default="128",
                   help="輸出 token 數。可用逗號給多個值製造變異，例如 32,224（依 --token-mix 的比例隨機挑）")
    p.add_argument("--token-mix", default=None, help="各 max-tokens 的比例，例如 0.5,0.5")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--ignore-eos", action="store_true",
                   help="llama.cpp 專用：強制輸出到 max_tokens，讓每個請求的服務時間一致")
    p.add_argument("--unique-prompt", action="store_true",
                   help="每個請求前面加一個不同的編號，避免伺服器重用快取")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()

'''
這裡下面在設定回傳的 token 數與機率和內容
'''
def load_token_choices(args):
    """回傳 [(輸出token數, 機率), ...]"""
    vals = [int(x) for x in str(args.max_tokens).split(",")]
    if args.token_mix:
        w = [float(x) for x in args.token_mix.split(",")]
        if len(w) != len(vals):
            raise SystemExit("--token-mix 的數量要和 --max-tokens 一樣多")
        t = sum(w); w = [x / t for x in w]
    else:
        w = [1.0 / len(vals)] * len(vals)
    return vals, w


def load_prompts(args):
    """讀 prompt 檔；沒指定就用內建的短 prompt。回傳 [(名稱, 內容), ...] 與各自的機率。"""
    if not args.prompt_file:
        return [("default", "Explain what a queue is in one paragraph.")], [1.0]
    prompts = []
    for path in args.prompt_file:
        with open(path, encoding="utf-8") as fh:
            prompts.append((os.path.splitext(os.path.basename(path))[0], fh.read()))
    if args.mix:
        w = [float(x) for x in args.mix.split(",")]
        if len(w) != len(prompts):
            raise SystemExit("--mix 的數量要和 --prompt-file 一樣多")
        s = sum(w)
        w = [x / s for x in w]
    else:
        w = [1.0 / len(prompts)] * len(prompts)
    return prompts, w

'''
這裡下面在設定單一請求的內容，包含送出、讀取後的回應、記錄時間等
'''

def one_request(args, req_id, ptype, prompt, max_tokens, t_sched, t0, rows, lock):
    """單一請求：送出、讀 streaming 回應、記錄時間。在自己的執行緒裡跑。"""
    if args.unique_prompt:
        prompt = f"[request {req_id:05d}] " + prompt
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(args.url, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    })

    t_send = time.perf_counter() - t0
    t_first = None
    out_tokens = 0
    status, error = "ok", ""
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            for raw in resp:                      # 逐行讀 SSE，不要一次讀完
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                content = delta.get("content")
                if content:                       # 我的理解：收到文字才算一個 token（空的回應片段不算，否則 TTFT 會低估）
                    if t_first is None:
                        t_first = time.perf_counter() - t0   # ← TTFT 記在這裡
                    out_tokens += 1
    except urllib.error.HTTPError as e:
        status, error = "http_error", f"{e.code} {e.reason}"
    except Exception as e:                        # 逾時、連線失敗等
        status, error = "error", f"{type(e).__name__}: {e}"

    t_done = time.perf_counter() - t0
    row = {
        "req_id": req_id, "prompt_type": f"{ptype}/{max_tokens}", "prompt_chars": len(prompt),
        "t_scheduled": round(t_sched, 4), "t_send": round(t_send, 4),
        "t_first_token": round(t_first, 4) if t_first is not None else "",
        "t_done": round(t_done, 4),
        "ttft": round(t_first - t_send, 4) if t_first is not None else "",
        "total": round(t_done - t_send, 4),
        "out_tokens": out_tokens, "status": status, "error": error,
    }
    with lock:                                    # 失敗的請求也要寫進來，不能默默丟掉
        rows.append(row)

'''
這裡下面在設定主程式，包含參數、排程、執行緒、等待完成、寫入 CSV 
'''

def main():
    args = parse_args()
    prompts, weights = load_prompts(args)
    tok_vals, tok_w = load_token_choices(args)
    rng = random.Random(args.seed)

    # ---- 先把所有送出時間排好（指數分布的間隔＝Poisson 到達）----
    schedule, t = [], 0.0
    for i in range(args.n):
        t += rng.expovariate(args.rate)   # 我的理解：用迴圈先把每個請求的時間間隔排好（指數分布＝Poisson 到達）
        ptype, prompt = rng.choices(prompts, weights=weights, k=1)[0]
        mt = rng.choices(tok_vals, weights=tok_w, k=1)[0]
        schedule.append((t, i, ptype, prompt, mt))

    print(f"λ={args.rate}/s，{args.n} 個請求，預計 {schedule[-1][0] / 60:.1f} 分鐘")
    print(f"prompt：{[(p[0], len(p[1])) for p in prompts]}，比例 {[round(w, 2) for w in weights]}")
    print(f"輸出 token：{tok_vals}，比例 {[round(w, 2) for w in tok_w]}")

    rows, lock, threads = [], threading.Lock(), []
    wall_start = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    for t_sched, i, ptype, prompt, mt in schedule:
        # 時間到才送；**不等前一個請求回來**（open-loop 的關鍵就是這一段）
        delay = t_sched - (time.perf_counter() - t0)
        if delay > 0:
            time.sleep(delay)
        th = threading.Thread(target=one_request,
                              args=(args, i, ptype, prompt, mt, t_sched, t0, rows, lock),
                              daemon=True)
        th.start()                        # 我的理解：沒等前一個請求回來，就依照原定時間送出（這就是 open-loop）
        threads.append(th)
        if (i + 1) % 20 == 0:
            done = sum(1 for r in rows)
            print(f"  已送出 {i + 1}/{args.n}，已完成 {done}", flush=True)

    print("全部送出，等待未完成的請求…")
    for th in threads:
        th.join()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows.sort(key=lambda r: r["req_id"])
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    bad = len(rows) - len(ok)
    print(f"\n完成：{len(rows)} 筆（失敗 {bad} 筆）→ {args.out}")
    print(f"實驗開始時間（UTC）：{wall_start}")
    if ok:
        ttfts = sorted(r["ttft"] for r in ok)
        totals = sorted(r["total"] for r in ok)
        mid = len(ok) // 2
        print(f"TTFT 中位數 {ttfts[mid]:.2f}s，總延遲中位數 {totals[mid]:.2f}s，"
              f"輸出 token 中位數 {sorted(r['out_tokens'] for r in ok)[mid]}")


if __name__ == "__main__":
    main()
