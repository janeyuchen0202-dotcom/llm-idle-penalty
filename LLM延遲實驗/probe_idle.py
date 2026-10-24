"""閒置效應測試（2026-10-06）：伺服器閒置多久之後，下一個請求會變慢？原因是不是顯示卡省電？

設計（closed-loop，一次只有一個請求，所以沒有排隊）：
  每個請求在前一個完成後，先等 g 秒再送。g = 0、0.5、1、1.5、2、3、5、8 秒，順序隨機。
  兩種條件交替進行（A K A K A K）：
    A  一般：等待期間什麼都不做
    K  保溫：等待期間每 0.5 秒送一個極短的請求（1 個 token），最後一個在正式請求前約 0.5 秒，
       讓顯示卡不會閒超過 0.5 秒
  同時用 nvidia-smi 每 100 毫秒記錄顯示卡的狀態（P-state、時脈、使用率），直接寫檔。

預期：A 條件下閒置 2 秒以上會多約 0.35 秒；如果原因是顯示卡省電，K 條件下就不會變慢。

用法：python probe_idle.py [--model 3b|7b] [--blocks 6] [--reps 4] [--tag probe_idle]
"""
import argparse
import csv
import ctypes
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:8080/v1/chat/completions"
HEALTH = "http://127.0.0.1:8080/health"
GAPS = [0, 0.5, 1, 1.5, 2, 3, 5, 8]
PING_EVERY = 0.5


def keep_awake(on=True):
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x1 | 0x2 if on else 0))


def server_up():
    try:
        return urllib.request.urlopen(HEALTH, timeout=5).status == 200
    except Exception:
        return False


def start_server(model, log_path):
    args = ["cmd", "/c", os.path.join(HERE, "start_llama.bat")] + ([model] if model != "3b" else [])
    subprocess.Popen(args, cwd=HERE, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    for _ in range(100):
        time.sleep(3)
        if server_up():
            return True
    return False


def request(prompt, max_tokens):
    """送出一個 streaming 請求。回傳 (送出、第一個 token、完成) 的 perf_counter 時間，送出的牆上時間，輸出 token 數"""
    payload = {"model": "local", "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": True, "ignore_eos": True}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer sk-no-key-required"})
    wall, t_send, t_first, n = time.time(), time.perf_counter(), None, 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            s = line[5:].strip()
            if s == "[DONE]":
                break
            try:
                delta = (json.loads(s).get("choices") or [{}])[0].get("delta") or {}
            except json.JSONDecodeError:
                continue
            if delta.get("content"):
                if t_first is None:
                    t_first = time.perf_counter()
                n += 1
    return t_send, t_first, time.perf_counter(), wall, n


def wait_until(t):
    while True:
        left = t - time.perf_counter()
        if left <= 0:
            return
        time.sleep(min(left, 0.01))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--tag", default="probe_idle")
    ap.add_argument("--gaps", default=None, help="閒置時間清單，逗號分隔；可重複，例如 0,0,0.5,1（0 出現兩次＝兩倍樣本）")
    ap.add_argument("--conds", default="A,K",
                    help="輪流的條件：A 一般、K 保溫（1 token 的極短請求）、H 重保溫（每次讀完整個 838 token 的 prompt，約 0.25 秒的顯示卡工作）")
    ap.add_argument("--no-gpu-log", action="store_true", help="不跑 nvidia-smi（檢查記錄本身會不會影響顯示卡狀態）")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080", help="伺服器位址；測別台電腦時填那台的 IP")
    ap.add_argument("--remote", action="store_true", help="伺服器在別台電腦：不幫它開關，只送請求")
    ap.add_argument("--prompt-file", default="medium.txt", help="prompts/ 底下的檔名")
    ap.add_argument("--ping-prompt-file", default=None, help="重保溫請求用的 prompt（預設和正式請求相同）；慢的裝置用 short.txt，免得保溫請求拖到正式請求")
    a = ap.parse_args()
    global URL, HEALTH
    URL, HEALTH = a.base_url + "/v1/chat/completions", a.base_url + "/health"
    global GAPS
    if a.gaps:
        GAPS = [float(x) for x in a.gaps.split(",")]

    prompt = open(os.path.join(HERE, "prompts", a.prompt_file), encoding="utf-8").read()
    ping_prompt = open(os.path.join(HERE, "prompts", a.ping_prompt_file), encoding="utf-8").read() if a.ping_prompt_file else prompt
    out = os.path.join(HERE, "out", f"{a.tag}.csv")
    keep_awake(True)
    started_here, smi = False, None
    try:
        if a.remote and not server_up():
            sys.exit(f"連不到 {a.base_url}，先確認那台的伺服器有開、防火牆有允許。")
        if not server_up():
            print("啟動 llama-server …", flush=True)
            if not start_server(a.model, os.path.join(HERE, f"llama_{a.tag}.log")):
                sys.exit("llama-server 沒有起來，停止。")
            started_here = True
        for i in range(3):                                    # 暖機，不記錄
            request(f"[warmup {i}] " + prompt, a.max_tokens)

        if not a.no_gpu_log:
            smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,pstate,clocks.gr,clocks.mem,"
                                    "utilization.gpu,power.draw", "--format=csv,noheader,nounits", "-lms", "100",
                                    "-f", os.path.join(HERE, "out", f"{a.tag}_gpu.csv")])
        rng = random.Random(2026)
        cl = a.conds.split(",")
        conds = [cl[i % len(cl)] for i in range(a.blocks)]
        label = {"A": "一般", "K": "保溫", "H": "重保溫"}
        fields = ["req_id", "block", "cond", "gap", "pings", "idle_before", "t_send", "wall_send", "ttft",
                  "service", "out_tokens"]
        fh = open(out, "w", newline="", encoding="utf-8")          # 每筆即時寫入，中途出事也不會全丟
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        rows, rid = [], 0
        last_end = time.perf_counter()
        for b, cond in enumerate(conds, 1):
            order = GAPS * a.reps
            rng.shuffle(order)
            print(f"區塊 {b}/{len(conds)}（{label[cond]}），{len(order)} 個請求", flush=True)
            for g in order:
                t_start, t_end = last_end, last_end + g
                pings = 0
                if cond in ("K", "H"):
                    k = 1
                    times = []
                    while t_end - PING_EVERY * k >= t_start - 1e-9:
                        times.append(t_end - PING_EVERY * k)
                        k += 1
                    for pt in sorted(times):
                        wait_until(pt)
                        ping = f"[ping {rid}-{pings}] " + (ping_prompt if cond == "H" else "Say hi.")
                        _, _, pe, _, _ = request(ping, 1)
                        last_end = pe
                        pings += 1
                wait_until(t_end)
                ts, tf, td, wall, n = request(f"[request {rid:05d}] " + prompt, a.max_tokens)
                rows.append(dict(req_id=rid, block=b, cond=cond, gap=g, pings=pings,
                                 idle_before=round(ts - last_end, 4), t_send=round(ts, 4),
                                 wall_send=round(wall, 4), ttft=round(tf - ts, 4) if tf else "",
                                 service=round(td - ts, 4), out_tokens=n))
                writer.writerow(rows[-1])
                fh.flush()
                last_end = td
                rid += 1
        fh.close()
        print(f"完成 {len(rows)} 筆 → {out}", flush=True)
    finally:
        if smi:
            smi.kill()
        if started_here:
            subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"], capture_output=True)
        keep_awake(False)


if __name__ == "__main__":
    main()
