"""第三組亂數（seed 2）的主實驗重跑（2026-10-06）：閒置效應模型的第二次事前預測檢驗

負載點與筆數和主實驗相同（plan_seed2.json，由 predict_seed2.py 產生；預測先寫進事前預期.md 才開跑）。
3B 模型、medium prompt、輸出 128 tokens、open-loop Poisson、一次處理一個請求。
安全措施同 run_strengthen.py：防休眠、每組檢查送出延誤、跑完關閉伺服器。

用法：python run_seed2.py
"""
import csv
import ctypes
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:8080/v1/chat/completions"
HEALTH = "http://127.0.0.1:8080/health"


def keep_awake(on=True):
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x1 | 0x2 if on else 0))


def server_up():
    try:
        return urllib.request.urlopen(HEALTH, timeout=5).status == 200
    except Exception:
        return False


def lateness(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    lat = [float(r["t_send"]) - float(r["t_scheduled"]) for r in rows]
    ok = sum(1 for r in rows if r["status"] == "ok")
    return max(lat), sum(1 for x in lat if x > 1), ok, len(rows)


def main():
    plan = json.load(open(os.path.join(HERE, "plan_seed2.json"), encoding="utf-8"))
    keep_awake(True)
    try:
        if server_up():
            sys.exit("8080 已經有伺服器在跑，先關掉再執行。")
        print("啟動 llama-server（3B）…", flush=True)
        subprocess.Popen(["cmd", "/c", os.path.join(HERE, "start_llama.bat")], cwd=HERE,
                         stdout=open(os.path.join(HERE, "llama_seed2.log"), "w"), stderr=subprocess.STDOUT)
        for _ in range(100):
            time.sleep(3)
            if server_up():
                break
        if not server_up():
            sys.exit("llama-server 沒有起來，停止。")
        order = plan["runs"][:]
        random.Random(2028).shuffle(order)
        total = sum(r["n"] / r["rate"] for r in order)
        print(f"共 {len(order)} 組，預估 {total / 3600:.1f} 小時；順序：{[r['tag'] for r in order]}", flush=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        with open(os.path.join(HERE, "out", "seed2_runlog.csv"), "w", newline="", encoding="utf-8") as lf:
            lw = csv.writer(lf)
            lw.writerow(["order", "tag", "lambda", "n", "start", "end", "minutes",
                         "max_lateness_s", "late_gt_1s", "ok", "total", "usable"])
            for i, r in enumerate(order, 1):
                out = os.path.join("out", f"{r['tag']}.csv")
                start = datetime.now()
                print(f"=== [{i}/{len(order)}] {r['tag']}  λ={r['rate']}/s  {r['n']} 筆  "
                      f"預估 {r['n'] / r['rate'] / 60:.0f} 分（{start:%H:%M} 開始）===", flush=True)
                subprocess.run([sys.executable, os.path.join(HERE, "load_gen.py"), "--url", URL,
                                "--rate", str(r["rate"]), "--n", str(r["n"]), "--prompt-file", "prompts/medium.txt",
                                "--max-tokens", "128", "--ignore-eos", "--unique-prompt",
                                "--seed", str(plan["seed"]), "--out", out], cwd=HERE, env=env)
                end = datetime.now()
                mx, late, ok, tot = lateness(os.path.join(HERE, out))
                usable = mx <= 1 and late == 0
                print(f"    送出延誤最大 {mx:.3f} 秒，成功 {ok}/{tot}，{'可用' if usable else '⚠️ 不可用'}", flush=True)
                lw.writerow([i, r["tag"], r["rate"], r["n"], f"{start:%H:%M:%S}", f"{end:%H:%M:%S}",
                             round((end - start).total_seconds() / 60, 1), round(mx, 3), late, ok, tot, usable])
                lf.flush()
        print("全部跑完。", flush=True)
    finally:
        subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"], capture_output=True)
        keep_awake(False)
        print("已關閉 llama-server。", flush=True)


if __name__ == "__main__":
    main()
