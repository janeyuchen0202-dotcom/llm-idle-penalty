"""7B 模型的排隊實驗（2026-10-06）：閒置效應模型換一個模型還準不準？

負載點與筆數寫在 plan_7b.json（由閒置效應測試的結果決定，預測先寫進事前預期.md 才開跑）：
  {"seed": 3, "runs": [{"tag": "b7_rho30", "rate": 0.05, "n": 200}, ...]}

和 3B 的實驗相同：medium prompt、open-loop Poisson、一次處理一個請求；輸出長度由 plan 的 max_tokens 決定
（7B 有一大部分放不進顯示卡，128 tokens 要 25 秒，所以改用 16 tokens）。
安全措施同 run_strengthen.py：防休眠、每組檢查送出延誤、記錄 CPU 效能、跑完關閉伺服器。

用法：python run_7b.py [plan_7b.json]
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
    plan = json.load(open(os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "plan_7b.json"),
                          encoding="utf-8"))
    keep_awake(True)
    try:
        if server_up():
            sys.exit("8080 已經有伺服器在跑，可能不是 7B，先關掉再執行。")
        print("啟動 llama-server（7B）…", flush=True)
        subprocess.Popen(["cmd", "/c", os.path.join(HERE, "start_llama.bat"), "7b"], cwd=HERE,
                         stdout=open(os.path.join(HERE, "llama_7b_runs.log"), "w"), stderr=subprocess.STDOUT)
        for _ in range(100):
            time.sleep(3)
            if server_up():
                break
        if not server_up():
            sys.exit("llama-server 沒有起來，停止。")

        order = plan["runs"][:]
        random.Random(2027).shuffle(order)
        total = sum(r["n"] / r["rate"] for r in order)
        print(f"共 {len(order)} 組，預估 {total / 3600:.1f} 小時；順序：{[r['tag'] for r in order]}", flush=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        log_path = os.path.join(HERE, "out", "b7_runlog.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as lf:
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
                                "--max-tokens", str(plan.get("max_tokens", 128)), "--ignore-eos", "--unique-prompt",
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
