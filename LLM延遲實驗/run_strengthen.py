"""補強實驗一次跑完（ICAIIC 投稿前，2026-10-05）

包含三部分，順序隨機化：
  rep_*      主實驗重跑：和上次同樣的 λ 與筆數，只換亂數種子（獨立的一組樣本）
  var2_*     變異實驗加大樣本：A、C 各 2 個負載點，每點 600 筆
  confirm_*  確認「飽和」只是睡眠造成：C 在 λ=0.35 重跑 450 筆，這次防休眠

安全措施：
- 全程防止睡眠與螢幕關閉（筆電上蓋不要闔上）
- 每組跑完立刻檢查 open-loop 送出延誤（t_send − t_scheduled），超過 1 秒就標記為不可用
- 每 60 秒記錄 CPU 效能百分比
- 全部跑完自動關閉 llama-server，不會接著跑別的

用法：python run_strengthen.py
"""
import csv
import ctypes
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:8080/v1/chat/completions"
HEALTH = "http://127.0.0.1:8080/health"
STOP = threading.Event()

A = ["--max-tokens", "128"]
C = ["--max-tokens", "32,224", "--token-mix", "0.5,0.5"]
RUNS = [  # (tag, λ, n, 額外參數)
    ("rep_rho30", 0.14995, 150, A),
    ("rep_rho50", 0.24991, 150, A),
    ("rep_rho70", 0.34988, 400, A),
    ("rep_rho85", 0.42485, 600, A),
    ("var2_A_rho50", 0.259, 600, A),
    ("var2_A_rho75", 0.389, 600, A),
    ("var2_C_rho50", 0.233, 600, C),
    ("var2_C_rho70", 0.330, 600, C),
    ("confirm_C_rho75", 0.35, 450, C),
]

PS_CMD = (r"$c = Get-Counter '\Processor Information(_Total)\% Processor Performance' "
          r"-ErrorAction SilentlyContinue; if ($c) { [math]::Round($c.CounterSamples[0].CookedValue,1) }")


def keep_awake(on=True):
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x1, 0x2
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def monitor(path, t0, interval=60):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "wall", "cpu_perf_pct"])
        while not STOP.is_set():
            try:
                val = subprocess.run(["powershell", "-NoProfile", "-Command", PS_CMD],
                                     capture_output=True, text=True, timeout=20).stdout.strip()
            except Exception:
                val = ""
            w.writerow([round(time.perf_counter() - t0, 1), datetime.now().strftime("%H:%M:%S"), val])
            fh.flush()
            STOP.wait(interval)


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
    keep_awake(True)
    try:
        if not server_up():
            print("啟動 llama-server …", flush=True)
            subprocess.Popen(["cmd", "/c", os.path.join(HERE, "start_llama.bat")], cwd=HERE,
                             stdout=open(os.path.join(HERE, "llama_strengthen.log"), "w"),
                             stderr=subprocess.STDOUT)
            for _ in range(60):
                time.sleep(3)
                if server_up():
                    break
        if not server_up():
            sys.exit("llama-server 沒有起來，停止。")
        print("llama-server 正常", flush=True)

        order = RUNS[:]
        random.Random(2026).shuffle(order)
        total = sum(n / lam for _, lam, n, _ in order)
        print(f"共 {len(order)} 組，預估 {total / 3600:.1f} 小時；執行順序：{[t for t, *_ in order]}\n", flush=True)

        t0 = time.perf_counter()
        th = threading.Thread(target=monitor, args=(os.path.join(HERE, "out", "strengthen_cpu.csv"), t0),
                              daemon=True)
        th.start()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        log_path = os.path.join(HERE, "out", "strengthen_runlog.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as lf:
            lw = csv.writer(lf)
            lw.writerow(["order", "tag", "lambda", "n", "start", "end", "minutes",
                         "max_lateness_s", "late_gt_1s", "ok", "total", "usable"])
            for i, (tag, lam, n, extra) in enumerate(order, 1):
                out = os.path.join("out", f"{tag}.csv")
                start = datetime.now()
                print(f"=== [{i}/{len(order)}] {tag}  λ={lam}/s  {n} 筆  預估 {n / lam / 60:.0f} 分"
                      f"（{start:%H:%M} 開始）===", flush=True)
                subprocess.run([sys.executable, os.path.join(HERE, "load_gen.py"), "--url", URL,
                                "--rate", f"{lam}", "--n", str(n), "--prompt-file", "prompts/medium.txt",
                                "--ignore-eos", "--unique-prompt", "--seed", "1", "--out", out, *extra],
                               cwd=HERE, env=env)
                end = datetime.now()
                mx, late, ok, tot = lateness(os.path.join(HERE, out))
                usable = mx <= 1 and late == 0
                print(f"    送出延誤最大 {mx:.3f} 秒，成功 {ok}/{tot}，{'可用' if usable else '⚠️ 不可用'}\n",
                      flush=True)
                lw.writerow([i, tag, lam, n, f"{start:%H:%M:%S}", f"{end:%H:%M:%S}",
                             round((end - start).total_seconds() / 60, 1), round(mx, 3), late, ok, tot, usable])
                lf.flush()
        print(f"全部跑完。執行紀錄：{log_path}", flush=True)
    finally:
        STOP.set()
        subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"], capture_output=True)
        keep_awake(False)
        print("已關閉 llama-server，恢復正常休眠設定。", flush=True)


if __name__ == "__main__":
    main()
