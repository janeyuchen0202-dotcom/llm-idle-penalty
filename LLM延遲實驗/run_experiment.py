"""主實驗執行器（Day 22–24）

依序跑多個負載點，並同時記錄 CPU 效能百分比（偵測降頻）。

- 負載點的**執行順序隨機化**，避免「越後面越熱」和「ρ 越大」混在一起
- 每 60 秒記錄一次 CPU 效能百分比（100% = 標稱頻率；持續低於 100% 代表降頻）
- 全部跑完自動和 P-K 理論比較

用法：
  python run_experiment.py --es 4.1734 --es2 17.4322 --n 150
"""
import argparse
import csv
import os
import random
import subprocess
import sys
import threading
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
STOP = threading.Event()

PS_CMD = (r"$c = Get-Counter '\Processor Information(_Total)\% Processor Performance' "
          r"-ErrorAction SilentlyContinue; if ($c) { [math]::Round($c.CounterSamples[0].CookedValue,1) }")


def monitor(path, t0, interval=60):
    """背景執行緒：定期記錄 CPU 效能百分比"""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "wall", "cpu_perf_pct"])
        while not STOP.is_set():
            try:
                out = subprocess.run(["powershell", "-NoProfile", "-Command", PS_CMD],
                                     capture_output=True, text=True, timeout=20)
                val = out.stdout.strip()
            except Exception:
                val = ""
            w.writerow([round(time.perf_counter() - t0, 1),
                        datetime.now().strftime("%H:%M:%S"), val])
            fh.flush()
            STOP.wait(interval)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--es", type=float, required=True, help="Day 19 量到的 E[S]")
    p.add_argument("--es2", type=float, required=True, help="Day 19 量到的 E[S^2]")
    p.add_argument("--rho", default="0.3,0.5,0.7,0.85")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--prompt", default="prompts/medium.txt")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--tag", default="main")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    a = p.parse_args()

    rhos = [float(x) for x in a.rho.split(",")]
    random.Random(a.seed).shuffle(rhos)          # ← 隨機化順序
    est = sum(a.n / (r / a.es) for r in rhos) / 60
    print(f"執行順序（已隨機化）：{rhos}")
    print(f"每點 {a.n} 筆，預估共 {est:.0f} 分鐘\n")

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    t0 = time.perf_counter()
    th = threading.Thread(target=monitor, args=(os.path.join(HERE, "out", f"{a.tag}_cpu.csv"), t0), daemon=True)
    th.start()

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    log_path = os.path.join(HERE, "out", f"{a.tag}_runlog.csv")
    runs = []
    with open(log_path, "w", newline="", encoding="utf-8") as lf:
        lw = csv.writer(lf)
        lw.writerow(["order", "rho", "lambda", "n", "start", "end", "minutes", "csv"])
        for i, rho in enumerate(rhos, 1):
            lam = rho / a.es
            out = f"out/{a.tag}_rho{int(round(rho * 100))}.csv"
            start = datetime.now()
            print(f"=== [{i}/{len(rhos)}] ρ={rho}  λ={lam:.4f}/s  {a.n} 筆  預估 {a.n / lam / 60:.0f} 分 "
                  f"（{start.strftime('%H:%M')} 開始）===", flush=True)
            subprocess.run([sys.executable, os.path.join(HERE, "load_gen.py"),
                            "--url", a.url, "--rate", f"{lam}", "--n", str(a.n),
                            "--prompt-file", a.prompt, "--max-tokens", str(a.max_tokens),
                            "--ignore-eos", "--unique-prompt", "--out", out],
                           cwd=HERE, env=env, check=True)
            end = datetime.now()
            lw.writerow([i, rho, round(lam, 5), a.n, start.strftime("%H:%M:%S"),
                         end.strftime("%H:%M:%S"), round((end - start).total_seconds() / 60, 1), out])
            lf.flush()
            runs.append(f"{out}:{lam}")

    STOP.set()
    print("\n=== 全部跑完，和 P-K 理論比較 ===\n")
    runs.sort(key=lambda s: float(s.rsplit(":", 1)[1]))
    subprocess.run([sys.executable, os.path.join(HERE, "compare_pk.py"),
                    "--es", f"{a.es}", "--es2", f"{a.es2}", "--warmup", "20", *runs],
                   cwd=HERE, env=env)
    print(f"\n執行紀錄：{log_path}")
    print(f"CPU 效能紀錄：out/{a.tag}_cpu.csv")


if __name__ == "__main__":
    main()
