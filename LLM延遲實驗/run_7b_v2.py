"""7B 補跑（第二次，2026-10-08）：一次啟動伺服器，閒置測試 → 事前預測 → 排隊實驗，中途不重開（規則 7）

1. 啟動 7B 伺服器（start_llama.bat 7b），記下放置位置
2. 閒置測試：probe_idle.py --remote 對 127.0.0.1（所以它不會自己開關伺服器），A／H，同時記錄顯示卡時脈 → out/probe_b7v2.csv
3. 事前預測：predict_plan.py（seed 7）→ plan_7b_v2.json，並把預測表附加到 事前預期.md（排隊實驗開始前寫入）
4. 排隊實驗：run_plan.py plan_7b_v2.json（plan 裡 local 是 null，所以 run_plan 不會重開伺服器；開跑前會做規則 7 的速度檢查）
5. 最後關掉伺服器

安全檢查：一般服務時間 b 超過 2 秒（代表模型又被放到 CPU 上，排隊實驗會超過 2 小時）就在第 3 步後停止，不跑排隊實驗。

用法：python run_7b_v2.py
"""
import ctypes
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "http://127.0.0.1:8080"
LOG = os.path.join(HERE, "llama_b7v2.log")
PLAN = "plan_7b_v2.json"
TARGETS = "b7v2_rho30:0.3:200,b7v2_rho50:0.5:300,b7v2_rho70:0.7:300"


def keep_awake(on=True):
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x1 | 0x2 if on else 0))


def server_up():
    try:
        return urllib.request.urlopen(BASE + "/health", timeout=5).status == 200
    except Exception:
        return False


def run(args):
    """執行子程式，輸出一邊印出來（方便看進度）一邊收集"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.Popen([sys.executable] + args, cwd=HERE, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    out = []
    for line in p.stdout:
        print(line, end="", flush=True)
        out.append(line)
    if p.wait() != 0:
        raise SystemExit(f"{args[0]} 失敗（exit {p.returncode}），停止。")
    return "".join(out)


def main():
    if server_up():
        sys.exit("本機 8080 已經有伺服器在跑，先關掉再執行。")
    keep_awake(True)
    try:
        print(f"[{datetime.now():%H:%M:%S}] 啟動 7B 伺服器 …", flush=True)
        subprocess.Popen(["cmd", "/c", os.path.join(HERE, "start_llama.bat"), "7b"], cwd=HERE,
                         stdin=subprocess.DEVNULL, stdout=open(LOG, "w"), stderr=subprocess.STDOUT)
        for _ in range(100):
            time.sleep(3)
            if server_up():
                break
        if not server_up():
            raise SystemExit("伺服器沒有起來，停止。")
        placement = [l.rstrip() for l in open(LOG, encoding="utf-8", errors="replace")
                     if any(k in l for k in ("Vulkan0", "fitting", "offload", "layers to GPU"))]
        print("放置位置：\n  " + "\n  ".join(placement[:8]), flush=True)

        print(f"[{datetime.now():%H:%M:%S}] 閒置測試 …", flush=True)
        run(["probe_idle.py", "--remote", "--base-url", BASE, "--conds", "A,H", "--gaps", "0,0,0.5,1,1.5,2,3,5,8",
             "--blocks", "4", "--reps", "3", "--max-tokens", "16", "--prompt-file", "medium.txt", "--tag", "probe_b7v2"])

        print(f"[{datetime.now():%H:%M:%S}] 事前預測 …", flush=True)
        pred_out = run(["predict_plan.py", "--probe-b", "probe_b7v2", "--seed", "7", "--max-tokens", "16",
                        "--prompt-file", "medium.txt", "--base-url", BASE, "--targets", TARGETS, "--out", PLAN])
        plan = json.load(open(os.path.join(HERE, PLAN), encoding="utf-8"))
        rows = "\n".join(f"| {r['tag']} | {r['rate']} | {r['n']} | {r['pred']}（{r['lo']}–{r['hi']}） | {r['naive_pk']} |"
                         for r in plan["runs"])
        with open(os.path.join(HERE, "事前預期.md"), "a", encoding="utf-8") as f:
            f.write(f"\n### 7B 補跑的事前預測（run_7b_v2.py 自動寫入，{datetime.now():%Y-%m-%d %H:%M}，"
                    f"閒置測試剛跑完、排隊實驗還沒開始）\n\n"
                    f"一般服務時間 b = {plan['b']} 秒，單獨請求 {plan['iso']} 秒。predict_plan.py 的完整輸出：\n\n"
                    f"```\n{pred_out.strip()}\n```\n\n"
                    f"| 組別 | λ | 筆數 | 模型預測（95% 範圍） | 天真 P-K |\n|---|---|---|---|---|\n{rows}\n")
        print("預測已寫入 事前預期.md", flush=True)
        if plan["b"] > 2.0:
            raise SystemExit(f"b = {plan['b']} 秒 > 2 秒：模型可能又被放到 CPU 上，排隊實驗會太久，停止。")

        print(f"[{datetime.now():%H:%M:%S}] 排隊實驗 …", flush=True)
        run(["run_plan.py", PLAN])
        print(f"[{datetime.now():%H:%M:%S}] 全部完成。", flush=True)
    finally:
        subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"], capture_output=True)
        print("已關閉本機 llama-server。", flush=True)
        keep_awake(False)


if __name__ == "__main__":
    main()
