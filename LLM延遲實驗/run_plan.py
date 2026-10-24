"""通用版排隊實驗（2026-10-08）：照 plan JSON（predict_plan.py 產生）跑一組負載點

- plan["local"] 有值（3b、7b）：在本機啟動 llama-server，跑完關掉
- plan["local"] 是 null：伺服器在別台電腦（plan["base_url"]），只送請求，不幫它開關
- 負載產生器一律在本機跑（load_gen.py，open-loop Poisson），送出時間、完成時間都在本機記錄
- 防休眠、每組檢查送出延誤，紀錄寫到 out/<plan 名稱>_runlog.csv
- 伺服器在別台電腦時，每組前後各量 20 次連線延遲（資料品質規則 4，記在 runlog）
- plan 有記 b 時，開跑前先暖機 1 次、再連續送 3 個請求，平均服務時間和 b 差超過 10% 就停止（規則 7）

用法：python run_plan.py plan_m1.json
"""
import csv
import ctypes
import json
import os
import random
import statistics as st
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))


def keep_awake(on=True):
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x1 | 0x2 if on else 0))


def server_up(health):
    try:
        return urllib.request.urlopen(health, timeout=5).status == 200
    except Exception:
        return False


def rtt20(base):
    r = []
    for _ in range(20):
        t = time.perf_counter()
        urllib.request.urlopen(base + "/health", timeout=10).read()
        r.append((time.perf_counter() - t) * 1000)
        time.sleep(0.1)
    return round(st.median(r), 1), round(max(r), 1)


def one_request(url, prompt, max_tokens):
    payload = {"model": "local", "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": True, "ignore_eos": True}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        for _ in resp:
            pass
    return time.perf_counter() - t0


def speed_check(url, plan):
    """規則 7：開跑前的服務時間要和寫預測時的 b 相差不到 10%"""
    prompt = open(os.path.join(HERE, "prompts", plan["prompt_file"]), encoding="utf-8").read()
    one_request(url, "[speedcheck warmup] " + prompt, plan["max_tokens"])
    s = [one_request(url, f"[speedcheck {i}] " + prompt, plan["max_tokens"]) for i in range(3)]
    m = sum(s) / len(s)
    diff = m / plan["b"] - 1
    print(f"規則 7 速度檢查：連續 3 個請求平均 {m:.3f} 秒，預測用的 b = {plan['b']:.3f} 秒，差 {100 * diff:+.1f}%",
          flush=True)
    return m, diff


def lateness(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    lat = [float(r["t_send"]) - float(r["t_scheduled"]) for r in rows]
    ok = sum(1 for r in rows if r["status"] == "ok")
    return max(lat), sum(1 for x in lat if x > 1), ok, len(rows)


def main():
    plan_file = sys.argv[1]
    plan = json.load(open(os.path.join(HERE, plan_file), encoding="utf-8"))
    name = os.path.splitext(os.path.basename(plan_file))[0]
    url, health = plan["base_url"] + "/v1/chat/completions", plan["base_url"] + "/health"
    keep_awake(True)
    try:
        if plan.get("local"):
            if server_up(health):
                sys.exit("本機 8080 已經有伺服器在跑，先關掉再執行。")
            args = ["cmd", "/c", os.path.join(HERE, "start_llama.bat")] + ([plan["local"]] if plan["local"] != "3b" else [])
            subprocess.Popen(args, cwd=HERE, stdout=open(os.path.join(HERE, f"llama_{name}.log"), "w"),
                             stderr=subprocess.STDOUT)
            for _ in range(100):
                time.sleep(3)
                if server_up(health):
                    break
        if not server_up(health):
            sys.exit(f"連不到 {plan['base_url']}，停止。")
        remote = not plan.get("local")
        if plan.get("b"):
            m, diff = speed_check(url, plan)
            if abs(diff) > 0.10:
                sys.exit("服務時間和寫預測時差超過 10%，依規則 7 停止：預測作廢，要重新測過。")
        order = plan["runs"][:]
        random.Random(plan["seed"] + 2000).shuffle(order)
        total = sum(r["n"] / r["rate"] for r in order)
        print(f"{name}：共 {len(order)} 組，預估 {total / 3600:.1f} 小時；順序：{[r['tag'] for r in order]}", flush=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        log_path = os.path.join(HERE, "out", f"{name}_runlog.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as lf:
            lw = csv.writer(lf)
            lw.writerow(["order", "tag", "lambda", "n", "start", "end", "minutes",
                         "max_lateness_s", "late_gt_1s", "ok", "total", "usable",
                         "rtt_before_med", "rtt_before_max", "rtt_after_med", "rtt_after_max", "net_flag"])
            for i, r in enumerate(order, 1):
                out = os.path.join("out", f"{r['tag']}.csv")
                rb = rtt20(plan["base_url"]) if remote else ("", "")
                start = datetime.now()
                print(f"=== [{i}/{len(order)}] {r['tag']}  λ={r['rate']}/s  {r['n']} 筆  "
                      f"預估 {r['n'] / r['rate'] / 60:.0f} 分（{start:%H:%M} 開始）===", flush=True)
                subprocess.run([sys.executable, os.path.join(HERE, "load_gen.py"), "--url", url,
                                "--rate", str(r["rate"]), "--n", str(r["n"]),
                                "--prompt-file", f"prompts/{plan['prompt_file']}",
                                "--max-tokens", str(plan["max_tokens"]), "--ignore-eos", "--unique-prompt",
                                "--seed", str(plan["seed"]), "--out", out], cwd=HERE, env=env)
                end = datetime.now()
                ra = rtt20(plan["base_url"]) if remote else ("", "")
                mx, late, ok, tot = lateness(os.path.join(HERE, out))
                usable = mx <= 1 and late == 0 and ok == tot
                unstable = remote and (rb[0] > 20 or ra[0] > 20 or rb[1] > 500 or ra[1] > 500)
                net = ("網路不穩" if unstable else "合格") if remote else ""
                print(f"    送出延誤最大 {mx:.3f} 秒，成功 {ok}/{tot}，{'可用' if usable else '⚠️ 不可用'}"
                      + (f"｜連線延遲 前 {rb[0]}/{rb[1]} ms、後 {ra[0]}/{ra[1]} ms（中位數/最大）→ {net}" if remote else ""),
                      flush=True)
                lw.writerow([i, r["tag"], r["rate"], r["n"], f"{start:%H:%M:%S}", f"{end:%H:%M:%S}",
                             round((end - start).total_seconds() / 60, 1), round(mx, 3), late, ok, tot, usable,
                             rb[0], rb[1], ra[0], ra[1], net])
                lf.flush()
        print("全部跑完。", flush=True)
    finally:
        if plan.get("local"):
            subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"], capture_output=True)
            print("已關閉本機 llama-server。", flush=True)
        keep_awake(False)


if __name__ == "__main__":
    main()
