"""方法 1 的獨立重算（10/24，看過結果之後；不用 analyze_queue.py）：只用校正資料
服務時間 = 從閒置 0 秒的請求抽樣；到的時候伺服器閒著，再加 δ(閒置秒數所在的那一段)；每次模擬重新抽樣校正資料
用法：python -I indep_predict.py <結果zip> <預測_方法1.json>
"""
import csv, io, json, random, sys, zipfile
import numpy as np
EDGES = [1.0, 2.0, 5.0]
z = zipfile.ZipFile(sys.argv[1])
info = json.loads(z.read("run_info.json"))
saved = {r["tag"]: r for r in json.load(open(sys.argv[2], encoding="utf-8"))["runs"]}
cal = [r for r in csv.DictReader(io.StringIO(z.read("calibration.csv").decode("utf-8"))) if not r["error"]]
S0 = np.array([float(r["service"]) for r in cal if float(r["gap"]) == 0])
bins = [np.array([float(r["service"]) for r in cal if float(r["gap"]) > 0 and np.digitize(float(r["idle_before"]), EDGES) == k]) for k in range(4)]
rng = np.random.default_rng(7)
for run in info["runs"]:
    r_, t, a = random.Random(run["seed"]), 0.0, []
    for _ in range(run["n"]):
        t += r_.expovariate(run["lambda"]); a.append(t)
    a = np.array(a); n = len(a); k = n // 3
    means = []
    for _ in range(2000):
        s0 = rng.choice(S0, len(S0)); d = [rng.choice(b, len(b)).mean() - s0.mean() for b in bins]
        S = rng.choice(s0, n)
        done, W = 0.0, np.zeros(n)
        for i in range(n):
            if a[i] >= done:
                S[i] += d[int(np.digitize(a[i] - done if i else 99.0, EDGES))]
                start = a[i]
            else:
                start = done
            W[i] = start - a[i]; done = start + S[i]
        means.append(W[k:].mean())
    m = np.array(means)
    s = saved[run["tag"]]
    print(f"   {run['tag']}：獨立重算 {m.mean():.3f}（{np.percentile(m, 2.5):.3f}–{np.percentile(m, 97.5):.3f}）｜存檔的預測 {s['pred']}（{s['lo']}–{s['hi']}）")
