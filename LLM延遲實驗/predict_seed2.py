"""第三組亂數（seed 2）主實驗重跑的事前預測（2026-10-06）

用全部 17 組估出的閒置效應模型（和 analyze_strengthen.py 完全相同的估法），
在 seed 2 的到達時間下模擬 2,000 次，寫下每組的預測與 95% 範圍。輸出 plan_seed2.json。
對照用：名目 P-K（主實驗的 E[S]=1.8966、E[S²]=3.7051）。

用法：python predict_seed2.py
"""
import csv
import json
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = 2000
EDGES = [1.0, 2.0, 5.0]
SEED = 2
RUN_NAMES = ["main2_rho30", "main2_rho50", "long70_rho70", "long85_rho85", "mix_A_rho50", "mix_A_rho75",
             "var_C_rho50", "var_C_rho70", "rep_rho30", "rep_rho50", "rep_rho70", "rep_rho85",
             "var2_A_rho50", "var2_A_rho75", "var2_C_rho50", "var2_C_rho70", "confirm_C_rho75"]
TARGETS = [("rep2_rho30", 0.14995, 150), ("rep2_rho50", 0.24991, 150),
           ("rep2_rho70", 0.34988, 400), ("rep2_rho85", 0.42485, 600)]
rng = np.random.default_rng(17)


def load(name):
    rows = sorted([r for r in csv.DictReader(open(os.path.join(HERE, "out", name + ".csv"), encoding="utf-8"))
                   if r["status"] == "ok"], key=lambda r: float(r["t_send"]))
    a = np.array([float(r["t_send"]) for r in rows])
    c = np.array([float(r["t_done"]) for r in rows])
    cls = np.array(["A" if int(r["out_tokens"]) == 128 else f"C{r['out_tokens']}" for r in rows])
    prev = np.empty(len(c)); _cp = -np.inf    # 10/22 改：照做完的順序重建（一次只處理一個請求，做完的順序就是處理的順序；原本照送出順序）
    for _j in np.argsort(c, kind="stable"):
        prev[_j] = _cp; _cp = c[_j]
    start = np.maximum(a, prev)
    return a, start - a, c - start, a - prev, cls


pools, gaps, res = {}, [], []
for name in RUN_NAMES:
    a, W, S, gap, cls = load(name)
    for k in set(cls):
        q = (W > 0) & (cls == k)
        pools.setdefault(k, []).extend(S[q])
        idle = (W == 0) & (cls == k)
        idle[0] = False
        gaps.extend(gap[idle])
        res.extend(S[idle] - S[q].mean())
poolA = np.array(pools["A"])
gaps, res = np.array(gaps), np.array(res)
b = np.digitize(gaps, EDGES)
dvals = np.array([res[b == i].mean() for i in range(4)])
print(f"一般服務時間（128 tokens）{poolA.mean():.3f} 秒（n={len(poolA)}）；δ = {np.round(dvals, 3)}")


def schedule(lam, n):
    r, t, a = random.Random(SEED), 0.0, []
    for _ in range(n):
        t += r.expovariate(lam)
        r.choices([0], weights=[1.0], k=1)
        r.choices([128], weights=[1.0], k=1)
        a.append(t)
    return np.array(a)


def sim(a):
    n, k = len(a), len(a) // 3
    base = rng.choice(poolA, size=(R, n))
    c_prev, acc = np.full(R, -np.inf), np.zeros(R)
    for i in range(n):
        start = np.maximum(a[i], c_prev)
        w = start - a[i]
        pen = np.where(w == 0, dvals[np.digitize(a[i] - c_prev, EDGES)], 0.0)
        c_prev = start + base[:, i] + pen
        if i >= k:
            acc += w
    return acc / (n - k)


ES, ES2 = 1.8966, 3.7051
plan = {"seed": SEED, "runs": []}
print(f"\n{'組別':<12}{'λ':>9}{'n':>5}{'預估分鐘':>9}{'λ̂/λ':>7}{'名目P-K':>9}{'模型預測':>9}{'95%範圍':>16}")
for tag, lam, n in TARGETS:
    a = schedule(lam, n)
    k = n // 3
    lh = (n - k - 1) / (a[-1] - a[k])
    pk = lam * ES2 / (2 * (1 - lam * ES))
    s = sim(a)
    lo, hi = np.percentile(s, [2.5, 97.5])
    plan["runs"].append({"tag": tag, "rate": lam, "n": n, "pred": round(float(s.mean()), 3),
                         "lo": round(float(lo), 3), "hi": round(float(hi), 3), "pk": round(pk, 3)})
    print(f"{tag:<12}{lam:>9.5f}{n:>5}{a[-1] / 60:>9.0f}{lh / lam:>7.3f}{pk:>9.3f}{s.mean():>9.3f}"
          f"{f'[{lo:.2f}, {hi:.2f}]':>16}")
json.dump(plan, open(os.path.join(HERE, "plan_seed2.json"), "w", encoding="utf-8"), indent=1)
print("\n→ plan_seed2.json")
