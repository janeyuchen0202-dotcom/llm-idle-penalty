"""7B 排隊實驗的事前預測（2026-10-06）：只用 7B 閒置效應測試的資料，預測還沒跑的排隊實驗

7B 有一大部分放不進顯示卡，128 tokens 要 25 秒，所以排隊實驗改用 16 tokens 的輸出。
- 一般服務時間 b：out/probe_idle_7b16.csv（16 tokens）中 g = 0 的請求
- 閒置後多出的時間 δ：out/probe_idle_7b.csv（128 tokens）的一般條件，依閒置時間分段（<1、1–2、2–5、≥5 秒），
  δ = 該段平均 − 同一次測試 g=0 的平均。多出的時間落在讀 prompt 的階段，和輸出長度無關，所以可以沿用；
  16 tokens 測試裡 g = 2、5 秒的 δ 拿來核對
- 負載點：目標 ρ = 0.3、0.5、0.7（用 b 換算 λ），亂數種子 3
- 用和 load_gen.py 完全相同的抽法重建到達時間，模擬閒置效應模型 2,000 次；每次模擬都重新抽樣測試資料
  （bootstrap），所以範圍也包含參數估計的不確定性
- 對照用的天真預測：只用「單獨送一個請求」的服務時間（16 tokens 測試中 g = 5 秒）代入 P-K
輸出 plan_7b.json 與預測表（貼進事前預期.md）

用法：python predict_7b.py
"""
import csv
import json
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = 2000
EDGES = [1.0, 2.0, 5.0]
SEED = 3
MAX_TOKENS = 16
TARGETS = [("b7_rho30", 0.3, 200), ("b7_rho50", 0.5, 300), ("b7_rho70", 0.7, 300)]
rng = np.random.default_rng(3)


def rows_of(tag):
    return [r for r in csv.DictReader(open(os.path.join(HERE, "out", f"{tag}.csv"), encoding="utf-8"))
            if r["cond"] == "A"]


r16, r128 = rows_of("probe_idle_7b16"), rows_of("probe_idle_7b")
S0 = np.array([float(r["service"]) for r in r16 if float(r["gap"]) == 0])
S0_128 = np.array([float(r["service"]) for r in r128 if float(r["gap"]) == 0])
by_bin = [[] for _ in range(4)]
for r in r128:
    if float(r["gap"]) > 0:
        by_bin[int(np.digitize(float(r["idle_before"]), EDGES))].append(float(r["service"]))
by_bin = [np.array(x) for x in by_bin]
iso = np.array([float(r["service"]) for r in r16 if float(r["gap"]) == 5])
print(f"16 tokens：一般服務時間 b = {S0.mean():.3f} 秒（n={len(S0)}，CV {S0.std() / S0.mean():.3f}）")
for g in (2, 5):
    s = np.array([float(r["service"]) for r in r16 if float(r["gap"]) == g])
    print(f"  16 tokens 核對：g={g} 秒 δ = {s.mean() - S0.mean():+.3f}（n={len(s)}）")
for i, lab in enumerate(["<1s", "1–2s", "2–5s", "≥5s"]):
    print(f"  128 tokens 測試，閒置 {lab:>5}: n={len(by_bin[i]):>3}  δ = {by_bin[i].mean() - S0_128.mean():+.3f} 秒")
print(f"單獨送一個請求（16 tokens、g=5 秒）的平均服務時間 = {iso.mean():.3f} 秒（n={len(iso)}）")


def schedule(lam, n):
    r, t, a = random.Random(SEED), 0.0, []
    for _ in range(n):
        t += r.expovariate(lam)
        r.choices([0], weights=[1.0], k=1)
        r.choices([MAX_TOKENS], weights=[1.0], k=1)
        a.append(t)
    return np.array(a)


def sim(a):
    n, k = len(a), len(a) // 3
    pool = S0[rng.integers(0, len(S0), (R, len(S0)))]
    base = pool[np.arange(R)[:, None], rng.integers(0, len(S0), (R, n))]
    p128 = S0_128[rng.integers(0, len(S0_128), (R, len(S0_128)))].mean(1)
    dv = np.stack([by_bin[i][rng.integers(0, len(by_bin[i]), (R, len(by_bin[i])))].mean(1) for i in range(4)],
                  1) - p128[:, None]
    c_prev, acc = np.full(R, -np.inf), np.zeros(R)
    for i in range(n):
        start = np.maximum(a[i], c_prev)
        w = start - a[i]
        b = np.digitize(a[i] - c_prev, EDGES)
        pen = np.where(w == 0, dv[np.arange(R), b], 0.0)
        c_prev = start + base[:, i] + pen
        if i >= k:
            acc += w
    return acc / (n - k)


b = S0.mean()
plan = {"seed": SEED, "max_tokens": MAX_TOKENS, "runs": []}
print(f"\n{'組別':<10}{'目標ρ':>6}{'λ':>8}{'n':>5}{'預估分鐘':>9}{'λ̂/λ':>7}{'天真P-K':>9}{'模型預測':>9}{'95%範圍':>16}")
for tag, rho, n in TARGETS:
    lam = round(rho / b, 5)
    a = schedule(lam, n)
    k = n // 3
    lh = (n - k - 1) / (a[-1] - a[k])
    es, es2 = iso.mean(), (iso ** 2).mean()
    naive = lam * es2 / (2 * (1 - lam * es)) if lam * es < 1 else float("inf")
    s = sim(a)
    lo, hi = np.percentile(s, [2.5, 97.5])
    plan["runs"].append({"tag": tag, "rate": lam, "n": n, "pred": round(float(s.mean()), 3),
                         "lo": round(float(lo), 3), "hi": round(float(hi), 3), "naive_pk": round(naive, 3)})
    print(f"{tag:<10}{rho:>6.2f}{lam:>8.4f}{n:>5}{a[-1] / 60:>9.0f}{lh / lam:>7.3f}{naive:>9.3f}{s.mean():>9.3f}"
          f"{f'[{lo:.2f}, {hi:.2f}]':>16}")
json.dump(plan, open(os.path.join(HERE, "plan_7b.json"), "w", encoding="utf-8"), indent=1)
print("\n→ plan_7b.json")
