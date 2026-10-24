"""通用版分析（2026-10-08）：照 plan JSON 對照事前預測，並做事後比較

1. 事前預測：實測平均等待是否落在模型的 95% 範圍內，是否比較接近模型而不是天真 P-K
2. 事後比較（以實際到達時間為條件）：一般 M/G/1（該組全部服務時間獨立抽樣）vs 閒置效應模型
   （用該組自己的資料重新估計：排隊後開始的服務時間＋依閒置時間分段的 δ），以及名目 P-K、批次平均信賴區間

用法：python analyze_plan.py plan_m1.json
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = 2000
EDGES = [1.0, 2.0, 5.0]
rng = np.random.default_rng(29)


def load(tag):
    rows = sorted([r for r in csv.DictReader(open(os.path.join(HERE, "out", tag + ".csv"), encoding="utf-8"))
                   if r["status"] == "ok"], key=lambda r: float(r["t_send"]))
    a = np.array([float(r["t_send"]) for r in rows])
    c = np.array([float(r["t_done"]) for r in rows])
    prev = np.empty(len(c)); _cp = -np.inf    # 10/22 改：照做完的順序重建（一次只處理一個請求，做完的順序就是處理的順序；原本照送出順序）
    for _j in np.argsort(c, kind="stable"):
        prev[_j] = _cp; _cp = c[_j]
    start = np.maximum(a, prev)
    return a, start - a, c - start, a - prev


def sim(a, pool, dvals):
    n, k = len(a), len(a) // 3
    base = rng.choice(pool, size=(R, n))
    c_prev, acc = np.full(R, -np.inf), np.zeros(R)
    for i in range(n):
        start = np.maximum(a[i], c_prev)
        w = start - a[i]
        pen = np.where(w == 0, dvals[np.digitize(a[i] - c_prev, EDGES)], 0.0) if dvals is not None else 0.0
        c_prev = start + base[:, i] + pen
        if i >= k:
            acc += w
    return acc / (n - k)


def main():
    plan_file = sys.argv[1]
    plan = json.load(open(os.path.join(HERE, plan_file), encoding="utf-8"))
    name = os.path.splitext(os.path.basename(plan_file))[0]
    log = {r["tag"]: r for r in csv.DictReader(open(os.path.join(HERE, "out", f"{name}_runlog.csv"),
                                                    encoding="utf-8"))}
    runs = [r for r in plan["runs"] if r["tag"] in log]
    print(f"一、{name} 事前預測的判定")
    inside = closer = 0
    for r in runs:
        a, W, S, gap = load(r["tag"])
        m = W[len(a) // 3:].mean()
        ins, cl = r["lo"] <= m <= r["hi"], abs(m - r["pred"]) < abs(m - r["naive_pk"])
        inside += ins
        closer += cl
        L = log[r["tag"]]
        print(f"  {r['tag']}: 實測 {m:.3f}｜模型 {r['pred']:.3f} [{r['lo']:.2f}, {r['hi']:.2f}]｜天真 P-K {r['naive_pk']:.3f}｜"
              f"{'在範圍內' if ins else '在範圍外'}，{'較接近模型' if cl else '較接近天真 P-K'}｜"
              f"延誤最大 {L['max_lateness_s']} 秒，成功 {L['ok']}/{L['total']}")
    print(f"  → {inside}/{len(runs)} 在範圍內，{closer}/{len(runs)} 較接近模型")

    print("\n二、事後比較（以實際到達時間為條件）")
    print(f"  {'組別':<12}{'n':>4}{'λ̂/λ':>7}{'實測W_q':>9}{'批次CI':>16}{'名目P-K':>9}{'一般MG1':>9}{'閒置模型':>9}"
          f"{'閒置後S':>9}{'排隊後S':>9}")
    for r in runs:
        a, W, S, gap = load(r["tag"])
        n, k = len(a), len(a) // 3
        aw, Ww, Sw = a[k:], W[k:], S[k:]
        lh = (len(aw) - 1) / (aw[-1] - aw[0])
        m = Ww.mean()
        bm = np.array([b.mean() for b in np.array_split(Ww, 10)])
        half = 2.262 * bm.std(ddof=1) / np.sqrt(10)
        pk = r["rate"] * (Sw ** 2).mean() / (2 * (1 - r["rate"] * Sw.mean()))
        q, idle = W > 0, W == 0
        idle[0] = False
        pool = S[q] if q.sum() > 5 else S
        res, b = S[idle] - pool.mean(), np.digitize(gap[idle], EDGES)
        dv = np.array([res[b == i].mean() if np.any(b == i) else 0.0 for i in range(4)])
        s_iid, s_mod = sim(a, S, None), sim(a, pool, dv)
        print(f"  {r['tag']:<12}{n - k:>4}{lh / r['rate']:>7.3f}{m:>9.3f}{f'[{m - half:.2f}, {m + half:.2f}]':>16}"
              f"{pk:>9.3f}{s_iid.mean():>9.3f}{s_mod.mean():>9.3f}{S[idle].mean():>9.3f}{pool.mean():>9.3f}")


if __name__ == "__main__":
    main()
