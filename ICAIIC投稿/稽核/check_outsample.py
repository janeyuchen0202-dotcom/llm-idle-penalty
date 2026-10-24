"""第十輪審閱的兩項重新分析（2026-10-11）

1. 假伺服器驗證：原本 III-E 拿實測平均和「穩態 P-K」比（偏低約 4%）。這裡改用和論文其他地方一樣的做法：
   代入實際到達時間、從空佇列開始、取同一段區間，模擬 2,000 次，看實測落在哪裡。
   服務時間用設計值 U[0.25, 0.75]（真正的分布），也用實測服務時間抽樣做對照。
2. 事前預測的 9 組 3B：除了已經寫下的閒置效應模型預測，也把標準 M/G/1 用「當時就有的資料」算出來，
   兩個模型並排比較（平均絕對誤差、平均偏差、95% 範圍命中數）。
   標準 M/G/1：服務時間從訓練組（seed 1 後 5 組用前 12 組；seed 2 用 17 組）同一種工作負載的全部實測值獨立抽樣，
   代入目標組的實際到達時間模擬（和 V-B 的做法相同，只是參數來自訓練組）。

用法：python 稽核/check_outsample.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute as rc  # noqa: E402

R = 2000
rng = np.random.default_rng(1011)


def sim_mean_w(a, draw, k):
    """以到達時間 a 模擬 R 次（從空佇列開始），回傳每次在 [k:] 的平均等待"""
    n = len(a)
    s = draw((R, n))
    c_prev, acc = np.full(R, -np.inf), np.zeros(R)
    for i in range(n):
        start = np.maximum(a[i], c_prev)
        if i >= k:
            acc += start - a[i]
        c_prev = start + s[:, i]
    return acc / (n - k)


def summary(sims, measured):
    p = sims.mean()
    lo, hi = np.percentile(sims, [2.5, 97.5])
    return p, lo, hi, measured / p - 1, lo <= measured <= hi


def fake_server():
    print("一、假伺服器（服務時間 U[0.25, 0.75]，丟掉前 20 筆）")
    out = []
    for name, lam in (("my_rho50", 1.0), ("my_rho80", 1.6)):
        d = rc.run(name)
        a, W, S = d["a"], d["W"], d["S"]
        m = W[20:].mean()
        es, es2 = 0.5, (0.75 ** 3 - 0.25 ** 3) / (3 * 0.5)
        pk = lam * es2 / (2 * (1 - lam * es))
        t = summary(sim_mean_w(a, lambda sz: rng.uniform(0.25, 0.75, sz), 20), m)
        e = summary(sim_mean_w(a, lambda sz: rng.choice(S, size=sz), 20), m)
        print(f"  {name}: 實測 {m:.3f}｜穩態 P-K {pk:.3f}（{m / pk - 1:+.1%}）"
              f"｜代入到達時間＋真正分布 {t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]（{t[3]:+.1%}，{'在範圍內' if t[4] else '範圍外'}）"
              f"｜代入到達時間＋實測服務時間 {e[0]:.3f}（{e[3]:+.1%}）")
        out.append((name, m, t[0], t[1], t[2], t[4]))
    return out


def outsample():
    print("\n二、事前預測的 9 組 3B：兩個模型並排")
    mid = {"rep_rho30": (0.327, 0.30, 0.37), "rep_rho50": (0.705, 0.63, 0.80),
           "var2_A_rho50": (0.723, 0.68, 0.77), "var2_A_rho75": (1.675, 1.54, 1.80),
           "var2_C_rho50": (0.874, 0.81, 0.95)}
    s2 = json.load(open(os.path.join(rc.EXP, "plan_seed2.json"), encoding="utf-8"))
    setup = dict(mid)
    setup.update({r["tag"]: (r["pred"], r["lo"], r["hi"]) for r in s2["runs"]})
    first12 = ["main2_rho30", "main2_rho50", "long70_rho70", "long85_rho85", "mix_A_rho50", "mix_A_rho75",
               "var_C_rho50", "var_C_rho70", "var2_C_rho70", "rep_rho85", "rep_rho70", "confirm_C_rho75"]
    all17 = rc.A_RUNS + rc.C_RUNS

    def pool(train, workload_c):
        vals = []
        for n in train:
            d = rc.run(n)
            is_c = n in rc.C_RUNS
            if is_c == workload_c:
                vals.extend(d["S"])
        return np.array(vals)

    rows = []
    for tag, (p_set, lo_set, hi_set) in setup.items():
        d = rc.run(tag)
        train = all17 if tag.startswith("rep2_") else first12
        is_c = tag in rc.C_RUNS
        S = pool(train, is_c)
        k = d["n"] // 3
        m = rc.wq(d)
        st = summary(sim_mean_w(d["a"], lambda sz: rng.choice(S, size=sz), k), m)
        rows.append((tag, m, p_set, lo_set <= m <= hi_set, st[0], st[4]))
        print(f"  {tag:<14} 實測 {m:.3f}｜閒置效應模型 {p_set:.3f}（{m / p_set - 1:+.1%}{'，中' if lo_set <= m <= hi_set else '，沒中'}）"
              f"｜標準 M/G/1 {st[0]:.3f} [{st[1]:.2f}, {st[2]:.2f}]（{st[3]:+.1%}{'，中' if st[4] else '，沒中'}）")
    e_set = [r[1] / r[2] - 1 for r in rows]
    e_std = [r[1] / r[4] - 1 for r in rows]
    print(f"\n  閒置效應模型：平均絕對誤差 {np.mean(np.abs(e_set)):.1%}，平均偏差 {np.mean(e_set):+.1%}，命中 {sum(r[3] for r in rows)}／9")
    print(f"  標準 M/G/1：  平均絕對誤差 {np.mean(np.abs(e_std)):.1%}，平均偏差 {np.mean(e_std):+.1%}，命中 {sum(r[5] for r in rows)}／9")
    print(f"  9 組中閒置效應模型比較接近實測的組數：{sum(abs(a) < abs(b) for a, b in zip(e_set, e_std))}")
    return np.mean(np.abs(e_set)), np.mean(np.abs(e_std))


if __name__ == "__main__":
    fake_server()
    outsample()
