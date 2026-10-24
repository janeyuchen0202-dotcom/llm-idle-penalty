"""第三組亂數（seed 2）重跑的分析（2026-10-06）

1. 對照事前預測（plan_seed2.json，06:08 寫下）：實測是否落在閒置效應模型的 95% 範圍內，
   是否比較接近模型而不是名目 P-K
2. 事後比較（以該組實際到達時間為條件）：一般 M/G/1（該組全部服務時間獨立抽樣）vs 閒置效應模型
   （和事前預測同一組參數），以及名目 P-K（該組服務時間的動差）、批次平均信賴區間

用法：python analyze_seed2.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = 2000
EDGES = [1.0, 2.0, 5.0]
rng = np.random.default_rng(19)
plan = json.load(open(os.path.join(HERE, "plan_seed2.json"), encoding="utf-8"))
log = {r["tag"]: r for r in csv.DictReader(open(os.path.join(HERE, "out", "seed2_runlog.csv"), encoding="utf-8"))}
DVALS = np.loadtxt(os.path.join(HERE, "out", "idle_penalty_bins.csv"), delimiter=",", skiprows=1)[:, 0]


def load(tag):
    rows = sorted([r for r in csv.DictReader(open(os.path.join(HERE, "out", tag + ".csv"), encoding="utf-8"))
                   if r["status"] == "ok"], key=lambda r: float(r["t_send"]))
    a = np.array([float(r["t_send"]) for r in rows])
    c = np.array([float(r["t_done"]) for r in rows])
    prev = np.empty(len(c)); _cp = -np.inf    # 10/22 改：照做完的順序重建（一次只處理一個請求，做完的順序就是處理的順序；原本照送出順序）
    for _j in np.argsort(c, kind="stable"):
        prev[_j] = _cp; _cp = c[_j]
    start = np.maximum(a, prev)
    return a, start - a, c - start


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


# 一般服務時間的樣本（和事前預測相同：17 組裡 128 tokens、排隊後開始的請求）
A_RUNS = ["main2_rho30", "main2_rho50", "long70_rho70", "long85_rho85", "mix_A_rho50", "mix_A_rho75",
          "rep_rho30", "rep_rho50", "rep_rho70", "rep_rho85", "var2_A_rho50", "var2_A_rho75"]
pool = np.concatenate([S[W > 0] for a, W, S in (load(n) for n in A_RUNS)])

print("一、事前預測的判定")
inside = closer = 0
for r in plan["runs"]:
    a, W, S = load(r["tag"])
    k = len(a) // 3
    m = W[k:].mean()
    ins, cl = r["lo"] <= m <= r["hi"], abs(m - r["pred"]) < abs(m - r["pk"])
    inside += ins
    closer += cl
    L = log[r["tag"]]
    print(f"  {r['tag']}: 實測 {m:.3f}｜模型 {r['pred']:.3f} [{r['lo']:.2f}, {r['hi']:.2f}]｜名目 P-K {r['pk']:.3f}｜"
          f"{'在範圍內' if ins else '在範圍外'}，{'較接近模型' if cl else '較接近 P-K'}｜"
          f"延誤最大 {L['max_lateness_s']} 秒，成功 {L['ok']}/{L['total']}")
print(f"  → {inside}/4 在範圍內（預期至少 3），{closer}/4 較接近模型（預期 4）："
      f"{'符合' if inside >= 3 and closer == 4 else '不符合'}")

print("\n二、事後比較（以實際到達時間為條件）")
print(f"  {'組別':<12}{'n':>4}{'λ̂/λ':>7}{'實測W_q':>9}{'批次CI':>16}{'名目P-K':>9}{'一般MG1':>9}{'閒置模型':>9}")
for r in plan["runs"]:
    a, W, S = load(r["tag"])
    n, k = len(a), len(a) // 3
    aw, Ww, Sw = a[k:], W[k:], S[k:]
    lh = (len(aw) - 1) / (aw[-1] - aw[0])
    m = Ww.mean()
    bm = np.array([b.mean() for b in np.array_split(Ww, 10)])
    half = 2.262 * bm.std(ddof=1) / np.sqrt(10)
    pk = r["rate"] * (Sw ** 2).mean() / (2 * (1 - r["rate"] * Sw.mean()))
    s_iid, s_mod = sim(a, S, None), sim(a, pool, DVALS)
    print(f"  {r['tag']:<12}{n - k:>4}{lh / r['rate']:>7.3f}{m:>9.3f}{f'[{m - half:.2f}, {m + half:.2f}]':>16}"
          f"{pk:>9.3f}{s_iid.mean():>9.3f}{s_mod.mean():>9.3f}")
