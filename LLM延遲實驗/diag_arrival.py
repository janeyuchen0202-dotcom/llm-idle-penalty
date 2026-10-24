"""期中診斷：重跑偏低是不是因為「這組亂數實際的到達率」和名目 λ 不同，以及單次實驗本來就有多大的起伏。

對每一組（新、舊）：
  λ̂      分析區間（後 2/3）實際的到達率
  PK名目  用名目 λ 算的 P-K（原本的做法）
  PK實際  用 λ̂ 算的 P-K
  批次CI  把等待時間切成 10 段連續批次算的 95% CI（考慮前後請求的相關）
  模擬範圍 M/G/1 模擬：Poisson(名目 λ)、服務時間從實測抽樣、同樣筆數與暖機，單次實驗平均等待的 95% 範圍
  條件模擬 到達時間用這次實際的，服務時間從實測抽樣：實測平均落在模擬分布的第幾百分位
"""
import csv
import os

import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
R = 2000
rng = np.random.default_rng(0)

PAIRS = [("rep_rho70", "long70_rho70", 0.34988), ("rep_rho85", "long85_rho85", 0.42485),
         ("var2_C_rho70", "var_C_rho70", 0.330), ("confirm_C_rho75", None, 0.35),
         ("rep_rho30", "main2_rho30", 0.14995), ("rep_rho50", "main2_rho50", 0.24991),
         ("var2_A_rho50", "mix_A_rho50", 0.259), ("var2_A_rho75", "mix_A_rho75", 0.389),
         ("var2_C_rho50", "var_C_rho50", 0.233)]
done = {r["tag"] for r in csv.DictReader(open(os.path.join(LAB, "out", "strengthen_runlog.csv"), encoding="utf-8"))}


def load(name):
    rows = sorted([r for r in csv.DictReader(open(os.path.join(LAB, "out", name + ".csv"), encoding="utf-8"))
                   if r["status"] == "ok"], key=lambda r: float(r["t_send"]))
    a = np.array([float(r["t_send"]) for r in rows])
    c = np.array([float(r["t_done"]) for r in rows])
    prev = np.empty(len(c)); _cp = 0.0    # 10/22 改：照做完的順序重建（一次只處理一個請求，做完的順序就是處理的順序；原本照送出順序）
    for _j in np.argsort(c, kind="stable"):
        prev[_j] = _cp; _cp = c[_j]
    start = np.maximum(a, prev)
    return a, start - a, c - start


def lindley(ia, s):
    """ia、s 形狀 (R, n)；ia[:, i] = 第 i 個到第 i+1 個的間隔"""
    w = np.zeros_like(s)
    for i in range(s.shape[1] - 1):
        w[:, i + 1] = np.maximum(0.0, w[:, i] + s[:, i] - ia[:, i])
    return w


def one(name, lam):
    a, W, S = load(name)
    n, k = len(a), len(a) // 3
    aw, Ww, Sw = a[k:], W[k:], S[k:]
    lam_hat = (len(aw) - 1) / (aw[-1] - aw[0])
    es, es2 = Sw.mean(), (Sw ** 2).mean()
    pk = lambda l: l * es2 / (2 * (1 - l * es))
    m = Ww.mean()
    bm = np.array([b.mean() for b in np.array_split(Ww, 10)])
    half = 2.262 * bm.std(ddof=1) / np.sqrt(10)
    # 無條件模擬：Poisson(名目 λ)
    ia = rng.exponential(1 / lam, size=(R, n))
    sims = lindley(ia, rng.choice(Sw, size=(R, n)))[:, k:].mean(axis=1)
    lo, hi = np.percentile(sims, [2.5, 97.5])
    # 條件模擬：到達時間固定為這次實際的
    ia_c = np.broadcast_to(np.diff(a, append=a[-1] + 1), (R, n))
    sims_c = lindley(ia_c, rng.choice(Sw, size=(R, n)))[:, k:].mean(axis=1)
    pct = (sims_c < m).mean() * 100
    print(f"  {name:<16} n={len(Ww):>3}  λ̂/λ={lam_hat / lam:.3f}  ρ名目={lam * es:.3f} ρ實際={lam_hat * es:.3f}  "
          f"W_q={m:.3f}  PK名目={pk(lam):.3f}({100 * (m / pk(lam) - 1):+.0f}%)  "
          f"PK實際={pk(lam_hat):.3f}({100 * (m / pk(lam_hat) - 1):+.0f}%)  "
          f"批次CI=[{m - half:.2f},{m + half:.2f}]  模擬範圍=[{lo:.2f},{hi:.2f}]  條件模擬百分位={pct:.0f}")


for new, old, lam in PAIRS:
    if new not in done:
        continue
    print(new)
    one(new, lam)
    if old:
        one(old, lam)
