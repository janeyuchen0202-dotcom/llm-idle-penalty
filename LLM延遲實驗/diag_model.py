"""期中診斷 3：「閒置後第一個請求較慢」能不能解釋實測等待時間？並為還沒跑的組別先寫下預測。

模型：一般服務時間從「排隊後開始」的實測值抽樣；到達時伺服器空著的請求，再加上依閒置時間而定的額外時間 δ。
到達時間用該組實際的（還沒跑的組別用 seed 1 重建，和 load_gen.py 完全相同的抽法）。
"""
import csv
import os
import random

import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
R = 2000
rng = np.random.default_rng(7)
EDGES = [1.0, 2.0, 5.0]
LABELS = ["<1s", "1–2s", "2–5s", "≥5s"]
DONE = [("main2_rho30", 0.14995), ("main2_rho50", 0.24991), ("long70_rho70", 0.34988),
        ("long85_rho85", 0.42485), ("mix_A_rho50", 0.259), ("mix_A_rho75", 0.389),
        ("var_C_rho50", 0.233), ("var_C_rho70", 0.330),
        ("rep_rho70", 0.34988), ("rep_rho85", 0.42485), ("var2_C_rho70", 0.330), ("confirm_C_rho75", 0.35)]
TODO = [("rep_rho30", 0.14995, 150, False, (1.8966, 3.7051)), ("rep_rho50", 0.24991, 150, False, (1.8966, 3.7051)),
        ("var2_A_rho50", 0.259, 600, False, (1.8950, 3.6126)), ("var2_A_rho75", 0.389, 600, False, (1.8950, 3.6126)),
        ("var2_C_rho50", 0.233, 600, True, (1.8906, 5.3243))]


def load(name):
    rows = sorted([r for r in csv.DictReader(open(os.path.join(LAB, "out", name + ".csv"), encoding="utf-8"))
                   if r["status"] == "ok"], key=lambda r: float(r["t_send"]))
    a = np.array([float(r["t_send"]) for r in rows])
    c = np.array([float(r["t_done"]) for r in rows])
    cls = np.array(["A" if int(r["out_tokens"]) == 128 else f"C{r['out_tokens']}" for r in rows])
    prev = np.empty(len(c)); _cp = -np.inf    # 10/22 改：照做完的順序重建（一次只處理一個請求，做完的順序就是處理的順序；原本照送出順序）
    for _j in np.argsort(c, kind="stable"):
        prev[_j] = _cp; _cp = c[_j]
    start = np.maximum(a, prev)
    return a, start - a, c - start, a - prev, cls


# ---- 1) 服務時間模型 ----
pools, gaps, res = {}, [], []
for name, _ in DONE:
    a, W, S, gap, cls = load(name)
    for k in set(cls):
        q = (W > 0) & (cls == k)
        pools.setdefault(k, []).extend(S[q])
        idle = (W == 0) & (cls == k)
        idle[0] = False
        gaps.extend(gap[idle])
        res.extend(S[idle] - S[q].mean())
pools = {k: np.array(v) for k, v in pools.items()}
gaps, res = np.array(gaps), np.array(res)
b = np.digitize(gaps, EDGES)
dvals = np.array([res[b == i].mean() for i in range(4)])
print("閒置後第一個請求多出的服務時間 δ（依閒置多久）：")
for i in range(4):
    print(f"  {LABELS[i]:>5}: n={np.sum(b == i):>4}  δ = {dvals[i]:+.3f} 秒  (SE {res[b == i].std() / np.sqrt(np.sum(b == i)):.3f})")
print("  一般服務時間平均：" + "，".join(f"{k} {v.mean():.3f} 秒（n={len(v)}）" for k, v in sorted(pools.items())))


def sim(a, cls, penalty=True):
    n, k = len(a), len(a) // 3
    base = np.empty((R, n))
    for key in set(cls):
        idx = np.where(cls == key)[0]
        base[:, idx] = rng.choice(pools[key], size=(R, len(idx)))
    c_prev, acc = np.full(R, -np.inf), np.zeros(R)
    for i in range(n):
        start = np.maximum(a[i], c_prev)
        w = start - a[i]
        pen = np.where(w == 0, dvals[np.digitize(a[i] - c_prev, EDGES)], 0.0) if penalty else 0.0
        c_prev = start + base[:, i] + pen
        if i >= k:
            acc += w
    return acc / (n - k)


# ---- 2) 已完成的組別：模型 vs 實測 ----
print(f"\n{'組別':<16}{'n':>4}{'實測W_q':>9}{'P-K':>8}{'偏差':>7}{'模型預測':>9}{'模型95%範圍':>16}{'實測百分位':>10}")
for name, lam in DONE:
    a, W, S, gap, cls = load(name)
    k = len(a) // 3
    m, Sw = W[k:].mean(), S[k:]
    pk = lam * (Sw ** 2).mean() / (2 * (1 - lam * Sw.mean()))
    s = sim(a, cls)
    lo, hi = np.percentile(s, [2.5, 97.5])
    print(f"{name:<16}{len(a) - k:>4}{m:>9.3f}{pk:>8.3f}{100 * (m / pk - 1):>+6.0f}%{s.mean():>9.3f}"
          f"{f'[{lo:.2f}, {hi:.2f}]':>16}{(s < m).mean() * 100:>9.0f}")


# ---- 3) 還沒跑的組別：用 seed 1 重建到達時間 ----
def schedule(lam, n, mix):
    r, t, a, cls = random.Random(1), 0.0, [], []
    for _ in range(n):
        t += r.expovariate(lam)
        r.choices([0], weights=[1.0], k=1)
        mt = r.choices([32, 224], weights=[0.5, 0.5], k=1)[0] if mix else r.choices([128], weights=[1.0], k=1)[0]
        a.append(t)
        cls.append("A" if mt == 128 else f"C{mt}")
    return np.array(a), np.array(cls)


# 檢查重建是否正確：和已跑完的 seed 1 組別比對
for name, lam, n, mix in [("rep_rho85", 0.42485, 600, False), ("var2_C_rho70", 0.330, 600, True)]:
    rows = sorted(csv.DictReader(open(os.path.join(LAB, "out", name + ".csv"), encoding="utf-8")),
                  key=lambda r: int(r["req_id"]))
    a_s, c_s = schedule(lam, n, mix)
    d = max(abs(float(r["t_scheduled"]) - x) for r, x in zip(rows, a_s))
    same = all((("A" if int(r["out_tokens"]) == 128 else f"C{r['out_tokens']}") == k)
               for r, k in zip(rows, c_s) if r["status"] == "ok")
    print(f"\n重建檢查 {name}: 排定時間最大差 {d:.4f} 秒，輸出長度全部相同 = {same}")

print(f"\n{'待跑組別':<14}{'λ̂/λ':>7}{'P-K名目':>9}{'P-K實際λ':>10}{'模型預測':>9}{'模型95%範圍':>16}{'無δ模擬':>9}")
for name, lam, n, mix, (es, es2) in TODO:
    a, cls = schedule(lam, n, mix)
    k = n // 3
    lh = (n - k - 1) / (a[-1] - a[k])
    pk = lambda l: l * es2 / (2 * (1 - l * es))
    s, s0 = sim(a, cls), sim(a, cls, penalty=False)
    lo, hi = np.percentile(s, [2.5, 97.5])
    print(f"{name:<14}{lh / lam:>7.3f}{pk(lam):>9.3f}{pk(lh):>10.3f}{s.mean():>9.3f}{f'[{lo:.2f}, {hi:.2f}]':>16}"
          f"{s0.mean():>9.3f}")
