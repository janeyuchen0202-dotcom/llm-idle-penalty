"""δ(I) 分更細的段，能不能解釋 3B 事前預測偏高（2026-10-20，事後分析，不算預測）

背景：3B 的 9 組事前預測，實測平均比預測低約 5%；改用各組自己的服務時間後，8 組 128 token 的偏高從 4.3% 降到 3.4%，其餘原因不明
想法：δ 原本分四段（<1、1–2、2–5、≥5 秒）。2–5 秒這段裡 δ 從約 0.17 秒升到約 0.3 秒；負載高的組閒置時間短，
     落在這段的多半靠近 2 秒（δ 較小），但四段模型給的是整段的平均，可能讓負載高的組高估
做法：用和論文相同的 17 組估 δ（每組自己的排隊服務時間當基準），比較四段和十段（0.5、1、1.5、2、2.5、3、4、5、8 秒）
  1. 17 組（樣本內，和論文 V-B 相同的比較）
  2. seed 2 的 4 組（樣本外：參數只來自 17 組；服務時間樣本和事前預測相同）
  3. 8 組 128 token 的事前預測組（期中 4 組＋seed 2 的 4 組），改用各組自己的服務時間（重現 4.3% → 3.4% 那個比較）
用法：python analyze_finer_bins.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = 2000
E4 = [1.0, 2.0, 5.0]
E10 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]
RUNS17 = ["main2_rho30", "main2_rho50", "long70_rho70", "long85_rho85", "mix_A_rho50", "mix_A_rho75", "var_C_rho50",
          "var_C_rho70", "rep_rho30", "rep_rho50", "rep_rho70", "rep_rho85", "var2_A_rho50", "var2_A_rho75",
          "var2_C_rho50", "var2_C_rho70", "confirm_C_rho75"]
A12 = ["main2_rho30", "main2_rho50", "long70_rho70", "long85_rho85", "mix_A_rho50", "mix_A_rho75",
       "rep_rho30", "rep_rho50", "rep_rho70", "rep_rho85", "var2_A_rho50", "var2_A_rho75"]      # 和 analyze_seed2.py 相同
MID_A = ["rep_rho30", "rep_rho50", "var2_A_rho50", "var2_A_rho75"]                            # 期中預測裡 128 token 的 4 組
SEED2 = [r["tag"] for r in json.load(open(os.path.join(HERE, "plan_seed2.json"), encoding="utf-8"))["runs"]]


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


D = {n: load(n) for n in set(RUNS17 + SEED2)}

# 和 analyze_strengthen.py 第三節相同：每組、每種輸出長度，排隊後開始的服務時間當基準，閒置後到達的多出多少
pools, gaps, res = {}, [], []
for name in RUNS17:
    a, W, S, gap, cls = D[name]
    for k in sorted(set(cls)):
        q = (W > 0) & (cls == k)
        pools.setdefault(k, []).extend(S[q])
        idle = (W == 0) & (cls == k)
        idle[0] = False
        gaps.extend(gap[idle])
        res.extend(S[idle] - S[q].mean())
pools = {k: np.array(v) for k, v in pools.items()}
gaps, res = np.array(gaps), np.array(res)


def deltas(edges):
    b = np.digitize(gaps, edges)
    return np.array([res[b == i].mean() for i in range(len(edges) + 1)]), np.bincount(b, minlength=len(edges) + 1), \
        np.array([res[b == i].std() / np.sqrt(max(1, np.sum(b == i))) for i in range(len(edges) + 1)])


def sim(a, cls, dv, edges, pool_of, seed):
    rng = np.random.default_rng(seed)
    n, k = len(a), len(a) // 3
    base = np.empty((R, n))
    for key in sorted(set(cls)):
        idx = np.where(cls == key)[0]
        base[:, idx] = rng.choice(pool_of(key), size=(R, len(idx)))
    c_prev, acc = np.full(R, -np.inf), np.zeros(R)
    for i in range(n):
        start = np.maximum(a[i], c_prev)
        w = start - a[i]
        pen = np.where(w == 0, dv[np.digitize(a[i] - c_prev, edges)], 0.0)
        c_prev = start + base[:, i] + pen
        if i >= k:
            acc += w
    return acc / (n - k)


def evaluate(names, pool_mode):
    out = []
    for j, name in enumerate(names):
        a, W, S, gap, cls = D[name]
        m = W[len(a) // 3:].mean()
        lh = (len(a) - len(a) // 3 - 1) / (a[-1] - a[len(a) // 3])
        row = {"run": name, "meas": m, "rho_hat": lh * S[W > 0].mean()}
        for lab, edges in (("4段", E4), ("10段", E10)):
            dv = deltas(edges)[0]
            if pool_mode == "global":
                pf = lambda key: pools[key]
            elif pool_mode == "A12":
                a12 = np.concatenate([D[n][2][D[n][1] > 0] for n in A12])
                pf = lambda key, a12=a12: a12
            else:                                                           # 各組自己的排隊服務時間
                pf = lambda key, S=S, W=W, cls=cls: S[(W > 0) & (cls == key)]
            s = sim(a, cls, dv, edges, pf, 100 + j)
            lo, hi = np.percentile(s, [2.5, 97.5])
            row[lab] = (s.mean(), lo, hi)
        out.append(row)
    return out


def report(title, rows):
    print(f"\n== {title}")
    print(f"  {'組別':<16}{'實際ρ':>7}{'實測':>8}{'4段預測':>9}{'偏高':>7}{'10段預測':>10}{'偏高':>7}")
    summ = {}
    for r in rows:
        p4, p10 = r["4段"][0], r["10段"][0]
        print(f"  {r['run']:<16}{r['rho_hat']:>7.2f}{r['meas']:>8.3f}{p4:>9.3f}{100 * (p4 / r['meas'] - 1):>+6.1f}%"
              f"{p10:>10.3f}{100 * (p10 / r['meas'] - 1):>+6.1f}%")
    for lab in ("4段", "10段"):
        b = np.mean([r[lab][0] / r["meas"] - 1 for r in rows]) * 100
        mae = np.mean([abs(r[lab][0] / r["meas"] - 1) for r in rows]) * 100
        ins = sum(1 for r in rows if r[lab][1] <= r["meas"] <= r[lab][2])
        summ[lab] = {"bias_pct": b, "mae_pct": mae, "inside": ins, "n": len(rows)}
        print(f"  {lab}：平均偏高 {b:+.1f}%，平均絕對誤差 {mae:.1f}%，落在 95% 範圍內 {ins}/{len(rows)}")
    return summ


d4, n4, se4 = deltas(E4)
d10, n10, se10 = deltas(E10)
lab4 = ["<1", "1–2", "2–5", "≥5"]
lab10 = ["<0.5", "0.5–1", "1–1.5", "1.5–2", "2–2.5", "2.5–3", "3–4", "4–5", "5–8", "≥8"]
print("δ（17 組，閒置後到達的請求比同組排隊請求多出的時間）")
print("  四段：" + "、".join(f"{l} 秒 {v:+.3f}（{c}）" for l, v, c in zip(lab4, d4, n4)))
print("  十段：" + "、".join(f"{l} 秒 {v:+.3f}±{s:.3f}（{c}）" for l, v, s, c in zip(lab10, d10, se10, n10)))
out = {"delta4": dict(zip(lab4, map(float, d4))), "delta10": dict(zip(lab10, map(float, d10))), "n10": dict(zip(lab10, map(int, n10)))}
out["in_sample_17"] = report("1. 17 組（樣本內）", evaluate(RUNS17, "global"))
out["seed2_out_of_sample"] = report("2. seed 2 的 4 組（樣本外；服務時間樣本和事前預測相同：12 組 128 token 的排隊請求）", evaluate(SEED2, "A12"))
out["eight_A_own_service"] = report("3. 8 組 128 token 的事前預測組，改用各組自己的服務時間", evaluate(MID_A + SEED2, "own"))
json.dump(out, open(os.path.join(HERE, "out", "finer_bins_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n→ out/finer_bins_summary.json")
