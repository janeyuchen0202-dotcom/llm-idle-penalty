"""暖機檢查（2026-10-10）：「丟掉每組前 1/3」夠不夠？用 MSER-5 規則檢查

做法：
- 每組實驗照公式 (1) 重建等待時間 W（照送出順序）
- MSER-5（White 1997 的 MSER，加上每 5 筆先平均一次）：把 W 每 5 筆分成一批、取批平均 Z_1..Z_m；
  對每個候選截斷點 d（0 ≤ d ≤ m/2），算 MSER(d) = Σ_{j>d} (Z_j − Z̄_d)² / (m − d)²，
  取讓它最小的 d*，建議丟掉前 5·d* 筆
- 和論文的做法（丟掉前 n/3 筆）比：MSER-5 建議的點在 1/3 之內 → 1/3 夠；
  超過 1/3 → 算「改丟到 MSER-5 的點」之後，平均等待時間會差多少

用法：python 稽核/check_warmup.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute as rc  # noqa: E402  只借用讀檔和公式 (1)

RUNS = (rc.A_RUNS + rc.C_RUNS
        + ["rep2_rho30", "rep2_rho50", "rep2_rho70", "rep2_rho85"]       # 3B 第三組亂數
        + ["b7_rho30", "b7_rho50", "b7_rho70"]                           # 7B 分開放
        + ["b7v2_rho30", "b7v2_rho50", "b7v2_rho70"]                     # 7B 全放顯示卡
        + ["m1_rho30", "m1_rho50", "m1_rho70"]                           # M1
        + ["win_rho30", "win_rho50", "win_rho70"])                       # HP


def mser5(w):
    m = len(w) // 5
    z = w[:m * 5].reshape(m, 5).mean(axis=1)
    best, dbest = np.inf, 0
    for d in range(0, m // 2 + 1):
        rest = z[d:]
        v = ((rest - rest.mean()) ** 2).sum() / (m - d) ** 2
        if v < best:
            best, dbest = v, d
    return 5 * dbest, dbest == m // 2


def main():
    ok, extra = 0, []
    print(f"{'組別':<16}{'n':>5}{'丟 1/3':>8}{'MSER-5':>8}  判定")
    for name in RUNS:
        d = rc.run(name)
        w = d["W"]
        k = d["n"] // 3
        cut, at_edge = mser5(w)
        if cut <= k:
            ok += 1
            verdict = "1/3 夠"
        else:
            change = w[cut:].mean() / w[k:].mean() - 1
            extra.append((name, cut - k, change))
            verdict = f"多丟 {cut - k} 筆 → 平均等待 {change:+.1%}" + ("（截斷點在上限，這組可能沒到穩態）" if at_edge else "")
        print(f"{name:<16}{d['n']:>5}{k:>8}{cut:>8}  {verdict}")
    print(f"\n共 {len(RUNS)} 組：{ok} 組在前 1/3 之內；{len(extra)} 組 MSER-5 建議多丟 "
          f"{min(e[1] for e in extra)}–{max(e[1] for e in extra)} 筆，平均等待改變 "
          f"{min(e[2] for e in extra):+.1%} 到 {max(e[2] for e in extra):+.1%}"
          if extra else f"\n共 {len(RUNS)} 組：全部在前 1/3 之內")


if __name__ == "__main__":
    main()
