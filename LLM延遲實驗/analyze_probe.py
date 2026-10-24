"""閒置效應測試的分析（2026-10-06）

輸入：out/<tag>.csv（probe_idle.py 的結果）、out/<tag>_gpu.csv（nvidia-smi 紀錄）
輸出：每種閒置時間 g、兩種條件（A 一般／K 保溫）下多出的服務時間 δ(g) = S(g) − S(0)，
      以及多出的時間落在第一個 token 之前（TTFT）還是之後；顯示卡在請求送出當下的狀態。

用法：python analyze_probe.py [tag]
"""
import csv
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "probe_idle"
rng = np.random.default_rng(0)

rows = list(csv.DictReader(open(os.path.join(HERE, "out", f"{TAG}.csv"), encoding="utf-8")))
for r in rows:
    for k in ("gap", "idle_before", "service", "ttft", "wall_send"):
        r[k] = float(r[k])


def boot_diff(x, y, b=4000):
    x, y = np.asarray(x), np.asarray(y)
    d = rng.choice(x, (b, len(x))).mean(1) - rng.choice(y, (b, len(y))).mean(1)
    return np.percentile(d, [2.5, 97.5])


gaps = sorted({r["gap"] for r in rows})
CONDS = [c for c in ("A", "K", "H") if any(r["cond"] == c for r in rows)]
summary = []
print(f"{'條件':<4}{'g(秒)':>6}{'n':>4}{'實際閒置':>9}{'服務時間':>9}{'δ':>8}{'95% CI':>18}{'TTFT':>8}{'δ_TTFT':>8}{'δ_其餘':>8}")
for cond in CONDS:
    base = [r for r in rows if r["cond"] == cond and r["gap"] == 0]
    s0, t0 = np.mean([r["service"] for r in base]), np.mean([r["ttft"] for r in base])
    for g in gaps:
        sel = [r for r in rows if r["cond"] == cond and r["gap"] == g]
        s = np.array([r["service"] for r in sel])
        t = np.array([r["ttft"] for r in sel])
        lo, hi = boot_diff(s, [r["service"] for r in base]) if g > 0 else (0.0, 0.0)
        d, dt = s.mean() - s0, t.mean() - t0
        summary.append(dict(cond=cond, gap=g, n=len(sel), idle=np.mean([r["idle_before"] for r in sel]),
                            service=s.mean(), delta=d, lo=lo, hi=hi, ttft=t.mean(), d_ttft=dt, d_rest=d - dt))
        print(f"{cond:<4}{g:>6.1f}{len(sel):>4}{summary[-1]['idle']:>9.2f}{s.mean():>9.3f}{d:>+8.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>18}{t.mean():>8.3f}{dt:>+8.3f}{d - dt:>+8.3f}")

with open(os.path.join(HERE, "out", f"{TAG}_summary.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0]))
    w.writeheader()
    for r in summary:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

# ---- 對照事前預期 ----
big = lambda cond: np.mean([r["delta"] for r in summary if r["cond"] == cond and r["gap"] >= 3])
NAME = {"A": "一般", "K": "保溫（1 token）", "H": "重保溫（讀完整 prompt）"}
print()
for c in CONDS:
    print(f"{NAME[c]}條件 g≥3 秒的平均 δ = {big(c):+.3f} 秒")
smalls = [r["delta"] for r in summary if r["cond"] == "A" and 0 < r["gap"] <= 1]
if smalls:
    print(f"一般條件 g≤1 秒的最大 δ = {np.max(smalls):+.3f} 秒")

# ---- 顯示卡狀態 ----
gpu_path = os.path.join(HERE, "out", f"{TAG}_gpu.csv")
if os.path.exists(gpu_path):
    g = []
    for line in open(gpu_path, encoding="utf-8", errors="replace"):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 6:
            continue
        try:
            ts = datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            g.append((ts, p[1], float(p[2]), float(p[4])))
        except ValueError:
            continue
    g.sort()
    gt = np.array([x[0] for x in g])
    gp = np.array([x[1] for x in g])
    gc = np.array([x[2] for x in g])
    print(f"\n顯示卡紀錄 {len(g)} 筆，間隔中位數 {np.median(np.diff(gt)) * 1000:.0f} 毫秒")
    print(f"{'條件':<4}{'g(秒)':>6}{'送出時在P8的比例':>16}{'送出時平均時脈(MHz)':>20}")
    for cond in CONDS:
        for gg in gaps:
            sel = [r for r in rows if r["cond"] == cond and r["gap"] == gg]
            idx = [max(0, np.searchsorted(gt, r["wall_send"]) - 1) for r in sel]
            p8 = np.mean([gp[i] == "P8" for i in idx])
            clk = np.mean([gc[i] for i in idx])
            print(f"{cond:<4}{gg:>6.1f}{p8:>16.2f}{clk:>20.0f}")
    # 一般條件下，從上一個請求結束到降到 P8 要多久
    drops = []
    for r in rows:
        if r["cond"] != "A" or r["gap"] < 3:
            continue
        t_end = r["wall_send"] - r["idle_before"]
        j = np.searchsorted(gt, t_end)
        k = next((i for i in range(j, len(gt)) if gt[i] < r["wall_send"] and gp[i] == "P8"), None)
        if k is not None:
            drops.append(gt[k] - t_end)
    if drops:
        print(f"\n一般條件、閒置 ≥3 秒：上一個請求結束後 {np.median(drops):.2f} 秒（中位數，"
              f"範圍 {np.min(drops):.2f}–{np.max(drops):.2f}）降到 P8，n={len(drops)}")
