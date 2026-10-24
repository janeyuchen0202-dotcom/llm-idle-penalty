"""論文 v0.2 與研究計畫書的圖（2026-10-06）

論文
  fig1_idle      (a) 閒置時間 vs 多出的服務時間：不保溫／輕保溫（1 token）／重保溫（讀完整 prompt），
                     加上排隊實驗的分段估計
                 (b) 閒置 8 秒期間顯示卡時脈的中位數（三種條件）
  fig2_accuracy  每組實驗「實測 / 預測」：標準 M/G/1 vs 閒置效應模型（都以該組實際到達時間為條件）
研究計畫書
  LLM延遲實驗/figure_column_v2  (a) 同 fig1(a)，(b) 同 fig2；欄寬 3.35 × 3.9 吋

資料：LLM延遲實驗/out/ 底下的 probe_idle_3b*（analyze_probe.py 的 _summary.csv）、
      strengthen_summary.csv、idle_penalty_bins.csv
用法：python make_paper_figs.py
"""
import csv
import os
import shutil
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "LLM延遲實驗")
OUT = os.path.join(LAB, "out")
# 條件 → (資料來源的 tag, 線型, 標記, 填色, 圖例)
SERIES = [("A", "probe_idle_3b", "-", "o", "black", "No keep-warm"),
          ("K", "probe_idle_3b", ":", "^", "white", "Light keep-warm (1 token every 0.5 s)"),
          ("H", "probe_idle_3b_heavy", "--", "s", "white", "Heavy keep-warm (838-token prompt every 0.5 s)")]

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 6.8, "axes.linewidth": 0.6, "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42})   # 10/11：PDF 改嵌 TrueType（Type 3 字型 IEEE PDF eXpress 可能不收）；PNG 不受影響


def save(fig, path):
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=300)
    print(f"→ {os.path.relpath(path, HERE)}.png / .pdf")


SHORT = {"Light keep-warm (1 token every 0.5 s)": "Light keep-warm (1 token / 0.5 s)",
         "Heavy keep-warm (838-token prompt every 0.5 s)": "Heavy keep-warm (838 tokens / 0.5 s)"}


def panel_idle(ax, top=0.64, compact=False):
    """compact：論文 Fig. 1 用（10/14）。圖例縮短、不列灰色橫條（圖說已說明），圖例才不會壓到資料"""
    for cond, tag, ls, mk, mfc, lab in SERIES:
        rows = [r for r in csv.DictReader(open(os.path.join(OUT, f"{tag}_summary.csv"), encoding="utf-8"))
                if r["cond"] == cond]
        g = np.array([float(r["gap"]) for r in rows])
        y = np.array([float(r["delta"]) for r in rows])
        lo = np.array([float(r["lo"]) for r in rows])
        hi = np.array([float(r["hi"]) for r in rows])
        ax.errorbar(g, y, yerr=[y - lo, hi - y], fmt=mk + ls, color="black", mfc=mfc, ms=3.6, lw=0.9,
                    capsize=1.8, elinewidth=0.6, label=SHORT.get(lab, lab) if compact else lab)
    bins = np.loadtxt(os.path.join(OUT, "idle_penalty_bins.csv"), delimiter=",", skiprows=1)
    edges = [0, 1, 2, 5, 8.6]
    for i in range(4):
        ax.plot([edges[i], edges[i + 1]], [bins[i, 0]] * 2, color="0.6", lw=2.4, solid_capstyle="butt", zorder=0,
                label="Queueing runs (17 runs)" if i == 0 and not compact else None)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlim(-0.2, 8.6)
    ax.set_ylim(-0.05, top)
    ax.set_xlabel("Idle time before the request (s)")
    ax.set_ylabel("Extra service time (s)")
    if compact:
        ax.legend(frameon=False, loc="upper right", handlelength=2.4, fontsize=6.5, labelspacing=0.35,
                  borderaxespad=0.3)
    else:
        ax.legend(frameon=False, loc="upper right", handlelength=2.4, fontsize=6.5)
    ax.grid(alpha=0.3, lw=0.4)


def gpu_trace(tag, cond, gap=8.0):
    rows = list(csv.DictReader(open(os.path.join(OUT, f"{tag}.csv"), encoding="utf-8")))
    g = []
    for line in open(os.path.join(OUT, f"{tag}_gpu.csv"), encoding="utf-8", errors="replace"):
        p = [s.strip() for s in line.split(",")]
        try:
            g.append((datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(p[2])))
        except (ValueError, IndexError):
            continue
    g.sort()
    gt, gc = np.array([x[0] for x in g]), np.array([x[1] for x in g])
    grid = np.arange(0, gap + 0.01, 0.1)
    tr = []
    for r in rows:
        if r["cond"] != cond or float(r["gap"]) != gap:
            continue
        # 對齊到「上一個正式請求結束」的時刻：送出時間 − 設定的閒置時間 g
        t0 = float(r["wall_send"]) - gap
        tr.append(np.interp(t0 + grid, gt, gc))
    return grid, np.median(np.array(tr), 0), len(tr)


def panel_clock(ax, compact=False):
    for cond, tag, ls, mk, mfc, lab in SERIES:
        grid, med, n = gpu_trace(tag, cond)
        ax.plot(grid, med, ls, color="black", lw=1.0, label=f"{lab.split(' (')[0]} (median of {n})")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 3200)
    ax.set_xlabel("Time since the previous measured request ended (s)")
    ax.set_ylabel("GPU clock (MHz)")
    if compact:                          # 圖例放在虛線（2,500 MHz）和點線之間的空白，不碰到線
        ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.66, 0.53), fontsize=6.5, labelspacing=0.3)
    else:
        ax.legend(frameon=False, loc="center right", fontsize=6.5)
    ax.grid(alpha=0.3, lw=0.4)


def panel_ratio(ax):
    rows = list(csv.DictReader(open(os.path.join(OUT, "strengthen_summary.csv"), encoding="utf-8")))
    for r in rows:
        x = float(r["rho_hat"])
        ax.plot([x, x], [float(r["W"]) / float(r["iid_mean"]), float(r["W"]) / float(r["mod_mean"])],
                color="0.7", lw=0.6, zorder=1)
    for key, mfc in (("iid_mean", "white"), ("mod_mean", "black")):
        for grp, mk in (("A", "o"), ("C", "^")):
            sel = [r for r in rows if (r["group"] == "C") == (grp == "C")]
            ax.scatter([float(r["rho_hat"]) for r in sel], [float(r["W"]) / float(r[key]) for r in sel],
                       marker=mk, s=15, facecolors=mfc, edgecolors="black", linewidths=0.7, zorder=2)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", mfc="white", mec="black", ms=4.2, label="Standard M/G/1"),
         Line2D([], [], ls="", marker="o", mfc="black", mec="black", ms=4.2, label="Idle-dependent setup"),
         Line2D([], [], ls="", marker="o", mfc="0.6", mec="0.6", ms=4.2, label="128 tokens"),
         Line2D([], [], ls="", marker="^", mfc="0.6", mec="0.6", ms=4.2, label="32/224 tokens")]
    ax.legend(handles=h, frameon=False, loc="upper center", ncol=2, handletextpad=0.2, columnspacing=1.2,
              fontsize=6.5)
    ax.axhline(1, color="black", lw=0.6)
    ax.axhspan(0.95, 1.05, color="0.92", lw=0, zorder=0)
    ax.set_xlabel(r"Utilization in the analyzed window, $\hat{\lambda}E[S]$")
    ax.set_ylabel(r"Measured / predicted $W_q$")
    ax.set_ylim(0.8, 1.3)
    ax.grid(alpha=0.3, lw=0.4)


def tag_panel(ax, s):
    ax.text(0.015, 0.97, s, transform=ax.transAxes, ha="left", va="top")


def main():
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 2.95), gridspec_kw={"height_ratios": [1.2, 1]})
    panel_idle(a, compact=True)
    tag_panel(a, "(a)")
    panel_clock(b, compact=True)
    tag_panel(b, "(b)")
    fig.tight_layout(pad=0.3, h_pad=0.6)
    save(fig, os.path.join(HERE, "fig1_idle"))

    fig, ax = plt.subplots(figsize=(3.45, 1.85))
    panel_ratio(ax)
    fig.tight_layout(pad=0.3)
    save(fig, os.path.join(HERE, "fig2_accuracy"))

    fig, (a, b) = plt.subplots(2, 1, figsize=(3.35, 3.4))
    panel_idle(a, top=0.9)
    tag_panel(a, "(a)")
    panel_ratio(b)
    tag_panel(b, "(b)")
    fig.tight_layout(pad=0.3, h_pad=0.5)
    base = os.path.join(LAB, "figure_column_v2")
    save(fig, base)
    dst = os.path.join(HERE, "..", "備審")
    if os.path.isdir(dst):          # 10/24：GitHub 附錄沒有 備審 資料夾，沒有就不複製
        for ext in ("png", "pdf"):
            shutil.copy(f"{base}.{ext}", os.path.join(dst, f"figure_column_v2.{ext}"))
        print("→ 備審/figure_column_v2.png / .pdf（副本）")


if __name__ == "__main__":
    main()
