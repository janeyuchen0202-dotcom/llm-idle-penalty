"""單欄寬度版本的合併圖（研究計畫書兩欄排版用）

(a) 平均等待時間 vs 利用率，與 P-K 理論比較（上）
(b) 相同平均服務時間下，服務時間變異的影響（下）

設計重點：
- 寬 3.35 英吋＝官方範本一欄的寬度，放進文件時不縮放，字級 7–7.5pt 就是印出來的大小
- 黑白可讀：靠標記形狀與線型區分，不靠顏色（範本要求以黑白列印為前提）
- 資料與計算沿用 make_combined_figure.py，數字與合併圖一致

用法：python make_column_figure.py
"""
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from make_combined_figure import HERE, MAIN, VAR, stats


def pk_curve(xs, es, es2):
    return [(r / es) * es2 / (2 * (1 - r)) for r in xs]


def main():
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.35, 3.9), dpi=300)
    xs = [i / 200 for i in range(1, 171)]

    # ---- (a) 主實驗 ----
    pts = [stats(p, l) for p, l in MAIN]
    es = st.mean([p["es"] for p in pts]); es2 = st.mean([p["es2"] for p in pts])
    ax1.plot(xs, pk_curve(xs, es, es2), "-", color="black", lw=1.0)
    ax1.errorbar([p["rho"] for p in pts], [p["wq"] for p in pts],
                 yerr=[[p["wq"] - p["lo"] for p in pts], [p["hi"] - p["wq"] for p in pts]],
                 fmt="o", color="black", ms=4, capsize=2.5, lw=0.8, ls="none",
                 markerfacecolor="white", markeredgewidth=0.8)
    ax1.legend(handles=[
        Line2D([], [], color="black", lw=1.0, label="P-K theory"),
        Line2D([], [], color="black", marker="o", ls="none", ms=4, mfc="white", mew=0.8,
               label="Measured (95% CI)")], frameon=False, loc="upper left")
    ax1.set_title("(a) Waiting time vs. load", loc="left")

    # ---- (b) 變異 ----
    handles = []
    for key, name, mk, ls in (("A", "A: fixed 128 tokens", "o", "-"),
                              ("C", "C: 32 or 224 tokens, same mean", "^", "--")):
        ps = sorted([stats(p, l) for p, l in VAR[key]], key=lambda d: d["rho"])
        es = st.mean([p["es"] for p in ps]); es2 = st.mean([p["es2"] for p in ps])
        ax2.plot(xs, pk_curve(xs, es, es2), ls, color="0.45", lw=1.0)
        ax2.errorbar([p["rho"] for p in ps], [p["wq"] for p in ps],
                     yerr=[[p["wq"] - p["lo"] for p in ps], [p["hi"] - p["wq"] for p in ps]],
                     fmt=mk, color="black", ms=4.5, capsize=2.5, lw=0.8, ls="none",
                     markerfacecolor="white", markeredgewidth=0.8)
        handles.append(Line2D([], [], color="0.45", lw=1.0, ls=ls, marker=mk, ms=4.5,
                              mfc="white", mec="black", mew=0.8, label=name))
    ax2.legend(handles=handles, frameon=False, loc="upper left",
               title="markers: measured; lines: P-K theory", title_fontsize=6.5,
               alignment="left")
    ax2.set_title("(b) Same mean service time, different variability", loc="left")

    for ax, xmax in ((ax1, 0.85), (ax2, 0.85)):
        ax.set_xlim(0, xmax); ax.set_ylim(0, 4.4)
        ax.set_xlabel("Utilization ρ"); ax.set_ylabel("Mean waiting time (s)")
        ax.grid(alpha=0.3, lw=0.4)

    fig.tight_layout(pad=0.3, h_pad=0.9)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"figure_column.{ext}"))
    print("→ figure_column.png / .pdf")


if __name__ == "__main__":
    main()
