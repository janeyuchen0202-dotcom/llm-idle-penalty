"""新 Fig. 2（2026-10-24）：全部排隊預測的「預測對實測」散佈圖，取代 10/14 版的 fig2_accuracy

- 橫軸實測、縱軸預測的平均等待（秒），對數座標；對角線是完全準，灰色帶子是 ±10%
- 實心：33 組事前預測（照登記的範圍畫誤差線）。藍色圓點 = RTX 4050、M1，
  紅色三角 = 一直忙就變慢的 HP 筆電、志願者第 4、6、9 台（圖例只寫機器名稱，不寫「穩定」：M1 一直忙幾分鐘也會慢）
- 空心：RTX 4050 的 17 組樣本內（閒置效應模型，參數從這 17 組估）
- 灰色 ×：和每組事前預測一起登記的對照公式（P-K）。算不出來（ρ ≥ 1）的一組不畫
- 顏色是 dataviz 的藍 #1c5cab、紅 #e34948（和 Fig. 3 相同，validate_palette.js 兩兩通過）；另用點的形狀區分，黑白也看得出來

資料：稽核/事前預測彙整.csv（稽核/事前預測彙整.py 產生）、LLM延遲實驗/out/strengthen_summary.csv
用法：python make_prereg_fig2.py   → fig2_predictions.pdf／.png
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(HERE, "稽核", "事前預測彙整.csv")
INS = os.path.join(HERE, "..", "LLM延遲實驗", "out", "strengthen_summary.csv")
BLUE, RED, GRAY = "#1c5cab", "#e34948", "0.55"
HEAT = ("RTX 20", "RTX 30", "HP")          # 一直忙就變慢的機器（device 欄的開頭）

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 6.8, "axes.linewidth": 0.6, "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42})


def main():
    pre = list(csv.DictReader(open(PRE, encoding="utf-8-sig")))
    ins = list(csv.DictReader(open(INS, encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    lo_ax, hi_ax = 0.2, 25
    ax.plot([lo_ax, hi_ax], [lo_ax, hi_ax], color="black", lw=0.6, zorder=1)
    ax.fill_between([lo_ax, hi_ax], [lo_ax * 0.9, hi_ax * 0.9], [lo_ax * 1.1, hi_ax * 1.1], color="0.9", lw=0, zorder=0)
    # 對照公式
    for r in pre:
        b = float(r["base"])
        if math.isfinite(b):
            ax.scatter(float(r["meas"]), b, marker="x", s=12, color=GRAY, linewidths=0.7, zorder=2)
    # 樣本內 17 組
    ax.scatter([float(r["W"]) for r in ins], [float(r["mod_mean"]) for r in ins], marker="o", s=16,
               facecolors="white", edgecolors="black", linewidths=0.6, zorder=3)
    # 事前預測 33 組
    for r in pre:
        heat = r["device"].startswith(HEAT)
        x, y = float(r["meas"]), float(r["pred"])
        ax.errorbar(x, y, yerr=[[y - float(r["lo"])], [float(r["hi"]) - y]], fmt="none", ecolor=RED if heat else BLUE,
                    elinewidth=0.7, capsize=1.5, capthick=0.6, zorder=4)
        ax.scatter(x, y, marker="^" if heat else "o", s=18 if heat else 14, color=RED if heat else BLUE,
                   edgecolors="white", linewidths=0.4, zorder=5)
    hp = max((r for r in pre if r["device"].startswith("HP")), key=lambda r: float(r["meas"]))
    ax.annotate("HP laptop", (float(hp["meas"]), float(hp["pred"])), xytext=(-36, -3), textcoords="offset points",
                fontsize=6.5, color="0.25")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo_ax, hi_ax)
    ax.set_ylim(lo_ax, hi_ax)
    ticks = [0.2, 0.5, 1, 2, 5, 10, 20]
    for setter in (ax.set_xticks, ax.set_yticks):
        setter(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Measured mean waiting time (s)")
    ax.set_ylabel("Predicted mean waiting time (s)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3, lw=0.4)
    h = [Line2D([], [], ls="", marker="o", color=BLUE, mec="white", ms=4.6, label="In advance: RTX 4050, M1"),
         Line2D([], [], ls="", marker="^", color=RED, mec="white", ms=5, label="In advance: HP laptop, desktops"),
         Line2D([], [], ls="", marker="o", mfc="white", mec="black", ms=4.2, label="In-sample (RTX 4050, 17 runs)"),
         Line2D([], [], ls="", marker="x", color=GRAY, ms=4, label="P-K formula listed with each prediction"),
         plt.Rectangle((0, 0), 1, 1, color="0.9", label="±10%")]
    ax.legend(handles=h, frameon=False, loc="upper left", handletextpad=0.3, borderaxespad=0.3, labelspacing=0.35)
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig2_predictions.{ext}"), dpi=300)
    print("→ fig2_predictions.pdf／.png")


if __name__ == "__main__":
    main()
