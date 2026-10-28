"""改寫版的 Fig. 3（第 VI 節，保溫的延遲和耗電取捨；10/23）：三次量測的五個 τ，加上事前預測和兩端的連線
- 實心：實測（同一串到達時間，ρ ≈ 0.18）；空心加誤差線：事前預測和 95% 模擬範圍；線：兩端實測的連線（在預設和一直鎖之間切換）
- 第 1 次 10/19（兩端早上、中間三點晚上）：LLM延遲實驗/out/power_results.json、power_prediction.json、plan_kw_tau*.json
- 第 2 次 10/22、第 3 次 10/23（同一段時間）：out/kw5_*.json、out/kw5m_*.json
- 顏色：藍、紫、紅（使用者選的；dataviz 驗證散佈圖兩兩全部通過）；印成黑白時靠點的形狀和線型分
輸出：fig3_keepwarm.pdf／.png（10/14 版的 fig3_offload 不動）
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "LLM延遲實驗")
J = lambda *p: json.load(open(os.path.join(LAB, *p), encoding="utf-8"))
TAUS = ["0", "3", "6", "12", "inf"]

# 第 1 次（10/19）
res1, pre1 = J("out", "power_results.json"), J("out", "power_prediction.json")
key1 = {"0": "default", "3": "kw_tau3", "6": "kw_tau6", "12": "kw_tau12", "inf": "locked"}
s1 = {"meas": {t: (res1[key1[t]]["power_W"], res1[key1[t]]["resp"]) for t in TAUS}, "pred": {}}
for t in ("0", "inf"):
    p = pre1[key1[t]]
    s1["pred"][t] = (p["power_pred_W"], p["resp_pred"], p["resp_lo"], p["resp_hi"])
for t in ("3", "6", "12"):
    r = J(f"plan_kw_tau{t}.json")["runs"][0]
    s1["pred"][t] = (r["power_pred_W"], r["resp_pred"], r["resp_lo"], r["resp_hi"])


def same_session(tag):
    r, p = J("out", f"{tag}_results.json")["points"], J("out", f"{tag}_prediction.json")["pred"]
    return {"meas": {t: (r[t]["power_W"], r[t]["resp"]) for t in TAUS},
            "pred": {t: (p[t]["power_pred_W"], p[t]["resp_pred"], p[t]["resp_95"][0], p[t]["resp_95"][1]) for t in TAUS}}


# 10/23 使用者：「我不喜歡橘色 改成藍色紫色紅色」→ 藍 550、紫（淺一階）、紅；dataviz 驗證（散佈圖兩兩）全部通過
SESS = [("Session 1", s1, "#1c5cab", "o", (0, (4, 2))),
        ("Session 2", same_session("kw5"), "#9085e9", "s", (0, (5, 1.5, 1, 1.5))),
        ("Session 3", same_session("kw5m"), "#e34948", "^", (0, (1, 1.2)))]

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 6.6, "axes.linewidth": 0.6, "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42})   # 10/24：字型和 Fig. 1、2 相同；不用 Type 3 字型（IEEE PDF eXpress 可能不收）
fig, ax = plt.subplots(figsize=(3.45, 2.5))   # 10/28：2.75 → 2.5（論文超過 6 頁）
for name, s, col, mk, ls in SESS:
    d, l = s["meas"]["0"], s["meas"]["inf"]
    ax.plot([d[0], l[0]], [d[1], l[1]], color=col, lw=0.9, ls=ls, zorder=1)
    for t in TAUS:
        pw, tp, lo, hi = s["pred"][t]
        ax.errorbar(pw, tp, yerr=[[tp - lo], [hi - tp]], fmt=mk, ms=3.6, mfc="white", mec=col, mew=0.8,
                    ecolor=col, elinewidth=0.6, capsize=1.4, zorder=2)
        ax.plot(*s["meas"][t], mk, ms=4.6, color=col, mec="white", mew=0.5, zorder=3)
# τ 的標籤：只標第 1 次的五個實測點（另外兩次沿著自己的線、同樣的順序），圖說裡說明
lab = {"0": r"$\tau=0$", "3": "3 s", "6": "6 s", "12": "12 s", "inf": r"$\tau=\infty$"}
off = {"0": (-24, -3), "3": (4, -2), "6": (-15, 3), "12": (-18, 3), "inf": (-8, 13)}
for t in TAUS:
    ax.annotate(lab[t], s1["meas"][t], textcoords="offset points", xytext=off[t], fontsize=6.4, color="0.15")
ax.set_xlim(17.8, 37.6)
ax.set_ylim(1.925, 2.30)
ax.set_xlabel("Mean GPU power (W)")
ax.set_ylabel("Mean response time (s)")
ax.grid(color="0.9", lw=0.5)
ax.set_axisbelow(True)
h1 = [Line2D([], [], color=col, ls=ls, lw=0.9, marker=mk, ms=4.6, mec="white", mew=0.5) for _, _, col, mk, ls in SESS]
leg1 = ax.legend(h1, [n for n, *_ in SESS], loc="upper right", frameon=False, handlelength=2.6)
ax.add_artist(leg1)
h2 = [Line2D([], [], color="0.35", marker="o", ms=4.6, ls="none", mec="white", mew=0.5),
      Line2D([], [], color="0.35", marker="o", ms=3.6, ls="none", mfc="white", mew=0.8),
      Line2D([], [], color="0.35", lw=0.9, ls=(0, (4, 2)))]
ax.legend(h2, ["measured", "prediction (95% range)", "switching (endpoint line)"], loc="lower left",
          frameon=False, handlelength=2.0)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig3_keepwarm.{ext}"), dpi=300)
print("→ fig3_keepwarm.pdf、.png")
