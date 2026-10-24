"""新第 VI 節的圖（草稿；10/19 畫兩端，10/20 改成五個實測點）：保溫時平均回應時間和顯示卡平均功耗的取捨

- ×：實測，同一串到達時間（ρ ≈ 0.18）：τ = 0（預設，10/19 早上）、3、6、12 秒（鎖住 τ 秒再放開，10/19 晚上）、∞（一直鎖時脈，10/19 早上）
- 空心圓和誤差線：事前預測和 95% 範圍（兩端 predict_power.py 09:12、中間三點 run_keepwarm.py 22:52；照同一串到達時間模擬）
- 虛線：兩端實測連成的直線，也就是在兩種設定之間切換能達到的點
輸出：out/keepwarm_tradeoff.png、out/keepwarm_tradeoff.pdf
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "out", "power_results.json"), encoding="utf-8"))
pre = json.load(open(os.path.join(HERE, "out", "power_prediction.json"), encoding="utf-8"))
pts = [("default", r"$\tau=0$"), ("kw_tau3", r"$\tau=3$"), ("kw_tau6", r"$\tau=6$"), ("kw_tau12", r"$\tau=12$"),
       ("locked", r"$\tau=\infty$")]
pred = {k: (pre[k]["power_pred_W"], pre[k]["resp_pred"], pre[k]["resp_lo"], pre[k]["resp_hi"]) for k in ("default", "locked")}
for t in (3, 6, 12):
    r = json.load(open(os.path.join(HERE, f"plan_kw_tau{t}.json"), encoding="utf-8"))["runs"][0]
    pred[f"kw_tau{t}"] = (r["power_pred_W"], r["resp_pred"], r["resp_lo"], r["resp_hi"])

fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)
for i, (k, _) in enumerate(pts):
    pw, tp, lo, hi = pred[k]
    ax.errorbar(pw, tp, yerr=[[tp - lo], [hi - tp]], fmt="o", ms=4, mfc="none", mec="#d62728", ecolor="#d62728",
                elinewidth=0.7, capsize=1.5, mew=0.8, label="predicted in advance (95% range)" if i == 0 else None)
d, l = res["default"], res["locked"]
ax.plot([d["power_W"], l["power_W"]], [d["resp"], l["resp"]], "--", color="0.45", lw=0.9,
        label="switching between default and locked")
ax.plot([res[k]["power_W"] for k, _ in pts], [res[k]["resp"] for k, _ in pts], "x", ms=5, color="k", mew=1.0, ls="none",
        label=r"measured ($\rho\approx0.18$)")
for k, lab in pts:
    ax.annotate(lab, (res[k]["power_W"], res[k]["resp"]), textcoords="offset points",
                xytext=(5, 2) if k != "locked" else (-24, 4), fontsize=5.5)
ax.set_xlabel("Mean GPU power (W)", fontsize=7)
ax.set_ylabel("Mean response time (s)", fontsize=7)
ax.tick_params(labelsize=6)
ax.grid(alpha=0.3, lw=0.4)
ax.legend(fontsize=5.5, frameon=False, loc="upper right")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "out", f"keepwarm_tradeoff.{ext}"))
print("→ out/keepwarm_tradeoff.png、.pdf")
