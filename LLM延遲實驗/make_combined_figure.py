"""把關鍵圖 1 與 2 合併成一張雙面板圖（計畫書用）

(a) 平均等待時間 vs 利用率，與 P-K 理論比較
(b) 相同平均服務時間下，服務時間變異的影響

設計重點（兩頁的計畫書用）：
- 整張寬 6.8 英吋、高 2.6 英吋，印出來約佔 A4 寬度的 80%、高度 0.25 頁
- 字體不隨圖縮小：座標軸 9pt、圖例 7.5pt
- 黑白可讀：靠線型與標記形狀區分，不靠顏色

用法：python make_combined_figure.py
"""
import csv
import os
import random
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(0)

MAIN = [("out/main2_rho30.csv", 0.14995), ("out/main2_rho50.csv", 0.24991),
        ("out/long70_rho70.csv", 0.34988), ("out/long85_rho85.csv", 0.42485)]
VAR = {"A": [("out/mix_A_rho50.csv", 0.259), ("out/mix_A_rho75.csv", 0.389)],
       "C": [("out/var_C_rho50.csv", 0.233), ("out/var_C_rho70.csv", 0.330)]}


def rebuild(path):
    rows = sorted([r for r in csv.DictReader(open(os.path.join(HERE, path), encoding="utf-8"))
                   if r["status"] == "ok"], key=lambda r: float(r["t_send"]))
    out, prev = [], 0.0
    for r in rows:
        snd, done = float(r["t_send"]), float(r["t_done"])
        start = max(snd, prev)
        out.append((start - snd, done - start))
        prev = done
    return out


def boot(xs, b=2000):
    m = sorted(st.mean(random.choices(xs, k=len(xs))) for _ in range(b))
    return m[int(0.025 * b)], m[int(0.975 * b)]


def stats(path, lam):
    d = rebuild(path)
    d = d[len(d) // 3:]
    W = [w for w, _ in d]
    S = [s for _, s in d]
    es, es2 = st.mean(S), sum(x * x for x in S) / len(S)
    rho = lam * es
    lo, hi = boot(W)
    return dict(rho=rho, wq=st.mean(W), lo=lo, hi=hi,
                th=lam * es2 / (2 * (1 - rho)), es=es, es2=es2,
                cv=st.pstdev(S) / es)


def main():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.5})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6), dpi=300)

    # ---- (a) 主實驗 ----
    pts = [stats(p, l) for p, l in MAIN]
    es = st.mean([p["es"] for p in pts]); es2 = st.mean([p["es2"] for p in pts])
    xs = [i / 200 for i in range(1, 171)]
    ax1.plot(xs, [(r / es) * es2 / (2 * (1 - r)) for r in xs], "-", color="black", lw=1.1,
             label="P-K theory")
    ax1.errorbar([p["rho"] for p in pts], [p["wq"] for p in pts],
                 yerr=[[p["wq"] - p["lo"] for p in pts], [p["hi"] - p["wq"] for p in pts]],
                 fmt="o", color="black", ms=5, capsize=3, lw=1, ls="none",
                 markerfacecolor="white", label="Measured (95% CI)")
    ax1.set_xlabel("Utilization  ρ"); ax1.set_ylabel("Mean waiting time (s)")
    ax1.set_title("(a) Waiting time vs. load", loc="left")
    ax1.set_xlim(0, 0.85); ax1.set_ylim(0, 4.4)
    ax1.grid(alpha=0.3, lw=0.4); ax1.legend(frameon=False, loc="upper left")

    # ---- (b) 變異 ----
    for key, name, mk, ls in (("A", "fixed 128 tok", "o", "-"), ("C", "32/224 tok", "^", "--")):
        ps = sorted([stats(p, l) for p, l in VAR[key]], key=lambda d: d["rho"])
        cv = st.mean([p["cv"] for p in ps])
        es = st.mean([p["es"] for p in ps]); es2 = st.mean([p["es2"] for p in ps])
        ax2.plot(xs, [(r / es) * es2 / (2 * (1 - r)) for r in xs], ls, color="gray", lw=1)
        ax2.errorbar([p["rho"] for p in ps], [p["wq"] for p in ps],
                     yerr=[[p["wq"] - p["lo"] for p in ps], [p["hi"] - p["wq"] for p in ps]],
                     fmt=mk, color="black", ms=5.5, capsize=3, lw=1, ls="none",
                     markerfacecolor="white", label=f"{name}  (CV={cv:.2f})")
    ax2.set_xlabel("Utilization  ρ"); ax2.set_ylabel("Mean waiting time (s)")
    ax2.set_title("(b) Same mean, different variability", loc="left")
    ax2.set_xlim(0, 0.8); ax2.set_ylim(0, 4.4)
    ax2.grid(alpha=0.3, lw=0.4); ax2.legend(frameon=False, loc="upper left")

    fig.tight_layout(pad=0.6, w_pad=1.6)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"figure_combined.{ext}"), bbox_inches="tight")
    print("→ figure_combined.png / .pdf")


if __name__ == "__main__":
    main()
