"""主實驗分析（Day 25–26）：重建等待時間、和 P-K 理論比較、畫關鍵圖 1

等待時間怎麼算：
  伺服器是單一服務台、先進先出，所以第 i 個請求開始被服務的時刻
    start_i = max(送出時刻_i, 完成時刻_{i-1})
  等待 = start_i − 送出時刻_i，服務時間 = 完成時刻_i − start_i
  好處：不必事先量 E[S]，也不受客戶端連線開銷影響。

暖機：捨棄每個負載點的前 1/3（高負載從空佇列開始要很久才到穩態）。

用法：python analyze_main.py
"""
import csv
import os
import random
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(0)

# (檔案, 名目 ρ, λ)
RUNS = [("out/main2_rho30.csv", 0.30, 0.14995),
        ("out/main2_rho50.csv", 0.50, 0.24991),
        ("out/long70_rho70.csv", 0.70, 0.34988),
        ("out/long85_rho85.csv", 0.85, 0.42485)]


def rebuild(path):
    rows = sorted([r for r in csv.DictReader(open(os.path.join(HERE, path), encoding="utf-8"))
                   if r["status"] == "ok"], key=lambda r: float(r["t_send"]))
    out, prev_done = [], 0.0
    for r in rows:
        snd, done = float(r["t_send"]), float(r["t_done"])
        # 我的理解：用實際量到的時間直接反推服務開始時刻，不必事先量 E[S]，
        # 也不會被連線開銷干擾（參見修正紀錄 #6 的 localhost 問題）
        start = max(snd, prev_done)
        out.append((start - snd, done - start))
        prev_done = done
    return out


def boot(xs, b=3000):
    m = sorted(st.mean(random.choices(xs, k=len(xs))) for _ in range(b))
    return m[int(0.025 * b)], m[int(0.975 * b)]


def main():
    print(f"{'名目ρ':>7}{'λ':>8}{'總筆數':>7}{'暖機':>6}{'n':>6}{'E[S]':>8}{'真實ρ':>8}"
          f"{'實測W_q':>10}{'95% CI':>20}{'P-K理論':>10}{'判定':>9}")
    pts = []
    for path, nom, lam in RUNS:
        d = rebuild(path)
        # 我的理解：實驗要進入穩態需要時間（佇列從空的開始排），
        # 前段的等待時間會偏低，所以捨棄
        warm = len(d) // 3                      # 捨棄前 1/3
        d = d[warm:]
        W = [w for w, _ in d]
        S = [s for _, s in d]
        es, es2 = st.mean(S), sum(x * x for x in S) / len(S)
        rho = lam * es
        th = lam * es2 / (2 * (1 - rho))
        lo, hi = boot(W)
        ok = lo <= th <= hi
        print(f"{nom:>7.2f}{lam:>8.3f}{len(W) + warm:>7}{warm:>6}{len(W):>6}{es:>8.3f}{rho:>8.3f}"
              f"{st.mean(W):>10.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}{th:>10.3f}{'✅ 吻合' if ok else '⚠️ 不吻合':>9}")
        pts.append(dict(nom=nom, lam=lam, rho=rho, mean=st.mean(W), lo=lo, hi=hi,
                        th=th, es=es, es2=es2, n=len(W)))
    plot(pts)
    return pts


def plot(pts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    es = st.mean([p["es"] for p in pts])
    es2 = st.mean([p["es2"] for p in pts])
    xs = [i / 200 for i in range(1, 181)]       # ρ = 0.005 ~ 0.90
    ys = [(r / es) * es2 / (2 * (1 - r)) for r in xs]

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    ax.plot(xs, ys, "-", color="black", lw=1.4,
            label=f"P-K theory  (E[S]={es:.2f}s, E[S²]={es2:.2f})")
    ax.errorbar([p["rho"] for p in pts], [p["mean"] for p in pts],
                yerr=[[p["mean"] - p["lo"] for p in pts], [p["hi"] - p["mean"] for p in pts]],
                fmt="o", color="black", markersize=6, capsize=4, lw=1.2,
                markerfacecolor="white", label="Measured (mean, 95% CI)")
    for p in pts:
        ax.annotate(f"n={p['n']}", (p["rho"], p["mean"]), textcoords="offset points",
                    xytext=(6, -12), fontsize=8)
    ax.set_xlabel("Utilization  ρ = λ·E[S]")
    ax.set_ylabel("Mean waiting time  $W_q$  (s)")
    ax.set_title("Queueing delay of local LLM inference (qwen2.5-coder-3B-Q4, CPU)")
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, max(p["hi"] for p in pts) * 1.35)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"figure1_wq_vs_rho.{ext}"))
    print(f"\n關鍵圖 1 → figure1_wq_vs_rho.png / .pdf")


if __name__ == "__main__":
    main()
