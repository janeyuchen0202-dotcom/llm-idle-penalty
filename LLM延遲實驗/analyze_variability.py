"""變異實驗分析（Day 29–34）：服務時間的變異如何影響排隊延遲

條件 A：固定輸出 128 tokens        → CV ≈ 0.07
條件 C：一半 32、一半 224 tokens   → CV ≈ 0.70（平均輸出一樣是 128）

兩者的平均服務時間幾乎相同，只有變異不同。
等待時間用 FIFO 重建法計算，捨棄每個負載點的前 1/3。

用法：python analyze_variability.py
"""
import csv
import os
import random
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(0)

RUNS = [("A", "out/mix_A_rho50.csv", 0.259),      # λ = 0.50 / 1.930
        ("A", "out/mix_A_rho75.csv", 0.389),      # λ = 0.75 / 1.930
        ("C", "out/var_C_rho50.csv", 0.233),      # λ = 0.50 / 2.143
        ("C", "out/var_C_rho70.csv", 0.330)]


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


def boot(xs, b=3000):
    m = sorted(st.mean(random.choices(xs, k=len(xs))) for _ in range(b))
    return m[int(0.025 * b)], m[int(0.975 * b)]


def main():
    print(f"{'條件':>4}{'n':>6}{'E[S]':>8}{'CV':>7}{'ρ':>8}{'實測W_q':>10}{'95% CI':>20}"
          f"{'P-K理論':>10}{'中位數':>8}{'p95':>8}")
    res = []
    for cond, path, lam in RUNS:
        d = rebuild(path)
        d = d[len(d) // 3:]
        W = sorted(w for w, _ in d)
        S = [s for _, s in d]
        es, es2 = st.mean(S), sum(x * x for x in S) / len(S)
        rho = lam * es
        th = lam * es2 / (2 * (1 - rho))
        lo, hi = boot([w for w, _ in d])
        print(f"{cond:>4}{len(W):>6}{es:>8.3f}{st.pstdev(S) / es:>7.3f}{rho:>8.3f}"
              f"{st.mean(W):>10.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}{th:>10.3f}"
              f"{st.median(W):>8.2f}{W[int(0.95 * len(W))]:>8.2f}")
        res.append(dict(cond=cond, n=len(W), es=es, cv=st.pstdev(S) / es, rho=rho,
                        wq=st.mean(W), lo=lo, hi=hi, th=th, es2=es2,
                        med=st.median(W), p95=W[int(0.95 * len(W))]))
    plot(res)
    return res


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.5), dpi=150)
    for cond, name, mk, ls in (("A", "A: fixed 128 output tokens", "o", "-"),
                               ("C", "C: 50% 32 / 50% 224 tokens", "^", "--")):
        pts = sorted([r for r in res if r["cond"] == cond], key=lambda r: r["rho"])
        cv = st.mean([p["cv"] for p in pts])
        es = st.mean([p["es"] for p in pts])
        es2 = st.mean([p["es2"] for p in pts])
        xs = [i / 200 for i in range(1, 171)]
        ax.plot(xs, [(r / es) * es2 / (2 * (1 - r)) for r in xs], ls, color="gray", lw=1.1)
        ax.errorbar([p["rho"] for p in pts], [p["wq"] for p in pts],
                    yerr=[[p["wq"] - p["lo"] for p in pts], [p["hi"] - p["wq"] for p in pts]],
                    fmt=mk, color="black", capsize=4, markerfacecolor="white", markersize=7,
                    lw=1.2, ls="none", label=f"{name}   CV={cv:.2f}")
    ax.plot([], [], "-", color="gray", lw=1.1, label="P-K theory (same line style)")
    ax.set_xlabel("Utilization  ρ = λ·E[S]")
    ax.set_ylabel("Mean waiting time  $W_q$  (s)")
    ax.set_title("Same mean service time, different variability")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 4.2)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"figure2_variability.{ext}"))
    print("\n關鍵圖 2 → figure2_variability.png / .pdf")


if __name__ == "__main__":
    main()
