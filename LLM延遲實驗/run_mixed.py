"""混合長度實驗（Day 29–34）：服務時間的變異如何影響排隊延遲

設計
  條件 A：全部用 medium.txt（838 tokens）        → 服務時間變異小
  條件 B：short.txt / long.txt 各半（118 / 1478） → 平均長度相近，但變異大
  兩個條件都在 ρ ≈ 0.5 與 0.75 各跑一次，比較實測等待時間。

流程（全自動）
  1. 校正：每個條件先低到達率跑 40 筆，量出 E[S]
  2. 主跑：依各自的 E[S] 換算 λ，跑 4 個組合
  3. 分析：用 FIFO 重建法算等待時間，和 P-K 比較，畫關鍵圖 2

用法：python run_mixed.py
"""
import csv
import os
import random
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:8080/v1/chat/completions"
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
random.seed(0)

COND = {
    "A": dict(label="fixed 838 tok", args=["--prompt-file", "prompts/medium.txt"]),
    "B": dict(label="50% 118 tok / 50% 1478 tok",
              args=["--prompt-file", "prompts/short.txt", "--prompt-file", "prompts/long.txt",
                    "--mix", "0.5,0.5"]),
}
RHOS = [0.50, 0.75]
N_BY_RHO = {0.50: 250, 0.75: 450}


def run_load(cond, lam, n, out):
    cmd = [PY, os.path.join(HERE, "load_gen.py"), "--url", URL, "--rate", f"{lam}",
           "--n", str(n), "--max-tokens", "128", "--ignore-eos", "--unique-prompt",
           "--out", out] + COND[cond]["args"]
    subprocess.run(cmd, cwd=HERE, env=ENV, check=True)


def rebuild(path):
    rows = sorted([r for r in csv.DictReader(open(os.path.join(HERE, path), encoding="utf-8"))
                   if r["status"] == "ok"], key=lambda r: float(r["t_send"]))
    out, prev = [], 0.0
    for r in rows:
        snd, done = float(r["t_send"]), float(r["t_done"])
        start = max(snd, prev)
        out.append((start - snd, done - start, r["prompt_type"]))
        prev = done
    return out


def boot(xs, b=3000):
    m = sorted(st.mean(random.choices(xs, k=len(xs))) for _ in range(b))
    return m[int(0.025 * b)], m[int(0.975 * b)]


def main():
    t_start = time.time()
    # ---- 1. 校正 ----
    es = {}
    for c in ("A", "B"):
        out = f"out/mix_cal_{c}.csv"
        print(f"\n=== 校正 {c}（{COND[c]['label']}）：40 筆，λ=0.12 ===", flush=True)
        run_load(c, 0.12, 40, out)
        d = rebuild(out)[5:]
        es[c] = st.mean([s for _, s, _ in d])
        cv = st.pstdev([s for _, s, _ in d]) / es[c]
        print(f"  E[S]={es[c]:.3f}s  CV={cv:.3f}", flush=True)

    # ---- 2. 主跑 ----
    plan = [(c, rho) for c in ("A", "B") for rho in RHOS]
    random.shuffle(plan)
    est = sum(N_BY_RHO[r] / (r / es[c]) for c, r in plan) / 60
    print(f"\n執行順序：{plan}，預估 {est:.0f} 分鐘\n", flush=True)
    for c, rho in plan:
        lam = rho / es[c]
        n = N_BY_RHO[rho]
        print(f"=== 條件 {c}  ρ≈{rho}  λ={lam:.3f}/s  {n} 筆（約 {n / lam / 60:.0f} 分）===", flush=True)
        run_load(c, lam, n, f"out/mix_{c}_rho{int(rho * 100)}.csv")

    # ---- 3. 分析 ----
    print("\n" + "=" * 96)
    print(f"{'條件':>4}{'內容':>26}{'ρ目標':>8}{'n':>6}{'E[S]':>8}{'CV':>7}{'真實ρ':>8}"
          f"{'實測W_q':>10}{'95% CI':>20}{'P-K理論':>10}")
    res = []
    for c in ("A", "B"):
        for rho in RHOS:
            path = f"out/mix_{c}_rho{int(rho * 100)}.csv"
            d = rebuild(path)
            d = d[len(d) // 3:]                       # 捨棄前 1/3
            W = [w for w, _, _ in d]
            S = [s for _, s, _ in d]
            lam = rho / es[c]
            m, m2 = st.mean(S), sum(x * x for x in S) / len(S)
            cv = st.pstdev(S) / m
            r_real = lam * m
            th = lam * m2 / (2 * (1 - r_real)) if r_real < 1 else float("inf")
            lo, hi = boot(W)
            print(f"{c:>4}{COND[c]['label']:>26}{rho:>8.2f}{len(W):>6}{m:>8.3f}{cv:>7.3f}"
                  f"{r_real:>8.3f}{st.mean(W):>10.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}{th:>10.3f}")
            res.append(dict(cond=c, rho=rho, n=len(W), es=m, cv=cv, rho_real=r_real,
                            wq=st.mean(W), lo=lo, hi=hi, th=th))
    # 條件內各 prompt 類型的服務時間
    print("\n各 prompt 類型的服務時間：")
    for c in ("A", "B"):
        d = sum((rebuild(f"out/mix_{c}_rho{int(r * 100)}.csv") for r in RHOS), [])
        by = {}
        for _, s, t in d:
            by.setdefault(t, []).append(s)
        print(f"  條件 {c}: " + "  ".join(f"{k}: {st.mean(v):.2f}s (n={len(v)})" for k, v in sorted(by.items())))
    plot(res)
    print(f"\n總耗時 {(time.time() - t_start) / 60:.0f} 分鐘")


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    styles = {"A": dict(marker="o", ls="-", label="A: fixed length (838 tok)"),
              "B": dict(marker="^", ls="--", label="B: mixed length (118 / 1478 tok)")}
    for c in ("A", "B"):
        pts = [r for r in res if r["cond"] == c]
        xs = [p["rho_real"] for p in pts]
        ys = [p["wq"] for p in pts]
        err = [[p["wq"] - p["lo"] for p in pts], [p["hi"] - p["wq"] for p in pts]]
        cv = st.mean([p["cv"] for p in pts])
        ax.errorbar(xs, ys, yerr=err, color="black", capsize=4, markerfacecolor="white",
                    markersize=7, lw=1.2, **{**styles[c], "label": f"{styles[c]['label']}, CV={cv:.2f}"})
        ax.plot(xs, [p["th"] for p in pts], "x", color="gray", markersize=8,
                label="P-K theory" if c == "A" else None)
    ax.set_xlabel("Utilization  ρ = λ·E[S]")
    ax.set_ylabel("Mean waiting time  $W_q$  (s)")
    ax.set_title("Effect of service-time variability on queueing delay")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"figure2_variability.{ext}"))
    print("\n關鍵圖 2 → figure2_variability.png / .pdf")


if __name__ == "__main__":
    main()
