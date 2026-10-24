"""服務時間變異實驗（修正版）：用「輸出長度」製造變異

背景：第一次用 prompt 長度（118 vs 1478 tokens）製造變異失敗，因為 prefill 比 decode
      快 42 倍（3,400 vs 80 tokens/秒），服務時間幾乎只由輸出長度決定。

設計（prompt 都用 medium.txt，只改輸出長度）
  條件 A：固定輸出 128 tokens          → 服務時間變異小（CV≈0.07）
  條件 C：一半 32、一半 224 tokens     → 平均輸出一樣是 128，但變異大
  兩個條件都在 ρ ≈ 0.5 與 0.75 各跑一次。

條件 A 直接沿用 run_mixed.py 跑過的 out/mix_A_rho*.csv。

用法：python run_varout.py
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
RHOS = [0.50, 0.75]
N_BY_RHO = {0.50: 250, 0.75: 450}
C_ARGS = ["--max-tokens", "32,224", "--token-mix", "0.5,0.5"]


def run_load(lam, n, out, extra):
    subprocess.run([PY, os.path.join(HERE, "load_gen.py"), "--url", URL, "--rate", f"{lam}",
                    "--n", str(n), "--prompt-file", "prompts/medium.txt",
                    "--ignore-eos", "--unique-prompt", "--out", out] + extra,
                   cwd=HERE, env=ENV, check=True)


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


def stats(path, lam):
    d = rebuild(path)
    d = d[len(d) // 3:]
    W = [w for w, _, _ in d]
    S = [s for _, s, _ in d]
    m, m2 = st.mean(S), sum(x * x for x in S) / len(S)
    rho = lam * m
    th = lam * m2 / (2 * (1 - rho)) if rho < 1 else float("inf")
    lo, hi = boot(W)
    return dict(n=len(W), es=m, cv=st.pstdev(S) / m, rho=rho, wq=st.mean(W),
                lo=lo, hi=hi, th=th, p95=sorted(W)[int(0.95 * len(W))])


def main():
    t0 = time.time()
    print("=== 校正條件 C（輸出 32/224 各半）：40 筆 ===", flush=True)
    run_load(0.12, 40, "out/var_cal_C.csv", C_ARGS)
    d = rebuild("out/var_cal_C.csv")[5:]
    esC = st.mean([s for _, s, _ in d])
    print(f"  E[S]={esC:.3f}s  CV={st.pstdev([s for _, s, _ in d]) / esC:.3f}\n", flush=True)

    for rho in RHOS:
        lam = rho / esC
        n = N_BY_RHO[rho]
        print(f"=== 條件 C  ρ≈{rho}  λ={lam:.3f}/s  {n} 筆（約 {n / lam / 60:.0f} 分）===", flush=True)
        run_load(lam, n, f"out/var_C_rho{int(rho * 100)}.csv", C_ARGS)

    # ---- 分析：A（沿用）vs C ----
    esA = 1.93
    print("\n" + "=" * 104)
    print(f"{'條件':>4}{'輸出長度':>16}{'ρ目標':>7}{'n':>6}{'E[S]':>8}{'CV':>7}{'真實ρ':>8}"
          f"{'實測W_q':>10}{'95% CI':>20}{'P-K理論':>10}{'p95 W_q':>10}")
    res = []
    for cond, label, es_cal, path_fmt in (("A", "固定 128", esA, "out/mix_A_rho{}.csv"),
                                          ("C", "32/224 各半", esC, "out/var_C_rho{}.csv")):
        for rho in RHOS:
            lam = rho / es_cal
            s = stats(path_fmt.format(int(rho * 100)), lam)
            print(f"{cond:>4}{label:>16}{rho:>7.2f}{s['n']:>6}{s['es']:>8.3f}{s['cv']:>7.3f}"
                  f"{s['rho']:>8.3f}{s['wq']:>10.3f}{f'[{s[chr(108)+chr(111)]:.3f}, {s[chr(104)+chr(105)]:.3f}]':>20}"
                  f"{s['th']:>10.3f}{s['p95']:>10.3f}")
            res.append(dict(cond=cond, label=label, rho_t=rho, **s))
    plot(res)
    print(f"\n總耗時 {(time.time() - t0) / 60:.0f} 分鐘")


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=150)
    style = {"A": dict(marker="o", ls="-"), "C": dict(marker="^", ls="--")}
    for c, name in (("A", "A: fixed 128 output tokens"), ("C", "C: 50% 32 / 50% 224 tokens")):
        pts = [r for r in res if r["cond"] == c]
        cv = st.mean([p["cv"] for p in pts])
        ax.errorbar([p["rho"] for p in pts], [p["wq"] for p in pts],
                    yerr=[[p["wq"] - p["lo"] for p in pts], [p["hi"] - p["wq"] for p in pts]],
                    color="black", capsize=4, markerfacecolor="white", markersize=7, lw=1.2,
                    label=f"{name}  (CV={cv:.2f})", **style[c])
        ax.plot([p["rho"] for p in pts], [p["th"] for p in pts], "x", color="gray", markersize=8,
                label="P-K theory" if c == "A" else None)
    ax.set_xlabel("Utilization  ρ = λ·E[S]")
    ax.set_ylabel("Mean waiting time  $W_q$  (s)")
    ax.set_title("Service-time variability and queueing delay (same mean service time)")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"figure2_variability.{ext}"))
    print("\n關鍵圖 2 → figure2_variability.png / .pdf")


if __name__ == "__main__":
    main()
