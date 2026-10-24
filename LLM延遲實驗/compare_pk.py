"""把實測等待時間和 P-K 理論值比較（Day 16–17、Day 25 都會用到）

用法（假伺服器驗證）：
  python compare_pk.py --es 0.5 --es2 0.2708 --warmup 20 out/val_rho50.csv:1.0 out/val_rho80.csv:1.6

參數
  --es    E[S]：平均服務時間（秒）
  --es2   E[S^2]：= 變異數 + 平均值^2
  --warmup 前幾筆當暖機捨棄（預設 20）
  之後每個參數是「CSV 路徑:該次實驗的 λ」

只用標準函式庫。
"""
import argparse
import csv
import random
import statistics as st


def bootstrap_ci(xs, b=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(b))
    return means[int(0.025 * b)], means[int(0.975 * b)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--es", type=float, required=True)
    p.add_argument("--es2", type=float, required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("runs", nargs="+", help="CSV:λ")
    a = p.parse_args()

    print(f"E[S]={a.es}  E[S^2]={a.es2}  暖機捨棄前 {a.warmup} 筆\n")
    print(f"{'ρ':>6}{'λ':>7}{'n':>6}{'實測平均等待':>14}{'95% 信賴區間':>24}{'P-K 理論':>11}{'判定':>8}")
    for item in a.runs:
        path, lam = item.rsplit(":", 1)
        lam = float(lam)
        rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["status"] == "ok"]
        bad = sum(1 for r in csv.DictReader(open(path, encoding="utf-8")) if r["status"] != "ok")
        w = [float(r["ttft"]) for r in rows[a.warmup:] if r["ttft"] != ""]
        rho = lam * a.es
        theory = lam * a.es2 / (2 * (1 - rho)) if rho < 1 else float("inf")
        lo, hi = bootstrap_ci(w)
        ok = lo <= theory <= hi
        print(f"{rho:>6.2f}{lam:>7.2f}{len(w):>6}{st.mean(w):>14.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>24}{theory:>11.3f}{'✅ 吻合' if ok else '⚠️ 不吻合':>8}")
        if bad:
            print(f"        ⚠️ {path} 有 {bad} 筆失敗的請求")
    print("\n判定說明：理論值落在 95% 信賴區間內就算吻合。不吻合時先檢查：")
    print("  1) 是不是 closed-loop（t_send 有沒有明顯晚於 t_scheduled）")
    print("  2) 樣本是否太少（每個負載點至少 150–200 筆）")
    print("  3) E[S]、E[S^2] 有沒有算錯")


if __name__ == "__main__":
    main()
