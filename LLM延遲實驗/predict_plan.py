"""通用版事前預測（2026-10-08）：用某台電腦的閒置效應測試資料，預測它還沒跑的排隊實驗，寫出 plan JSON

和 predict_7b.py 相同的方法：
- 一般服務時間 b：--probe-b 那次測試中，一般條件（A）、g = 0 的請求
- 閒置後多出的時間 δ：--probe-d 那次測試的一般條件，依實際閒置時間分段（<1、1–2、2–5、≥5 秒），
  δ = 該段平均 − 同一次測試 g = 0 的平均（多出的時間在讀 prompt 的階段，和輸出長度無關）
- 負載點：目標 ρ 用 b 換算 λ；用和 load_gen.py 完全相同的抽法重建到達時間，模擬 2,000 次，
  每次都重新抽樣測試資料（bootstrap），所以範圍也包含參數估計的不確定性
- 對照用的天真預測：只用「單獨送一個請求」的服務時間（--probe-b 中 g ≥ 5 秒）代入 P-K

用法（例）：
  python predict_plan.py --probe-b probe_idle_7b16 --probe-d probe_idle_7b --seed 4 --max-tokens 16 \
      --targets b7s4_rho30:0.3:200,b7s4_rho50:0.5:300,b7s4_rho70:0.7:300 --local 7b --out plan_7b_seed4.json
  python predict_plan.py --probe-b probe_m1 --seed 5 --max-tokens 128 --base-url http://192.168.0.20:8080 \
      --targets m1_rho30:0.3:150,m1_rho50:0.5:200,m1_rho70:0.7:200 --out plan_m1.json
"""
import argparse
import csv
import json
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = 2000
EDGES = [1.0, 2.0, 5.0]


def rows_of(tag):
    return [r for r in csv.DictReader(open(os.path.join(HERE, "out", f"{tag}.csv"), encoding="utf-8"))
            if r["cond"] == "A"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-b", required=True)
    ap.add_argument("--probe-d", default=None, help="估 δ 用的測試（預設和 --probe-b 相同）")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--prompt-file", default="medium.txt")
    ap.add_argument("--targets", required=True, help="tag:目標ρ:筆數,…")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--local", default=None, help="伺服器在本機時要啟動的模型（3b、7b）；別台電腦就不填")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed + 100)

    rb, rd = rows_of(a.probe_b), rows_of(a.probe_d or a.probe_b)
    S0 = np.array([float(r["service"]) for r in rb if float(r["gap"]) == 0])
    S0d = np.array([float(r["service"]) for r in rd if float(r["gap"]) == 0])
    by_bin = [[] for _ in range(4)]
    for r in rd:
        if float(r["gap"]) > 0:
            by_bin[int(np.digitize(float(r["idle_before"]), EDGES))].append(float(r["service"]))
    by_bin = [np.array(x) if x else np.array([S0d.mean()]) for x in by_bin]   # 沒測到的段當作沒有額外時間
    iso = np.array([float(r["service"]) for r in rb if float(r["gap"]) >= 5])
    print(f"一般服務時間 b = {S0.mean():.3f} 秒（n={len(S0)}，CV {S0.std() / S0.mean():.3f}）")
    for i, lab in enumerate(["<1s", "1–2s", "2–5s", "≥5s"]):
        print(f"  閒置 {lab:>5}: n={len(by_bin[i]):>3}  δ = {by_bin[i].mean() - S0d.mean():+.3f} 秒")
    print(f"單獨送一個請求（g≥5 秒）的平均服務時間 = {iso.mean():.3f} 秒（n={len(iso)}）")

    def schedule(lam, n):
        r, t, arr = random.Random(a.seed), 0.0, []
        for _ in range(n):
            t += r.expovariate(lam)
            r.choices([0], weights=[1.0], k=1)
            r.choices([a.max_tokens], weights=[1.0], k=1)
            arr.append(t)
        return np.array(arr)

    def sim(arr):
        n, k = len(arr), len(arr) // 3
        pool = S0[rng.integers(0, len(S0), (R, len(S0)))]
        base = pool[np.arange(R)[:, None], rng.integers(0, len(S0), (R, n))]
        pd = S0d[rng.integers(0, len(S0d), (R, len(S0d)))].mean(1)
        dv = np.stack([by_bin[i][rng.integers(0, len(by_bin[i]), (R, len(by_bin[i])))].mean(1) for i in range(4)],
                      1) - pd[:, None]
        c_prev, acc = np.full(R, -np.inf), np.zeros(R)
        for i in range(n):
            start = np.maximum(arr[i], c_prev)
            w = start - arr[i]
            pen = np.where(w == 0, dv[np.arange(R), np.digitize(arr[i] - c_prev, EDGES)], 0.0)
            c_prev = start + base[:, i] + pen
            if i >= k:
                acc += w
        return acc / (n - k)

    b = S0.mean()
    plan = {"seed": a.seed, "max_tokens": a.max_tokens, "prompt_file": a.prompt_file, "base_url": a.base_url,
            "local": a.local, "probe_b": a.probe_b, "probe_d": a.probe_d or a.probe_b,
            "b": round(float(b), 4), "iso": round(float(iso.mean()), 4), "runs": []}
    print(f"\n{'組別':<12}{'目標ρ':>6}{'λ':>9}{'n':>5}{'預估分鐘':>9}{'λ̂/λ':>7}{'天真P-K':>9}{'模型預測':>9}{'95%範圍':>16}")
    for spec in a.targets.split(","):
        tag, rho, n = spec.split(":")
        rho, n = float(rho), int(n)
        lam = round(rho / b, 5)
        arr = schedule(lam, n)
        k = n // 3
        lh = (n - k - 1) / (arr[-1] - arr[k])
        es, es2 = iso.mean(), (iso ** 2).mean()
        naive = lam * es2 / (2 * (1 - lam * es)) if lam * es < 1 else float("inf")
        s = sim(arr)
        lo, hi = np.percentile(s, [2.5, 97.5])
        plan["runs"].append({"tag": tag, "rate": lam, "n": n, "pred": round(float(s.mean()), 3),
                             "lo": round(float(lo), 3), "hi": round(float(hi), 3), "naive_pk": round(naive, 3),
                             "minutes": round(arr[-1] / 60, 1)})
        print(f"{tag:<12}{rho:>6.2f}{lam:>9.5f}{n:>5}{arr[-1] / 60:>9.0f}{lh / lam:>7.3f}{naive:>9.3f}"
              f"{s.mean():>9.3f}{f'[{lo:.2f}, {hi:.2f}]':>16}")
    json.dump(plan, open(os.path.join(HERE, a.out), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
