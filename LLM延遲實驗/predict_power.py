"""保溫取捨的事前預測（2026-10-19，新第 VI 節）：預設和鎖時脈在同一個低負載下，平均等待時間和顯示卡平均功耗各是多少

延遲的算法和 predict_plan.py 完全相同（同一個函式邏輯）：
- b、δ 用 10/11 的閒置測試：預設用 probe_base_1011，鎖時脈用 probe_locked_1011
- 用和 load_gen.py 相同的抽法重建到達時間，模擬 2,000 次（每次重新抽樣測試資料）
- 兩組用同一個 λ（照預設的 b 換算 ρ = 0.2），才能直接比

功耗的算法（新的）：
- 請求進行中：那次測試請求進行中的平均功耗（nvidia-smi 的 power.draw）
- 閒置時：那次測試「閒置開始後每 0.5 秒」的平均功耗曲線（用閒置 8 秒的那 12 段），超過 8 秒用 P8 的功耗（鎖時脈用 8 秒時的值）
- nvidia-smi 的讀數有延遲，所以請求剛結束的 0.5 秒讀數偏高、請求剛開始的讀數偏低；兩邊都用同一種讀數，加總起來的能量不受影響
- 平均功耗 = 量測區間（丟掉前 1/3 的請求，和延遲相同）的總能量 ÷ 總時間

用法：python predict_power.py
"""
import csv
import datetime
import json
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = 2000
EDGES = [1.0, 2.0, 5.0]
SEED, MAX_TOKENS, N, RHO = 11, 128, 180, 0.2


def parse(s):
    return datetime.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def load(tag):
    rows = sorted([r for r in csv.DictReader(open(os.path.join(HERE, "out", f"{tag}.csv"), encoding="utf-8"))
                   if r["cond"] == "A"], key=lambda r: float(r["wall_send"]))
    g = []
    for p in csv.reader(open(os.path.join(HERE, "out", f"{tag}_gpu.csv"), encoding="utf-8")):
        try:
            g.append((parse(p[0]), p[1].strip(), float(p[5])))
        except (ValueError, IndexError):
            pass
    ts = np.array([x[0] for x in g])
    pw = np.array([x[2] for x in g])
    ps = np.array([x[1] for x in g])
    busy = np.zeros(len(ts), bool)
    for r in rows:
        ws, sv = float(r["wall_send"]), float(r["service"])
        busy |= (ts >= ws + 0.1) & (ts <= ws + sv - 0.05)
    prof = {}
    for j in range(1, len(rows)):
        r = rows[j]
        if float(r["gap"]) != 8.0:
            continue
        a = float(r["wall_send"]) - float(r["idle_before"])
        for k in range(16):
            sel = (ts >= a + 0.5 * k) & (ts < a + 0.5 * (k + 1))
            if sel.any():
                prof.setdefault(k, []).append(pw[sel].mean())
    profile = np.array([np.mean(prof[k]) for k in range(16)])
    p8 = pw[ps == "P8"]
    tail = float(np.median(p8)) if len(p8) else float(profile[-1])
    return rows, float(pw[busy].mean()), profile, tail


def main():
    out = {}
    tags = {"default": "probe_base_1011", "locked": "probe_locked_1011"}
    data = {k: load(t) for k, t in tags.items()}
    S0 = {k: np.array([float(r["service"]) for r in d[0] if float(r["gap"]) == 0]) for k, d in data.items()}
    lam = round(RHO / S0["default"].mean(), 5)            # 兩組同一個 λ
    rng = np.random.default_rng(SEED + 100)

    r_, t, arr = random.Random(SEED), 0.0, []              # 和 load_gen.py 相同的抽法
    for _ in range(N):
        t += r_.expovariate(lam)
        r_.choices([0], weights=[1.0], k=1)
        r_.choices([MAX_TOKENS], weights=[1.0], k=1)
        arr.append(t)
    arr = np.array(arr)
    k0 = N // 3

    for cond, (rows, p_busy, profile, tail) in data.items():
        s0 = S0[cond]
        by_bin = [[] for _ in range(4)]
        for r in rows:
            if float(r["gap"]) > 0:
                by_bin[int(np.digitize(float(r["idle_before"]), EDGES))].append(float(r["service"]))
        by_bin = [np.array(x) if x else np.array([s0.mean()]) for x in by_bin]
        pool = s0[rng.integers(0, len(s0), (R, len(s0)))]
        base = pool[np.arange(R)[:, None], rng.integers(0, len(s0), (R, N))]
        pd_ = s0[rng.integers(0, len(s0), (R, len(s0)))].mean(1)
        dv = np.stack([by_bin[i][rng.integers(0, len(by_bin[i]), (R, len(by_bin[i])))].mean(1) for i in range(4)], 1) - pd_[:, None]
        cum = np.concatenate([[0.0], np.cumsum(profile * 0.5)])      # 閒置 I 秒的能量：前 8 秒照曲線，之後照 tail

        def idle_energy(I):
            I = np.maximum(I, 0.0)
            full = np.minimum(I, 8.0)
            k = np.floor(full / 0.5).astype(int)
            e = cum[k] + (full - 0.5 * k) * profile[np.minimum(k, 15)]
            return e + np.maximum(I - 8.0, 0.0) * tail

        c_prev, acc_w, acc_t, energy = np.full(R, -np.inf), np.zeros(R), np.zeros(R), np.zeros(R)
        t_start = None
        for i in range(N):
            start = np.maximum(arr[i], c_prev)
            w = start - arr[i]
            idle = np.where(np.isfinite(c_prev), np.maximum(arr[i] - c_prev, 0.0), 0.0)
            pen = np.where(w == 0, dv[np.arange(R), np.digitize(arr[i] - c_prev, EDGES)], 0.0)
            svc = base[:, i] + pen
            if i >= k0:
                if t_start is None:
                    t_start = c_prev.copy()     # 量測區間從前一個請求做完開始，到最後一個做完：總時間＝每段閒置＋每個服務時間
                acc_w += w
                acc_t += w + svc                 # 回應時間＝等待＋服務
                energy += idle_energy(idle) + p_busy * svc
            c_prev = start + svc
        wait = acc_w / (N - k0)
        resp = acc_t / (N - k0)
        rlo, rhi = np.percentile(resp, [2.5, 97.5])
        p_mean = energy / (c_prev - t_start)
        lo, hi = np.percentile(wait, [2.5, 97.5])
        plo, phi = np.percentile(p_mean, [2.5, 97.5])
        out[cond] = {"probe": tags[cond], "b": round(float(s0.mean()), 4), "lambda": lam, "rho": round(lam * s0.mean(), 4),
                     "wait_pred": round(float(wait.mean()), 3), "wait_lo": round(float(lo), 3), "wait_hi": round(float(hi), 3),
                     "resp_pred": round(float(resp.mean()), 3), "resp_lo": round(float(rlo), 3), "resp_hi": round(float(rhi), 3),
                     "power_pred_W": round(float(p_mean.mean()), 2), "power_lo_W": round(float(plo), 2), "power_hi_W": round(float(phi), 2),
                     "p_busy_W": round(p_busy, 1), "p_idle_profile_W": [round(float(x), 1) for x in profile], "p_tail_W": round(tail, 1),
                     "energy_per_request_J": round(float((energy / (N - k0)).mean()), 1)}
        print(f"{cond:>8}：b = {s0.mean():.3f} 秒，ρ = {lam * s0.mean():.3f}｜平均等待 {wait.mean():.3f} 秒 [{lo:.3f}, {hi:.3f}]｜"
              f"平均回應時間 {resp.mean():.3f} 秒 [{rlo:.3f}, {rhi:.3f}]｜"
              f"平均功耗 {p_mean.mean():.1f} W [{plo:.1f}, {phi:.1f}]｜每個請求 {(energy / (N - k0)).mean():.0f} J")
    print(f"λ = {lam}/秒，{N} 筆（丟掉前 {k0} 筆），約 {arr[-1] / 60:.0f} 分鐘；種子 {SEED}")
    d, l = out["default"], out["locked"]
    print(f"鎖時脈：平均等待少 {d['wait_pred'] - l['wait_pred']:.3f} 秒，平均回應時間少 {d['resp_pred'] - l['resp_pred']:.3f} 秒，"
          f"平均功耗多 {l['power_pred_W'] - d['power_pred_W']:.1f} W")
    for cond in ("default", "locked"):
        plan = {"seed": SEED, "max_tokens": MAX_TOKENS, "prompt_file": "medium.txt", "base_url": "http://127.0.0.1:8080",
                "local": "3b", "probe_b": tags[cond], "probe_d": tags[cond], "b": out[cond]["b"], "condition": cond,
                "runs": [{"tag": f"power_{cond}_rho20", "rate": lam, "n": N, "pred": out[cond]["wait_pred"],
                          "lo": out[cond]["wait_lo"], "hi": out[cond]["wait_hi"], "resp_pred": out[cond]["resp_pred"],
                          "resp_lo": out[cond]["resp_lo"], "resp_hi": out[cond]["resp_hi"], "power_pred_W": out[cond]["power_pred_W"],
                          "power_lo_W": out[cond]["power_lo_W"], "power_hi_W": out[cond]["power_hi_W"],
                          "minutes": round(arr[-1] / 60, 1)}]}
        json.dump(plan, open(os.path.join(HERE, f"plan_power_{cond}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(out, open(os.path.join(HERE, "out", "power_prediction.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
