"""新第 VI 節：保溫 τ 秒的延遲和耗電取捨（2026-10-19）

模型
- 驅動程式本來就會在閒置後維持高時脈約 2 秒（H = 2）；保溫 τ 秒＝在這 2 秒之後再多維持 τ 秒（耗電 P_w）
- 之後顯示卡照預設的方式降頻：閒置 I 秒的代價是 δ(I − τ)（δ 在 2 秒內本來就是 0），
  閒置耗電：前 2 秒和預設相同（含 nvidia-smi 讀數延遲造成的第一個 0.5 秒偏高）、接著 τ 秒是 P_w、之後是預設曲線往後移 τ 秒
- τ = 0 就是預設，τ = ∞ 就是一直保溫（鎖時脈）
- Poisson 到達：閒置時間 I 是指數分布（速率 λ），沒有記憶性，所以
    E[δ_τ(I)] = q·E[δ(I)]，E[δ_τ(I)²] = q·E[δ(I)²]，q = e^(−λτ)（閒置超過 τ 的機率）
  代進論文的公式 (3)（第一個顧客服務時間特別的 M/G/1）：
    b0 = b + q·m1，b02 = b2 + 2b·q·m1 + q·m2
    W = λb2/(2(1−ρ)) + λ(b02 − b2)/(2(1 − ρ + λb0))，ρ = λb
    P0 = (1 − ρ)/(1 − ρ + λb0)（到達時伺服器閒著的機率，也是閒著的時間比例）
    回應時間 T = W + b + P0·q·m1（閒著時到達的請求要多付平均 q·m1）
- 耗電：每段閒置的平均能量 E_idle = ∫ p(s)·P(I > s) ds = A + P_w·e^(−λH)·(1 − q)/λ + q·B，
    A = ∫_0^H P_d(u)·e^(−λu) du（前 2 秒），B = ∫_H^∞ P_d(u)·e^(−λu) du（降頻的部分）
    平均功耗 P̄ = (1 − P0)·P_busy + λ·P0·E_idle（每單位時間有 λ·P0 段閒置）
- 只用到指數分布的基本積分；δ 是分段常數（論文的四段），P_d 是每 0.5 秒一段的曲線

驗證
1. 公式本身：和長時間的事件模擬比（同樣的假設），檢查算得對不對（和驗證公式 (3) 的做法相同）
2. 兩個端點（τ = 0、τ = ∞）和 10/19 的實測比（事前預測 6/6 命中的那兩組）

10/20 改（看過 τ = 3、6、12 的實測之後）
- 實際的保溫做法是「閒置一開始鎖住時脈、τ 秒後放開」，放開後馬上降頻，沒有上面假設的「驅動程式再撐 2 秒」，
  放開後到達的代價也比較大（0.31–0.41 秒）。中間的 τ 改用 closed_form_lockunlock：同樣的推導、同樣只透過 q 變化，
  參數來自 10/19 晚上的校正（驗證 1b：和模擬比；驗證 3：和三個中間點比）
- 取捨改成兩端（預設、一直保溫）之間的直線：在兩種設定之間切換（一直保溫的時間比例 f）就能落在直線上的任一點；
  鎖住再放開的三個點都在直線上或稍微上方，沒有比較好

用法：python keepwarm_model.py
"""
import csv
import datetime
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EDGES = [1.0, 2.0, 5.0]
BIN = 0.5                 # 功耗曲線每段 0.5 秒
NBIN = 16                 # 前 8 秒用曲線，之後用尾端的值
H = 2.0                   # 驅動程式本來維持高時脈的時間（秒）


def parse(s):
    return datetime.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def params(tag):
    """從閒置測試取參數：服務時間、δ 的四段、請求中功耗、閒置開始後的功耗曲線、尾端（P8）功耗"""
    rows = sorted([r for r in csv.DictReader(open(os.path.join(HERE, "out", f"{tag}.csv"), encoding="utf-8"))
                   if r["cond"] == "A"], key=lambda r: float(r["wall_send"]))
    S0 = np.array([float(r["service"]) for r in rows if float(r["gap"]) == 0])
    by = [[] for _ in range(4)]
    for r in rows:
        if float(r["gap"]) > 0:
            by[int(np.digitize(float(r["idle_before"]), EDGES))].append(float(r["service"]))
    d = np.array([np.mean(x) - S0.mean() if x else 0.0 for x in by])
    g = []
    for p in csv.reader(open(os.path.join(HERE, "out", f"{tag}_gpu.csv"), encoding="utf-8")):
        try:
            g.append((parse(p[0]), p[1].strip(), float(p[5])))
        except (ValueError, IndexError):
            continue
    ts, ps, pw = np.array([x[0] for x in g]), np.array([x[1] for x in g]), np.array([x[2] for x in g])
    busy = np.zeros(len(ts), bool)
    for r in rows:
        ws, sv = float(r["wall_send"]), float(r["service"])
        busy |= (ts >= ws + 0.1) & (ts <= ws + sv - 0.05)
    prof = {}
    for j in range(1, len(rows)):
        if float(rows[j]["gap"]) != 8.0:
            continue
        a = float(rows[j]["wall_send"]) - float(rows[j]["idle_before"])
        for k in range(NBIN):
            sel = (ts >= a + BIN * k) & (ts < a + BIN * (k + 1))
            if sel.any():
                prof.setdefault(k, []).append(pw[sel].mean())
    profile = np.array([np.mean(prof[k]) for k in range(NBIN)])
    p8 = pw[ps == "P8"]
    tail = float(np.median(p8)) if len(p8) else float(profile[-1])
    return {"S0": S0, "b": float(S0.mean()), "b2": float((S0 ** 2).mean()), "d": d, "p_busy": float(pw[busy].mean()),
            "profile": profile, "tail": tail}


def bin_probs(lam):
    """閒置時間落在四段（<1、1–2、2–5、≥5 秒）的機率"""
    e = np.array([0.0] + EDGES + [np.inf])
    return np.exp(-lam * e[:-1]) - np.exp(-lam * e[1:])


def idle_energy_parts(lam, profile, tail):
    """A = ∫_0^H P_d e^(−λu) du，B = ∫_H^∞ P_d e^(−λu) du（預設曲線，每 0.5 秒一段，8 秒以後是尾端）"""
    k = np.arange(NBIN)
    seg = (np.exp(-lam * BIN * k) - np.exp(-lam * BIN * (k + 1))) / lam
    h = int(H / BIN)
    A = float((profile[:h] * seg[:h]).sum())
    B = float((profile[h:] * seg[h:]).sum() + tail * np.exp(-lam * BIN * NBIN) / lam)
    return A, B


def closed_form(lam, tau, pr, p_warm, p_busy=None):
    """保溫 τ 秒時的平均等待 W、平均回應時間 T、到達時閒著的機率 P0、平均功耗 P̄（τ 可以是 np.inf）"""
    b, b2, d = pr["b"], pr["b2"], pr["d"]
    pb = bin_probs(lam)
    m1, m2 = float((d * pb).sum()), float((d ** 2 * pb).sum())
    q = 0.0 if np.isinf(tau) else float(np.exp(-lam * tau))
    rho = lam * b
    b0, b02 = b + q * m1, b2 + 2 * b * q * m1 + q * m2
    W = lam * b2 / (2 * (1 - rho)) + lam * (b02 - b2) / (2 * (1 - rho + lam * b0))
    P0 = (1 - rho) / (1 - rho + lam * b0)
    T = W + b + P0 * q * m1
    A, B = idle_energy_parts(lam, pr["profile"], pr["tail"])
    E_idle = A + p_warm * np.exp(-lam * H) * (1 - q) / lam + q * B
    P = (1 - P0) * (pr["p_busy"] if p_busy is None else p_busy) + lam * P0 * E_idle
    return {"W": W, "T": T, "P0": P0, "P": P, "q": q, "m1": m1, "m2": m2}


def simulate(lam, tau, pr, p_warm, n=100_000, seed=1):
    """長時間的事件模擬（同樣的假設），用來檢查封閉解：服務時間從閒置 0 秒的請求抽，閒著時到達加 δ_τ(I)"""
    rng = np.random.default_rng(seed)
    arr = np.cumsum(rng.exponential(1 / lam, n))
    S = rng.choice(pr["S0"], n)
    d, prof, tail, pb = pr["d"], pr["profile"], pr["tail"], pr["p_busy"]
    cum = np.concatenate([[0.0], np.cumsum(prof * BIN)])

    def e_def(u):                      # 預設曲線從 0 到 u 秒的能量
        u = min(u, BIN * NBIN)
        k = int(u // BIN)
        return cum[k] + (u - BIN * k) * prof[min(k, NBIN - 1)]

    c_prev, W, T, energy, idle_n = -np.inf, np.zeros(n), np.zeros(n), 0.0, 0
    t_first = None
    for i in range(n):
        if arr[i] >= c_prev:                                 # 到達時伺服器閒著
            I = arr[i] - c_prev if np.isfinite(c_prev) else np.inf
            if np.isfinite(I):
                idle_n += 1
                if I <= H:
                    energy += e_def(I)
                elif I <= H + tau:
                    energy += e_def(H) + p_warm * (I - H)
                else:
                    u = I - tau                                  # 降頻從 H 開始，往後移了 τ
                    energy += p_warm * tau + e_def(u) + max(u - BIN * NBIN, 0.0) * tail
                pen = d[int(np.digitize(I - tau, EDGES))] if I > tau else 0.0
            else:
                pen = d[3]
                t_first = arr[i]
            start = arr[i]
        else:
            start, pen = c_prev, 0.0
        svc = S[i] + pen
        energy += pb * svc
        c_prev = start + svc
        W[i], T[i] = start - arr[i], c_prev - arr[i]
    k = n // 10                                              # 丟掉開頭 10%
    return {"W": float(W[k:].mean()), "T": float(T[k:].mean()), "P": energy / (c_prev - t_first)}


# ---------------------------------------------------------------- 實際的保溫做法：閒置一開始鎖住、τ 秒後放開（10/20 加）
def lockunlock_params():
    """參數全部來自 10/19 晚上的校正（kw_cal.csv，run_keepwarm.cal_params；三組之前就固定）：
    一個接一個的服務時間、保溫中到達的 δ_w、放開後 u 秒到達的 δ_after(u)（四段）、三種請求的請求中功耗、鎖住和放開後的閒置功耗曲線"""
    import run_keepwarm as rk
    rows = [{k: (float(v) if k in ("service", "wall_send", "idle_before", "gap") or (k == "u_after_unlock" and v != "") else v)
             for k, v in r.items()} for r in csv.DictReader(open(os.path.join(HERE, "out", "kw_cal.csv"), encoding="utf-8"))]
    return rk.cal_params(rows)


def _lap(lam, prof, tail, t_from=0.0, t_to=np.inf):
    """∫ p(s)·e^(−λs) ds，從 t_from 到 t_to；p 每 0.5 秒一段，曲線之後是 tail"""
    k = np.arange(len(prof))
    lo, hi = np.clip(BIN * k, t_from, t_to), np.clip(BIN * (k + 1), t_from, t_to)
    v = float((prof * (np.exp(-lam * lo) - np.exp(-lam * hi)) / lam).sum())
    end = BIN * len(prof)
    if t_to > end:
        v += tail * (np.exp(-lam * max(end, t_from)) - np.exp(-lam * t_to)) / lam
    return v


def closed_form_lockunlock(lam, tau, P):
    """鎖住再放開的封閉解：和 closed_form 同樣的推導（Poisson 沒有記憶性，代價的一、二階動差只透過 q 變化），差別是
    保溫中到達付 δ_w、放開後 u 秒到達付 δ_after(u)（放開後馬上降頻，沒有驅動程式的 2 秒），請求中功耗分三種，
    閒置耗電 = 鎖住曲線積到 τ + q × 放開後曲線"""
    b, b2 = P["S0"].mean(), (P["S0"] ** 2).mean()
    dw = P["warm"].mean() - b
    da = np.array([x.mean() for x in P["cold_by"]]) - b
    pb = bin_probs(lam)
    m1, m2 = float((da * pb).sum()), float((da ** 2 * pb).sum())
    q = 0.0 if np.isinf(tau) else float(np.exp(-lam * tau))
    rho = lam * b
    e1 = (1 - q) * dw + q * m1                                  # 閒著時到達的請求平均多付
    b0, b02 = b + e1, b2 + 2 * b * e1 + (1 - q) * dw ** 2 + q * m2
    W = lam * b2 / (2 * (1 - rho)) + lam * (b02 - b2) / (2 * (1 - rho + lam * b0))
    P0 = (1 - rho) / (1 - rho + lam * b0)
    T = W + b + P0 * e1
    pq, pw_, pc = P["p_busy"]["queued"], P["p_busy"]["warm"], P["p_busy"]["cold"]
    E_req = (1 - P0) * pq * b + P0 * (1 - q) * pw_ * (b + dw) + P0 * q * pc * (b + m1)
    E_idle = _lap(lam, P["lock_prof"], P["lock_flat"], 0.0, tau) + q * _lap(lam, P["unl_prof"], P["unl_tail"])
    return {"W": W, "T": T, "P0": P0, "P": lam * E_req + lam * P0 * E_idle, "q": q}


def simulate_lockunlock(lam, tau, P, n=100_000, seed=1):
    """長時間的事件模擬（和 closed_form_lockunlock 同樣的假設，δ 用各段的平均），用來檢查封閉解"""
    rng = np.random.default_rng(seed)
    arr = np.cumsum(rng.exponential(1 / lam, n))
    S = rng.choice(P["S0"], n)
    b = P["S0"].mean()
    dw = P["warm"].mean() - b
    da = np.array([x.mean() for x in P["cold_by"]]) - b
    lp, ul = P["lock_prof"], P["unl_prof"]
    lcum, ucum = np.concatenate([[0.0], np.cumsum(lp * BIN)]), np.concatenate([[0.0], np.cumsum(ul * BIN)])

    def e_curve(u, prof, cum, tail):
        a = min(u, BIN * len(prof))
        k = int(a // BIN)
        return cum[k] + (a - BIN * k) * prof[min(k, len(prof) - 1)] + max(u - BIN * len(prof), 0.0) * tail

    c_prev, T, energy, t_first = -np.inf, np.zeros(n), 0.0, None
    for i in range(n):
        if arr[i] >= c_prev:
            I = arr[i] - c_prev if np.isfinite(c_prev) else np.inf
            if np.isfinite(I):
                if I <= tau:
                    energy += e_curve(I, lp, lcum, P["lock_flat"])
                    pen, pb = dw, P["p_busy"]["warm"]
                else:
                    energy += e_curve(tau, lp, lcum, P["lock_flat"]) + e_curve(I - tau, ul, ucum, P["unl_tail"])
                    pen, pb = da[int(np.digitize(I - tau, EDGES))], P["p_busy"]["cold"]
            else:
                pen, pb, t_first = da[3], P["p_busy"]["cold"], arr[i]
            start = arr[i]
        else:
            start, pen, pb = c_prev, 0.0, P["p_busy"]["queued"]
        svc = S[i] + pen
        energy += pb * svc
        c_prev = start + svc
        T[i] = c_prev - arr[i]
    k = n // 10
    return {"T": float(T[k:].mean()), "P": energy / (c_prev - t_first)}


def main():
    pd_, pl = params("probe_base_1011"), params("probe_locked_1011")
    p_warm = float(pl["profile"][2:].mean())                 # 鎖時脈後閒著的功耗（閒置 1 秒以後的平均）
    out = {"B1": {"default": {"p_busy_W": round(pd_["p_busy"], 1), "p8_W": round(pd_["tail"], 1),
                              "idle_profile_W": [round(float(x), 1) for x in pd_["profile"]]},
                  "locked": {"p_busy_W": round(pl["p_busy"], 1), "p_warm_idle_W": round(p_warm, 1)},
                  "delta_steps_s": {"<1s": round(float(pd_["d"][0]), 3), "1-2s": round(float(pd_["d"][1]), 3),
                                    "2-5s": round(float(pd_["d"][2]), 3), ">=5s": round(float(pd_["d"][3]), 3)},
                  "b_default": round(pd_["b"], 4), "b_locked": round(pl["b"], 4)}}
    print("== B1 各狀態的功耗（10/11 閒置測試）")
    print(f"  請求中：預設 {pd_['p_busy']:.1f} W、鎖時脈 {pl['p_busy']:.1f} W｜鎖時脈閒著 {p_warm:.1f} W｜P8 {pd_['tail']:.1f} W")
    print(f"  預設閒置開始後的功耗（每 0.5 秒）：{np.round(pd_['profile'], 1).tolist()}")
    print(f"  δ 四段：{out['B1']['delta_steps_s']}；b 預設 {pd_['b']:.3f}、鎖時脈 {pl['b']:.3f} 秒")

    print("\n== 驗證 1：封閉解 vs 長時間模擬（同樣假設，10 萬個請求）")
    checks = []
    for rho in (0.1, 0.2, 0.5):
        lam = rho / pd_["b"]
        for tau in (0.0, 2.0, 5.0, 15.0, np.inf):
            cf = closed_form(lam, tau, pd_, p_warm)
            sm = simulate(lam, tau, pd_, p_warm)
            checks.append({"rho": rho, "tau": tau if np.isfinite(tau) else "inf", "T_formula": cf["T"], "T_sim": sm["T"],
                           "P_formula": cf["P"], "P_sim": sm["P"], "W_formula": cf["W"], "W_sim": sm["W"]})
            print(f"  ρ={rho:.1f} τ={'∞' if np.isinf(tau) else f'{tau:g}':>3}：回應時間 公式 {cf['T']:.4f}／模擬 {sm['T']:.4f}"
                  f"（{100 * (cf['T'] / sm['T'] - 1):+.2f}%）｜平均功耗 公式 {cf['P']:.2f}／模擬 {sm['P']:.2f} W"
                  f"（{100 * (cf['P'] / sm['P'] - 1):+.2f}%）")
    out["check_vs_sim"] = checks

    print("\n== 驗證 2：兩個端點 vs 10/19 實測（ρ = 0.2）")
    res = json.load(open(os.path.join(HERE, "out", "power_results.json"), encoding="utf-8"))
    lam = 0.10885
    e0 = closed_form(lam, 0.0, pd_, p_warm)
    einf = closed_form(lam, np.inf, pl, p_warm, p_busy=pl["p_busy"])
    out["endpoints"] = {"default": {"T_formula": e0["T"], "P_formula": e0["P"], "T_meas": res["default"]["resp"], "P_meas": res["default"]["power_W"]},
                        "locked": {"T_formula": einf["T"], "P_formula": einf["P"], "T_meas": res["locked"]["resp"], "P_meas": res["locked"]["power_W"]}}
    for k, v in out["endpoints"].items():
        print(f"  {k:>7}（設定的 λ）：回應時間 公式 {v['T_formula']:.3f}／實測 {v['T_meas']:.3f}（{100 * (v['T_formula'] / v['T_meas'] - 1):+.1f}%）｜"
              f"平均功耗 公式 {v['P_formula']:.1f}／實測 {v['P_meas']:.1f} W（{100 * (v['P_formula'] / v['P_meas'] - 1):+.1f}%）")
    # 那次 180 個請求的實際到達率（量測區間，和 load_gen.py 相同的抽法）比設定的低；用實際到達率再比一次
    import random
    r_, t, arr = random.Random(11), 0.0, []
    for _ in range(180):
        t += r_.expovariate(lam)
        r_.choices([0], weights=[1.0], k=1)
        r_.choices([128], weights=[1.0], k=1)
        arr.append(t)
    lam_hat = (180 - 60 - 1) / (arr[-1] - arr[60])
    e0h = closed_form(lam_hat, 0.0, pd_, p_warm)
    einfh = closed_form(lam_hat, np.inf, pl, p_warm, p_busy=pl["p_busy"])
    out["endpoints_realized_lambda"] = {"lambda_hat": lam_hat, "ratio": lam_hat / lam,
                                        "default": {"T": e0h["T"], "P": e0h["P"]}, "locked": {"T": einfh["T"], "P": einfh["P"]}}
    print(f"  實際到達率 λ̂ = {lam_hat:.5f}（設定的 {lam_hat / lam:.3f} 倍）：")
    for k, e in (("default", e0h), ("locked", einfh)):
        m = out["endpoints"][k]
        print(f"  {k:>7}（實際的 λ̂）：回應時間 公式 {e['T']:.3f}／實測 {m['T_meas']:.3f}（{100 * (e['T'] / m['T_meas'] - 1):+.1f}%）｜"
              f"平均功耗 公式 {e['P']:.1f}／實測 {m['P_meas']:.1f} W（{100 * (e['P'] / m['P_meas'] - 1):+.1f}%）")

    # 10/20 改：實際的保溫做法（鎖住 τ 秒再放開）放開後馬上降頻，不是原本假設的「驅動程式再撐 2 秒」，
    # 所以中間的 τ 改用 closed_form_lockunlock（參數來自 10/19 晚上的校正）；取捨改成兩端之間的直線
    pk = lockunlock_params()
    b_k = float(pk["S0"].mean())
    print("\n== 驗證 1b：鎖住再放開的封閉解 vs 長時間模擬（同樣假設，10 萬個請求）")
    checks_k = []
    for rho in (0.1, 0.2, 0.5):
        lam = rho / b_k
        for tau in (0.0, 3.0, 6.0, 12.0, np.inf):
            cf, sm = closed_form_lockunlock(lam, tau, pk), simulate_lockunlock(lam, tau, pk)
            checks_k.append({"rho": rho, "tau": tau if np.isfinite(tau) else "inf", "T_formula": cf["T"], "T_sim": sm["T"],
                             "P_formula": cf["P"], "P_sim": sm["P"]})
            print(f"  ρ={rho:.1f} τ={'∞' if np.isinf(tau) else f'{tau:g}':>3}：回應時間 公式 {cf['T']:.4f}／模擬 {sm['T']:.4f}"
                  f"（{100 * (cf['T'] / sm['T'] - 1):+.2f}%）｜平均功耗 公式 {cf['P']:.2f}／模擬 {sm['P']:.2f} W（{100 * (cf['P'] / sm['P'] - 1):+.2f}%）")

    print(f"\n== 驗證 3：三個中間點（10/19 晚上，同一串到達時間）")
    interior = []
    seg_d, seg_l = res["default"], res["locked"]
    slope = (seg_l["resp"] - seg_d["resp"]) / (seg_l["power_W"] - seg_d["power_W"])
    for tau in (3, 6, 12):
        m = res[f"kw_tau{tau}"]
        pr = json.load(open(os.path.join(HERE, f"plan_kw_tau{tau}.json"), encoding="utf-8"))["runs"][0]
        cf = closed_form_lockunlock(lam_hat, float(tau), pk)
        gap = m["resp"] - (seg_d["resp"] + slope * (m["power_W"] - seg_d["power_W"]))
        interior.append({"tau": tau, "T_meas": m["resp"], "P_meas": m["power_W"], "T_pred": pr["resp_pred"], "T_lo": pr["resp_lo"],
                         "T_hi": pr["resp_hi"], "P_pred": pr["power_pred_W"], "T_formula": cf["T"], "P_formula": cf["P"],
                         "gap_to_line_s": gap})
        print(f"  τ = {tau:>2}：實測 {m['resp']:.3f} 秒、{m['power_W']:.1f} W｜事前預測 {pr['resp_pred']}（{pr['resp_lo']}–{pr['resp_hi']}）、"
              f"{pr['power_pred_W']} W｜封閉解（λ̂）{cf['T']:.3f} 秒（{100 * (cf['T'] / m['resp'] - 1):+.1f}%）、{cf['P']:.1f} W"
              f"（{100 * (cf['P'] / m['power_W'] - 1):+.1f}%）｜比兩端實測連線慢 {gap:+.3f} 秒（沒有換算早晚的功耗差）")
    out["lockunlock"] = {"b": b_k, "warm_delta": float(pk["warm"].mean() - b_k),
                         "delta_after": [float(x.mean() - b_k) for x in pk["cold_by"]], "p_busy_W": pk["p_busy"],
                         "lock_idle_W": pk["lock_flat"], "check_vs_sim": checks_k, "interior": interior}

    print("\n== 取捨：兩端之間的直線（預設 τ = 0 用預設的參數；一直保溫用鎖時脈自己的參數）")
    switching = {}
    for rho in (0.1, 0.2, 0.3, 0.5):
        lam = rho / pd_["b"]
        a = closed_form(lam, 0.0, pd_, p_warm)
        z = closed_form(lam, np.inf, pl, p_warm, p_busy=pl["p_busy"])
        per = (z["P"] - a["P"]) / (10 * (a["T"] - z["T"]))
        switching[str(rho)] = {"T_default": a["T"], "P_default": a["P"], "T_locked": z["T"], "P_locked": z["P"], "W_per_100ms": per}
        print(f"  ρ={rho:.1f}：預設 {a['T']:.3f} 秒、{a['P']:.1f} W｜一直保溫 {z['T']:.3f} 秒、{z['P']:.1f} W｜"
              f"省 {a['T'] - z['T']:.3f} 秒、多 {z['P'] - a['P']:.1f} W（每省 0.1 秒要多 {per:.1f} W）")
    out["switching"] = switching

    print("\n== 延遲目標（ρ = 0.2）：在兩種設定之間切換，一直保溫的時間比例 f 和平均功耗")
    s = switching["0.2"]
    targets = []
    for target in (2.25, 2.20, 2.15, 2.10, 2.05):
        f = (s["T_default"] - target) / (s["T_default"] - s["T_locked"])
        ok = 0 <= f <= 1
        targets.append({"target_T": target, "f": f if ok else None, "P": s["P_default"] + f * (s["P_locked"] - s["P_default"]) if ok else None})
        print(f"  回應時間 ≤ {target:.2f} 秒：" + (f"f = {f:.2f}，平均功耗 {targets[-1]['P']:.1f} W" if ok else "做不到"))
    out["targets_rho20"] = targets
    json.dump(out, open(os.path.join(HERE, "out", "keepwarm_model.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False,
              default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print("\n→ out/keepwarm_model.json")


if __name__ == "__main__":
    main()
