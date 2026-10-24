"""保溫中間點三組的事後檢查（2026-10-20，看過結果之後寫的，不算事前預測）
1. 資料品質規則（保溫中間點實驗設計.md）：時脈放開晚於預定 1 秒以上的次數
2. 各組實際付的代價：照到達時的狀態（排隊、保溫中、放開後）分開算平均服務時間，和校正值比
3. 封閉解（鎖住再放開、第 VI 節原本的模型）vs 實測
4. 早晚的功耗差：把晚上的鎖著閒置、深度閒置功耗換成早上的值，再和兩端的連線比

用法：python check_keepwarm_runs.py
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TAUS = [3, 6, 12]
EDGES = [1.0, 2.0, 5.0]
LABELS = ["<1s", "1-2s", "2-5s", ">=5s"]
BIN = 0.5                                                        # 功耗曲線每段 0.5 秒（和 run_keepwarm.py 相同）


def load(tag):
    rows = [r for r in csv.DictReader(open(os.path.join(OUT, f"{tag}.csv"), encoding="utf-8")) if r["status"] == "ok"]
    rows.sort(key=lambda r: float(r["t_send"]))
    return rows


def main():
    log = json.load(open(sorted(glob.glob(os.path.join(OUT, "kw_clock_log_*.json")))[-1], encoding="utf-8"))
    cal = list(csv.DictReader(open(os.path.join(OUT, "kw_cal.csv"), encoding="utf-8")))
    b_cal = np.mean([float(r["service"]) for r in cal if r["state"] == "queued"])
    cold = [r for r in cal if r["state"] == "cold"]
    u_cal = np.array([float(r["u_after_unlock"]) for r in cold])
    d_cal = np.array([float(r["service"]) for r in cold]) - b_cal
    bins_cal = np.digitize(u_cal, EDGES)
    d_bin = {i: d_cal[bins_cal == i].mean() for i in range(4)}
    print(f"校正：b = {b_cal:.4f} 秒；放開後的代價 " + "、".join(f"{LABELS[i]} {d_bin[i]:+.3f}（{(bins_cal == i).sum()} 個）" for i in range(4)))

    print("\n== 1. 時脈放開的時間（預定 = 這段閒置開始 + τ）")
    # 時脈已經鎖著時，lock() 不會再記一筆，所以閒置開始的時間要從請求紀錄算：
    # 請求紀錄的時間是相對時間，用 wall_send − t_send 換成和時脈紀錄相同的牆上時間
    for tau in TAUS:
        tag = f"kw_tau{tau}"
        rows = load(tag)
        off = np.median([float(r["wall_send"]) - float(r["t_send"]) for r in rows])
        a = np.array([float(r["t_send"]) for r in rows]) + off
        c = np.array([float(r["t_done"]) for r in rows]) + off
        st = [r["state_at_send"] for r in rows]
        unl = np.array([e[1] for e in log if e[0] == "unlock" and e[4] == f"{tag} idle {tau}s"])
        fails = sum(1 for e in log if e[4].startswith(tag) and not e[3])
        late, wrong_state, extra = [], 0, 0
        busy_end = np.maximum.accumulate(c)
        for i in range(len(rows) - 1):
            if a[i + 1] <= busy_end[i]:                  # 下一個到達時還在忙：沒有閒置
                continue
            t_idle, t_next = busy_end[i], a[i + 1]
            inside = unl[(unl >= t_idle) & (unl < t_next)]
            if t_next - t_idle > tau:
                if len(inside):
                    late.append(inside[0] - (t_idle + tau))
                wrong_state += st[i + 1] != "cold"
            else:
                extra += len(inside)                     # 不該放開卻放開
                wrong_state += st[i + 1] != "warm"
        late = np.array(late)
        print(f"  τ = {tau:>2}：該放開的閒置 {len(late)} 段，比預定晚 中位數 {np.median(late):.3f} 秒、最多 {late.max():.3f} 秒，"
              f"晚 1 秒以上 {(late >= 1).sum()} 次；不該放開卻放開 {extra} 次；到達時狀態對不上 {wrong_state} 個；"
              f"nvidia-smi 呼叫失敗 {fails} 次")

    print("\n== 2. 各組實際付的代價（服務時間 − 同一組排隊請求的平均）")
    for tau in TAUS:
        rows = load(f"kw_tau{tau}")
        a = np.array([float(r["t_send"]) for r in rows])
        c = np.array([float(r["t_done"]) for r in rows])
        prev = np.empty(len(c)); _cp = -np.inf    # 10/22 改：照做完的順序重建（一次只處理一個請求，做完的順序就是處理的順序；原本照送出順序）
        for _j in np.argsort(c, kind="stable"):
            prev[_j] = _cp; _cp = c[_j]
        svc = (c - np.maximum(a, prev))[60:]               # 服務時間 = 完成 − 開始服務；和 analyze_power.py 一樣丟掉前 60 個
        idle = (a - prev)[60:]
        # 照閒置時間分類，不用紀錄裡的 state_at_send：前一個剛做完、還在鎖時脈的那 0.1–0.2 秒內到達的，
        # 紀錄會寫成 cold（其實顯示卡剛忙完）；閒置剛超過 τ、放開的指令還沒完成就到達的，紀錄會寫成 warm
        st = np.where(idle <= 0, "queued", np.where(idle <= tau, "warm", "cold"))
        b_run = svc[st == "queued"].mean()
        line = f"  τ = {tau:>2}：排隊 {np.sum(st == 'queued')} 個 b = {b_run:.3f}；"
        line += f"保溫中到達 {np.sum(st == 'warm')} 個 代價 {svc[st == 'warm'].mean() - b_run:+.3f}；"
        cm = st == "cold"
        u = idle[cm] - tau
        d = svc[cm] - b_run
        bins = np.digitize(u, EDGES)
        expect = np.mean([d_bin[i] for i in bins])
        line += f"放開後到達 {cm.sum()} 個 代價 {d.mean():+.3f}（照校正的分段值、同樣的 u 分布應是 {expect:+.3f}）"
        print(line)
        print("        放開後各段：" + "、".join(f"{LABELS[i]} {d[bins == i].mean():+.3f}（{(bins == i).sum()} 個）"
                                         for i in range(4) if (bins == i).any()))


    closed_form_check()
    session_offset_check()


def lam_hat():
    """那串到達時間（種子 11，和 load_gen.py 相同的抽法）在量測區間的實際到達率"""
    import random
    r_, t, arr = random.Random(11), 0.0, []
    for _ in range(180):
        t += r_.expovariate(0.10885)
        r_.choices([0], weights=[1.0], k=1)
        r_.choices([128], weights=[1.0], k=1)
        arr.append(t)
    return (180 - 60 - 1) / (arr[-1] - arr[60])


def closed_form_check():
    """3. 封閉解 vs 實測（事後計算；參數是校正和 10/11 閒置測試的，都在三組之前就固定）"""
    import keepwarm_model as km
    P, lh = km.lockunlock_params(), lam_hat()
    res = json.load(open(os.path.join(OUT, "power_results.json"), encoding="utf-8"))
    pd_, pl = km.params("probe_base_1011"), km.params("probe_locked_1011")
    p_warm = float(pl["profile"][2:].mean())
    print(f"\n== 3. 封閉解 vs 實測（事後計算，實際到達率 λ̂ = {lh:.4f}）")
    print("   (a) 鎖住再放開（keepwarm_model.closed_form_lockunlock，校正的參數）　(b) 第 VI 節原本的模型（假設放開後驅動程式還會撐 2 秒），同一個 τ")
    for tau in TAUS:
        m = res[f"kw_tau{tau}"]
        a_ = km.closed_form_lockunlock(lh, float(tau), P)
        b_ = km.closed_form(lh, float(tau), pd_, p_warm)
        print(f"  τ = {tau:>2}：實測 {m['resp']:.3f} 秒、{m['power_W']:.1f} W｜(a) {a_['T']:.3f} 秒（{100 * (a_['T'] / m['resp'] - 1):+.1f}%）、"
              f"{a_['P']:.1f} W（{100 * (a_['P'] / m['power_W'] - 1):+.1f}%）｜(b) {b_['T']:.3f} 秒（{100 * (b_['T'] / m['resp'] - 1):+.1f}%）、"
              f"{b_['P']:.1f} W（{100 * (b_['P'] / m['power_W'] - 1):+.1f}%）")


def run_states(tag, tau):
    """量測區間（第 60 個請求做完到最後一個）每筆 GPU 讀數的狀態：忙、鎖著閒置（閒置 1 秒以後到放開）、深度閒置、其他（過渡）
    tau = None：預設（深度閒置 = 閒置 5 秒以後）；inf：一直鎖；其他：鎖住 τ 秒再放開（深度閒置 = 放開 2 秒以後）"""
    import datetime
    import re
    rows = load(tag)
    log = open(os.path.join(OUT, f"{tag}_runlog.txt"), encoding="utf-8").read()
    t0 = datetime.datetime.fromisoformat(re.search(r"實驗開始時間（UTC）：(\S+)", log).group(1)).timestamp()
    a = np.array([float(r["t_send"]) for r in rows]) + t0
    c = np.array([float(r["t_done"]) for r in rows]) + t0
    g = []
    for p in csv.reader(open(os.path.join(OUT, f"{tag}_gpu.csv"), encoding="utf-8")):
        try:
            g.append((datetime.datetime.strptime(p[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(p[5])))
        except (ValueError, IndexError):
            continue
    ts, pw = np.array([x[0] for x in g]), np.array([x[1] for x in g])
    win = (ts >= c[59]) & (ts <= c[-1])
    st = np.full(len(ts), "other", dtype=object)
    be = np.maximum.accumulate(c)
    start = np.maximum(a, np.concatenate([[-np.inf], be[:-1]]))
    for i in range(len(a)):
        st[(ts >= start[i]) & (ts <= c[i])] = "busy"
    for i in range(len(a) - 1):
        s, e = be[i], a[i + 1]
        if e <= s:
            continue
        if tau is None:
            st[(ts >= s + 5.0) & (ts < e)] = "deep"
        elif np.isinf(tau):
            st[(ts >= s + 1.0) & (ts < e)] = "locked"
        else:
            st[(ts >= s + 1.0) & (ts < min(e, s + tau))] = "locked"
            st[(ts >= s + tau + 2.0) & (ts < e)] = "deep"
    out = {n: (float((win & (st == n)).sum() / win.sum()), float(pw[win & (st == n)].mean()) if (win & (st == n)).any() else np.nan)
           for n in ("busy", "locked", "deep", "other")}
    return out, float(pw[win].mean())


def session_offset_check():
    """4. 早上（兩端）和晚上（三個中間點）的功耗差：晚上鎖著閒置時比早上耗電。把晚上鎖著閒置、深度閒置的功耗換成早上的值，
    再和兩端實測的連線比。深度閒置的時間範圍早晚不同（晚上包含放開後 2–5 秒的下降段），所以這個換算偏向高估換算量"""
    res = json.load(open(os.path.join(OUT, "power_results.json"), encoding="utf-8"))
    md, _ = run_states("power_default_rho20", None)
    ml, _ = run_states("power_locked_rho20", np.inf)
    ref_lock, ref_deep = ml["locked"][1], md["deep"][1]
    d, l = res["default"], res["locked"]
    slope = (l["resp"] - d["resp"]) / (l["power_W"] - d["power_W"])
    print(f"\n== 4. 早晚的功耗差（早上：鎖著閒置 {ref_lock:.1f} W、深度閒置 {ref_deep:.1f} W；忙的時候 預設 {md['busy'][1]:.1f}、鎖時脈 {ml['busy'][1]:.1f} W）")
    saved = {}
    for tau in TAUS:
        v, allp = run_states(f"kw_tau{tau}", float(tau))
        adj = allp + v["locked"][0] * (ref_lock - v["locked"][1]) + v["deep"][0] * (ref_deep - v["deep"][1])
        m = res[f"kw_tau{tau}"]
        raw = m["resp"] - (d["resp"] + slope * (m["power_W"] - d["power_W"]))
        fix = m["resp"] - (d["resp"] + slope * (adj - d["power_W"]))
        print(f"  τ = {tau:>2}：鎖著閒置 {100 * v['locked'][0]:.0f}% 的時間 {v['locked'][1]:.1f} W、深度閒置 {100 * v['deep'][0]:.0f}% {v['deep'][1]:.1f} W｜"
              f"平均 {allp:.2f} W → 換成早上的值 {adj:.2f} W｜比兩端連線慢：沒換算 {raw:+.3f} 秒、換算後 {fix:+.3f} 秒")
        saved[str(tau)] = {"power_W": allp, "power_adjusted_W": adj, "gap_raw_s": raw, "gap_adjusted_s": fix,
                           "locked_idle": v["locked"], "deep_idle": v["deep"], "morning_locked_idle_W": ref_lock, "morning_deep_idle_W": ref_deep}
    json.dump(saved, open(os.path.join(OUT, "keepwarm_session_offset.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("  → out/keepwarm_session_offset.json")


if __name__ == "__main__":
    main()