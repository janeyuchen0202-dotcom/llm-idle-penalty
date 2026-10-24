"""朋友電腦排隊測試的分析（2026-10-19）：先預測、記下時間，之後才打開排隊的資料

python -I analyze_queue.py predict <結果zip> <預測輸出.json>
    只讀 calibration.csv 和 run_info.json（不讀 queue_*.csv），用論文 predict_plan.py 的方法：
    - 一般服務時間 b：校正裡閒置 0 秒的請求
    - 閒置後多出的時間 δ：校正裡依實際閒置時間分四段（<1、1–2、2–5、≥5 秒），δ = 該段平均 − 閒置 0 秒的平均
    - 到達時間用和測試包相同的種子、λ、筆數重建（random.Random(seed).expovariate(λ)）
    - 模擬 2,000 次（每次重新抽樣校正資料），丟掉前 1/3 的請求，得到平均等待的預測和 95% 範圍
    - 對照用的天真 P-K：只用「單獨送一個請求」的服務時間（校正裡閒置 ≥ 5 秒）
python -I analyze_queue.py predict2 <結果zip> <預測_方法2_時間.json>
    方法 2（10/20 加）：每組的 b 改用這一組前 1/3 請求裡排隊中的請求（只讀前 1/3），其他相同
python -I analyze_queue.py predict3 <結果zip> <預測_方法3_時間.json>
    方法 3（10/21 加，排隊測試/設計說明.md「方法 3 的事前登記」）：b 照實際負載 ρ̂ 在冷的 b 和熱的 b 之間內插，δ 照每種閒置秒數內插；
    只讀 calibration.csv、sustained.csv、run_info.json。舊版測試包沒有 sustained.csv：只套 δ 內插，檔名用「事後預測_方法3δ_…」
python -I analyze_queue.py predict3t <結果zip> <預測_方法3t_時間.json>
    方法 3′（10/21 加，設計說明.md「方法 3′ 的事前登記」）：服務時間跟著溫度走，溫度用一階熱模型照忙碌的情形隨時間變；
    熱模型和「溫度 → 服務時間」都只用校正和連續請求那兩段擬合，不讀 queue_*.csv
python -I analyze_queue.py predict3s <結果zip> <預測_方法3s_時間.json>
    方法 3′（10/21 改成這個；溫度那版 predict3t 在開發資料上沒有比較好，不登記）：b 照這組的忙碌比例，在實際量到的
    冷的 b、50%、75%、100% 四個點之間內插，加上服務時間的前後相關（AR(1)）；只讀 calibration、duty_steps、sustained、run_info
python -I analyze_queue.py compare <結果zip> <預測輸出.json>
    讀排隊的資料，和預測比較（排序照送出時間；開始服務＝max(送出, 前一個完成)；和 analyze_plan.py 相同）
    10/21 加：每組的誤差（預測 − 實測，佔實測的 %）和各組誤差絕對值的平均
"""
import csv
import datetime
import io
import json
import random
import sys
import zipfile

import numpy as np

R = 2000
EDGES = [1.0, 2.0, 5.0]


def predict(zpath, out):
    z = zipfile.ZipFile(zpath)
    info = json.loads(z.read("run_info.json"))
    cal = list(csv.DictReader(io.StringIO(z.read("calibration.csv").decode("utf-8"))))
    cal = [r for r in cal if not r["error"]]
    S0 = np.array([float(r["service"]) for r in cal if float(r["gap"]) == 0])
    by_bin = [[] for _ in range(4)]
    for r in cal:
        if float(r["gap"]) > 0:
            by_bin[int(np.digitize(float(r["idle_before"]), EDGES))].append(float(r["service"]))
    by_bin = [np.array(x) if x else np.array([S0.mean()]) for x in by_bin]
    iso = np.array([float(r["service"]) for r in cal if float(r["gap"]) >= 5])
    rng = np.random.default_rng(29)
    res = {"zip": zpath.replace("\\", "/").rsplit("/", 1)[-1], "made_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "b": round(float(S0.mean()), 4), "b_cv": round(float(S0.std() / S0.mean()), 3),
           "delta_by_bin": {lab: round(float(by_bin[i].mean() - S0.mean()), 3) for i, lab in enumerate(["<1s", "1-2s", "2-5s", ">=5s"])},
           "n_by_bin": [len(x) for x in by_bin], "iso": round(float(iso.mean()), 4), "runs": []}
    for run in info["runs"]:
        lam, n, seed = run["lambda"], run["n"], run["seed"]
        r_, t, arr = random.Random(seed), 0.0, []
        for _ in range(n):
            t += r_.expovariate(lam)
            arr.append(t)
        arr = np.array(arr)
        k = n // 3
        pool = S0[rng.integers(0, len(S0), (R, len(S0)))]
        base = pool[np.arange(R)[:, None], rng.integers(0, len(S0), (R, n))]
        pd_ = S0[rng.integers(0, len(S0), (R, len(S0)))].mean(1)
        dv = np.stack([by_bin[i][rng.integers(0, len(by_bin[i]), (R, len(by_bin[i])))].mean(1) for i in range(4)], 1) - pd_[:, None]
        c_prev, acc = np.full(R, -np.inf), np.zeros(R)
        for i in range(n):
            start = np.maximum(arr[i], c_prev)
            w = start - arr[i]
            pen = np.where(w == 0, dv[np.arange(R), np.digitize(arr[i] - c_prev, EDGES)], 0.0)
            c_prev = start + base[:, i] + pen
            if i >= k:
                acc += w
        s = acc / (n - k)
        lo, hi = np.percentile(s, [2.5, 97.5])
        es, es2 = iso.mean(), (iso ** 2).mean()
        naive = lam * es2 / (2 * (1 - lam * es)) if lam * es < 1 else float("inf")
        res["runs"].append({"tag": run["tag"], "lambda": lam, "n": n, "seed": seed, "rho": round(lam * S0.mean(), 3),
                            "pred": round(float(s.mean()), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3),
                            "naive_pk": round(naive, 3)})
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"預測（{res['made_at']}，只用校正資料）：b = {res['b']} 秒（CV {res['b_cv']}），δ = {res['delta_by_bin']}，各段筆數 {res['n_by_bin']}")
    for r in res["runs"]:
        print(f"  {r['tag']}：ρ = {r['rho']}，平均等待預測 {r['pred']} 秒 [{r['lo']}, {r['hi']}]｜天真 P-K {r['naive_pk']}")



def served_prev(a, c):
    """10/22 加：每個請求「前一個被處理的請求」做完的時間。一次只處理一個請求，所以照做完的順序就是處理的順序；
    兩個請求幾乎同時送出時，伺服器可能先處理晚送的那個，照送出順序重建會算錯（資料稽核_20261022.md）"""
    prev = np.empty(len(c))
    cp = -np.inf
    for i in np.argsort(c, kind="stable"):
        prev[i] = cp
        cp = c[i]
    return prev


def predict2(zpath, out):
    """方法 2（10/20 加，排隊測試/設計說明.md 最後一節）：每組的 b 改用這一組前 1/3 請求裡排隊中的請求；其他和 predict 相同
    只讀每組前 1/3 的請求（照排定的送出時間），後 2/3 不讀"""
    z = zipfile.ZipFile(zpath)
    info = json.loads(z.read("run_info.json"))
    cal = [r for r in csv.DictReader(io.StringIO(z.read("calibration.csv").decode("utf-8"))) if not r["error"]]
    S0 = np.array([float(r["service"]) for r in cal if float(r["gap"]) == 0])
    by_bin = [[] for _ in range(4)]
    for r in cal:
        if float(r["gap"]) > 0:
            by_bin[int(np.digitize(float(r["idle_before"]), EDGES))].append(float(r["service"]))
    by_bin = [np.array(x) if x else np.array([S0.mean()]) for x in by_bin]
    iso = np.array([float(r["service"]) for r in cal if float(r["gap"]) >= 5])
    rng = np.random.default_rng(29)
    res = {"zip": zpath.replace("\\", "/").rsplit("/", 1)[-1], "made_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "method": "2：b 用每組前 1/3 的排隊請求", "b_cal": round(float(S0.mean()), 4),
           "delta_by_bin": {lab: round(float(by_bin[i].mean() - S0.mean()), 3) for i, lab in enumerate(["<1s", "1-2s", "2-5s", ">=5s"])},
           "runs": []}
    for run in info["runs"]:
        lam, n, seed = run["lambda"], run["n"], run["seed"]
        k = n // 3
        rows = list(csv.DictReader(io.StringIO(z.read(f"queue_{run['tag']}.csv").decode("utf-8"))))
        rows = sorted(rows, key=lambda r: float(r["t_scheduled"]))[:k]          # 只留前 1/3
        rows = sorted([r for r in rows if r["status"] == "ok"], key=lambda r: float(r["t_send"]))
        a = np.array([float(r["t_send"]) for r in rows])
        c = np.array([float(r["t_done"]) for r in rows])
        prev = served_prev(a, c)                              # 10/22 改：照做完的順序（原本照送出順序）
        own = (c - np.maximum(a, prev))[a < prev]
        if len(own) < 10:
            res["runs"].append({"tag": run["tag"], "skipped": f"前 1/3 只有 {len(own)} 個排隊請求（少於 10 個），不做方法 2"})
            continue
        r_, t, arr = random.Random(seed), 0.0, []
        for _ in range(n):
            t += r_.expovariate(lam)
            arr.append(t)
        arr = np.array(arr)
        pool = own[rng.integers(0, len(own), (R, len(own)))]
        base = pool[np.arange(R)[:, None], rng.integers(0, len(own), (R, n))]
        pd_ = S0[rng.integers(0, len(S0), (R, len(S0)))].mean(1)
        dv = np.stack([by_bin[i][rng.integers(0, len(by_bin[i]), (R, len(by_bin[i])))].mean(1) for i in range(4)], 1) - pd_[:, None]
        c_prev, acc = np.full(R, -np.inf), np.zeros(R)
        for i in range(n):
            start = np.maximum(arr[i], c_prev)
            w = start - arr[i]
            pen = np.where(w == 0, dv[np.arange(R), np.digitize(arr[i] - c_prev, EDGES)], 0.0)
            c_prev = start + base[:, i] + pen
            if i >= k:
                acc += w
        s = acc / (n - k)
        lo, hi = np.percentile(s, [2.5, 97.5])
        es, es2 = iso.mean(), (iso ** 2).mean()
        naive = lam * es2 / (2 * (1 - lam * es)) if lam * es < 1 else float("inf")
        res["runs"].append({"tag": run["tag"], "lambda": lam, "n": n, "seed": seed, "rho": round(lam * S0.mean(), 3),
                            "b_first_third": round(float(own.mean()), 4), "n_first_third_queued": int(len(own)),
                            "pred": round(float(s.mean()), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3),
                            "naive_pk": round(naive, 3)})
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"方法 2 預測（{res['made_at']}，只用校正資料和每組前 1/3）：校正 b = {res['b_cal']} 秒，δ = {res['delta_by_bin']}")
    for r in res["runs"]:
        if "skipped" in r:
            print(f"  {r['tag']}：{r['skipped']}")
        else:
            print(f"  {r['tag']}：前 1/3 排隊請求 {r['n_first_third_queued']} 個、b = {r['b_first_third']} 秒｜"
                  f"平均等待預測 {r['pred']} 秒 [{r['lo']}, {r['hi']}]｜天真 P-K {r['naive_pk']}")


def arrivals_of(run):
    r_, t, arr = random.Random(run["seed"]), 0.0, []
    for _ in range(run["n"]):
        t += r_.expovariate(run["lambda"])
        arr.append(t)
    return np.array(arr)


def predict3(zpath, out):
    """方法 3（10/21 事前登記）：模型（公式 (3)）不變，只改校正
    - 冷的 b：校正裡閒置 0 秒的平均；熱的 b：連續請求 8 分鐘的後一半的平均
    - 每組 b(ρ̂) = 冷的 b + ρ̂ ×（熱的 b − 冷的 b），ρ̂ = 冷的 b × 量測區間（後 2/3）的實際到達率（照種子重建到達時間）
    - 每個請求的服務時間 = 閒置 0 秒的服務時間抽樣 × b(ρ̂) ÷ 冷的 b
    - δ：每種閒置秒數的平均 − 冷的 b，橫軸是該秒數的實際平均閒置時間，(0, 0) 起，中間線性內插，超過最長的用最長的
    - 模擬 2,000 次，每次重新抽樣冷的 b、熱的 b、各秒數的服務時間；丟掉前 1/3"""
    z = zipfile.ZipFile(zpath)
    info = json.loads(z.read("run_info.json"))
    cal = [r for r in csv.DictReader(io.StringIO(z.read("calibration.csv").decode("utf-8"))) if not r["error"]]
    S0 = np.array([float(r["service"]) for r in cal if float(r["gap"]) == 0])
    gaps = sorted({float(r["gap"]) for r in cal if float(r["gap"]) > 0})
    per = {g: np.array([float(r["service"]) for r in cal if float(r["gap"]) == g]) for g in gaps}
    xs = np.array([0.0] + [float(np.mean([float(r["idle_before"]) for r in cal if float(r["gap"]) == g])) for g in gaps])
    iso = np.array([float(r["service"]) for r in cal if float(r["gap"]) >= 5])
    hot, note = None, ""
    if "sustained.csv" in z.namelist():
        sus = [r for r in csv.DictReader(io.StringIO(z.read("sustained.csv").decode("utf-8"))) if not r["error"]]
        T = float((info.get("sustained") or {}).get("seconds") or max(float(r["t_send"]) for r in sus))
        hot = np.array([float(r["service"]) for r in sus if float(r["t_send"]) >= T / 2])
        if len(hot) < 10 or not (info.get("sustained") or {}).get("complete"):
            hot, note = None, f"連續請求沒有跑完或後一半少於 10 個（{len(sus)} 個），不做方法 3"
    else:
        note = "舊版測試包沒有連續請求：只套 δ 內插（b 用冷的 b），事後檢查"
    if hot is None and "sustained.csv" in z.namelist():
        res = {"zip": zpath.replace("\\", "/").rsplit("/", 1)[-1], "made_at": datetime.datetime.now().isoformat(timespec="seconds"),
               "method": "3", "runs": [{"tag": r["tag"], "skipped": note} for r in info["runs"]]}
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("方法 3：" + note)
        return
    rng = np.random.default_rng(29)
    d_pts = {f"{x:.2f}": round(float(per[g].mean() - S0.mean()), 3) for x, g in zip(xs[1:], gaps)}
    res = {"zip": zpath.replace("\\", "/").rsplit("/", 1)[-1], "made_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "method": "3：b 隨負載改變＋δ 內插" if hot is not None else "3δ：只有 δ 內插（事後檢查）", "note": note,
           "b_cold": round(float(S0.mean()), 4), "b_hot": round(float(hot.mean()), 4) if hot is not None else None,
           "n_hot": int(len(hot)) if hot is not None else 0, "delta_points": d_pts, "runs": []}
    for run in info["runs"]:
        n = run["n"]
        k = n // 3
        arr = arrivals_of(run)
        lam_hat = (n - k - 1) / (arr[-1] - arr[k])
        pool = S0[rng.integers(0, len(S0), (R, len(S0)))]
        pd_ = pool.mean(1)                                                    # 每次模擬的冷的 b
        hot_m = hot[rng.integers(0, len(hot), (R, len(hot)))].mean(1) if hot is not None else pd_
        rho_hat = pd_ * lam_hat
        scale = (pd_ + rho_hat * (hot_m - pd_)) / pd_
        base = pool[np.arange(R)[:, None], rng.integers(0, len(S0), (R, n))] * scale[:, None]
        gm = np.stack([per[g][rng.integers(0, len(per[g]), (R, len(per[g])))].mean(1) for g in gaps], 1) - pd_[:, None]
        gm = np.concatenate([np.zeros((R, 1)), gm], 1)
        c_prev, acc = np.full(R, -np.inf), np.zeros(R)
        for i in range(n):
            start = np.maximum(arr[i], c_prev)
            w = start - arr[i]
            x = np.clip(np.where(np.isfinite(c_prev), arr[i] - c_prev, xs[-1]), 0.0, xs[-1])
            j = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
            f = (x - xs[j]) / (xs[j + 1] - xs[j])
            d = gm[np.arange(R), j] * (1 - f) + gm[np.arange(R), j + 1] * f
            c_prev = start + base[:, i] + np.where(w == 0, d, 0.0)
            if i >= k:
                acc += w
        s = acc / (n - k)
        lo, hi = np.percentile(s, [2.5, 97.5])
        es, es2 = iso.mean(), (iso ** 2).mean()
        lam = run["lambda"]
        naive = lam * es2 / (2 * (1 - lam * es)) if lam * es < 1 else float("inf")
        res["runs"].append({"tag": run["tag"], "lambda": lam, "n": n, "seed": run["seed"], "rho_hat": round(float(rho_hat.mean()), 3),
                            "b_run": round(float((pd_ * scale).mean()), 4), "pred": round(float(s.mean()), 3),
                            "lo": round(float(lo), 3), "hi": round(float(hi), 3), "naive_pk": round(naive, 3)})
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"方法 {res['method']}（{res['made_at']}，只用校正和連續請求的資料）：冷的 b = {res['b_cold']} 秒、熱的 b = {res['b_hot']}"
          f"（後一半 {res['n_hot']} 個）｜δ（實際閒置秒數：δ）= {d_pts}" + (f"｜{note}" if note else ""))
    for r in res["runs"]:
        print(f"  {r['tag']}：ρ̂ = {r['rho_hat']}、b(ρ̂) = {r['b_run']} 秒｜平均等待預測 {r['pred']} 秒 [{r['lo']}, {r['hi']}]｜天真 P-K {r['naive_pk']}")


# ---------------------------------------------------------------- 方法 3′（10/21）：溫度
def gpu_temp_log(z):
    """round_all_gpu_nvidia.csv（測試包的 Monitors）：時間（本機時間）、使用率、溫度"""
    ts, ut, tc = [], [], []
    for p in csv.reader(io.StringIO(z.read("round_all_gpu_nvidia.csv").decode("utf-8", "replace"))):
        try:
            t, u, c = datetime.datetime.strptime(p[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(p[5]), float(p[6])
        except (ValueError, IndexError):
            continue
        ts.append(t)
        ut.append(u)
        tc.append(c)
    o = np.argsort(ts)
    return np.array(ts)[o], np.array(ut)[o], np.array(tc)[o]


def busy_flags(t, intervals):
    """每個取樣時間是不是在某個請求的處理中（intervals：(開始, 結束) 的牆上時間，照開始排序、不重疊）"""
    s = np.array([a for a, b in intervals])
    e = np.array([b for a, b in intervals])
    j = np.searchsorted(s, t, side="right") - 1
    return np.where(j >= 0, t <= e[np.maximum(j, 0)], False).astype(float)


def thermal_step(T, dt, u, T_idle, T_full, tau):
    """一階熱模型走 dt 秒：dT/dt = (T_eq − T)/τ，T_eq = T_idle + u (T_full − T_idle)"""
    eq = T_idle + u * (T_full - T_idle)
    return eq + (T - eq) * np.exp(-dt / tau)


def fit_thermal(windows):
    """擬合 T_idle、T_full、τ：windows = [(取樣時間, 忙不忙, 量到的溫度), ...]，每段的起點溫度用量到的。
    給定 τ 時溫度對 (T_idle, T_full) 是線性的 → 最小平方；τ 在 5–1800 秒之間格點搜尋"""
    best = None
    for tau in np.geomspace(5, 1800, 120):
        rows, rhs = [], []
        for t, u, y in windows:
            basis = np.zeros((len(t), 3))                    # 起點溫度、T_idle、T_full 各自的貢獻
            basis[0] = [1, 0, 0]
            for n in range(1, len(t)):
                a = np.exp(-(t[n] - t[n - 1]) / tau)
                basis[n] = basis[n - 1] * a + (1 - a) * np.array([0, 1 - u[n - 1], u[n - 1]])
            rows.append(basis[:, 1:])
            rhs.append(y - basis[:, 0] * y[0])
        A, b = np.vstack(rows), np.concatenate(rhs)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        sse = float(np.sum((A @ coef - b) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(coef[0]), float(coef[1]), float(tau), len(b))
    sse, T_idle, T_full, tau, n = best
    return {"T_idle": T_idle, "T_full": T_full, "tau": tau, "rmse": (sse / n) ** 0.5, "n": n}


def isotonic(x, y):
    """單調不減的迴歸（pool adjacent violators）：溫度越高，服務時間只會一樣或變長"""
    o = np.argsort(x)
    x, y = np.asarray(x, float)[o], np.asarray(y, float)[o]
    blocks = []                                              # [平均, 權重, 起點索引, 終點索引]
    for i, v in enumerate(y):
        blocks.append([v, 1.0, i, i])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, s2, _ = blocks.pop(-2)
            m1, w1, _, e1 = blocks.pop()
            blocks.append([(m2 * w2 + m1 * w1) / (w2 + w1), w2 + w1, s2, e1])
    fit = np.empty(len(y))
    for m, w, s, e in blocks:
        fit[s:e + 1] = m
    ux = np.unique(x)
    return ux, np.array([fit[x == v].mean() for v in ux])


def mean_temp(ts, tc, a, b):
    sel = (ts >= a) & (ts <= b)
    if sel.any():
        return float(tc[sel].mean())
    return float(tc[np.argmin(np.abs(ts - (a + b) / 2))])


def predict3t(zpath, out):
    """方法 3′（10/21 事前登記）：
    - 熱模型：一階，T_eq = T_idle + u (T_full − T_idle)，u = 1 處理請求中、0 閒著；T_idle、T_full、τ 用校正和連續請求兩段的溫度紀錄擬合
    - 溫度 → 服務時間：校正裡閒置 0 秒的請求和連續請求的每個請求（都是一個接一個），配上處理那段時間的平均溫度，做單調不減的迴歸；
      範圍外用最邊的值
    - 模擬：照實際跑的順序（run_info.json 的 runs）一路模擬，第一組從校正結束時量到的溫度開始，下一組接著上一組模擬出的溫度；
      每個請求開始前先照閒置時間降溫，服務時間 = 閒置 0 秒的服務時間抽樣 × b(開始時的溫度) ÷ 冷的 b，再加 δ（和方法 3 一樣內插），
      處理中照熱模型升溫
    - 其他和方法 3 相同：同一串到達時間、模擬 2,000 次（每次重新抽樣冷的 b、各秒數的 δ）、丟掉前 1/3；熱模型和溫度曲線固定不抽樣"""
    z = zipfile.ZipFile(zpath)
    info = json.loads(z.read("run_info.json"))
    cal = [r for r in csv.DictReader(io.StringIO(z.read("calibration.csv").decode("utf-8"))) if not r["error"]]
    res = {"zip": zpath.replace("\\", "/").rsplit("/", 1)[-1], "made_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "method": "3′：溫度（熱模型＋溫度→服務時間）＋δ 內插", "runs": []}
    sus_ok = "sustained.csv" in z.namelist() and (info.get("sustained") or {}).get("complete") \
        and "round_all_gpu_nvidia.csv" in z.namelist()
    if not sus_ok:
        res["runs"] = [{"tag": r["tag"], "skipped": "沒有完整的連續請求或溫度紀錄，不做方法 3′"} for r in info["runs"]]
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("方法 3′：沒有完整的連續請求或溫度紀錄，不做")
        return
    sus = [r for r in csv.DictReader(io.StringIO(z.read("sustained.csv").decode("utf-8"))) if not r["error"]]
    ts, ut, tc = gpu_temp_log(z)
    S0 = np.array([float(r["service"]) for r in cal if float(r["gap"]) == 0])
    b_cold = float(S0.mean())
    gaps = sorted({float(r["gap"]) for r in cal if float(r["gap"]) > 0})
    per = {g: np.array([float(r["service"]) for r in cal if float(r["gap"]) == g]) for g in gaps}
    xs = np.array([0.0] + [float(np.mean([float(r["idle_before"]) for r in cal if float(r["gap"]) == g])) for g in gaps])
    iso = np.array([float(r["service"]) for r in cal if float(r["gap"]) >= 5])

    # 1. 熱模型：校正、連續請求兩段（各自的處理時間當成「忙」）
    windows, check = [], []
    for rows in (cal, sus):
        iv = sorted((float(r["wall_send"]), float(r["wall_send"]) + float(r["service"])) for r in rows)
        sel = (ts >= iv[0][0]) & (ts <= iv[-1][1])
        u = busy_flags(ts[sel], iv)
        windows.append((ts[sel], u, tc[sel]))
        check.append(float(ut[sel][u > 0].mean()) if (u > 0).any() else float("nan"))   # 對時檢查：處理中使用率應該很高
    th = fit_thermal(windows)
    # 2. 溫度 → 服務時間（一個接一個的請求）
    pairs = [(mean_temp(ts, tc, float(r["wall_send"]), float(r["wall_send"]) + float(r["service"])), float(r["service"]))
             for r in [r for r in cal if float(r["gap"]) == 0] + sus]
    Tg, bg = isotonic([p[0] for p in pairs], [p[1] for p in pairs])
    cal_end = max(float(r["wall_send"]) + float(r["service"]) for r in cal)
    T_start = mean_temp(ts, tc, cal_end - 1.0, cal_end + 1.0)
    res.update(thermal={k: round(v, 3) if isinstance(v, float) else v for k, v in th.items()},
               busy_util_check=[round(x, 1) for x in check], b_cold=round(b_cold, 4),
               b_of_T={f"{t:.1f}": round(float(np.interp(t, Tg, bg)), 4) for t in np.linspace(Tg[0], Tg[-1], 6)},
               T_range=[round(float(Tg[0]), 1), round(float(Tg[-1]), 1)], n_pairs=len(pairs), T_start=round(T_start, 1))

    # 3. 模擬：照實際順序一路模擬溫度
    rng = np.random.default_rng(29)
    T = np.full(R, T_start)
    for run in info["runs"]:
        n, k = run["n"], run["n"] // 3
        arr = arrivals_of(run)
        pool = S0[rng.integers(0, len(S0), (R, len(S0)))]
        pd_ = pool.mean(1)
        base = pool[np.arange(R)[:, None], rng.integers(0, len(S0), (R, n))] / b_cold
        gm = np.stack([per[g][rng.integers(0, len(per[g]), (R, len(per[g])))].mean(1) for g in gaps], 1) - pd_[:, None]
        gm = np.concatenate([np.zeros((R, 1)), gm], 1)
        c_prev, acc, t_now = np.full(R, -np.inf), np.zeros(R), np.zeros(R)
        temps, bvals = [], []
        for i in range(n):
            start = np.maximum(arr[i], c_prev)
            w = start - arr[i]
            idle = np.where(np.isfinite(c_prev), np.maximum(arr[i] - c_prev, 0.0), arr[i])
            T = thermal_step(T, idle, 0.0, th["T_idle"], th["T_full"], th["tau"])          # 閒著：降溫
            x = np.clip(np.where(np.isfinite(c_prev), arr[i] - c_prev, xs[-1]), 0.0, xs[-1])
            j = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
            f = (x - xs[j]) / (xs[j + 1] - xs[j])
            d = gm[np.arange(R), j] * (1 - f) + gm[np.arange(R), j + 1] * f
            bT = np.interp(T, Tg, bg)
            svc = base[:, i] * bT + np.where(w == 0, d, 0.0)
            T = thermal_step(T, svc, 1.0, th["T_idle"], th["T_full"], th["tau"])           # 處理中：升溫
            c_prev = start + svc
            if i >= k:
                acc += w
                temps.append(T.mean())
                bvals.append(bT.mean())
        s = acc / (n - k)
        lo, hi = np.percentile(s, [2.5, 97.5])
        es, es2 = iso.mean(), (iso ** 2).mean()
        lam = run["lambda"]
        naive = lam * es2 / (2 * (1 - lam * es)) if lam * es < 1 else float("inf")
        res["runs"].append({"tag": run["tag"], "lambda": lam, "n": n, "seed": run["seed"],
                            "T_sim_mean": round(float(np.mean(temps)), 1), "b_sim_mean": round(float(np.mean(bvals)), 4),
                            "pred": round(float(s.mean()), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3),
                            "naive_pk": round(naive, 3)})
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"方法 3′（{res['made_at']}，只用校正和連續請求的資料）：熱模型 T_idle {th['T_idle']:.1f}°C、T_full {th['T_full']:.1f}°C、"
          f"τ {th['tau']:.0f} 秒（誤差 {th['rmse']:.2f}°C，{th['n']} 筆）｜對時檢查（處理中使用率）{res['busy_util_check']}｜"
          f"溫度 → 服務時間：{res['b_of_T']}｜第一組開始時 {T_start:.1f}°C")
    for r in res["runs"]:
        print(f"  {r['tag']}：模擬的溫度 {r['T_sim_mean']}°C、b {r['b_sim_mean']} 秒｜平均等待預測 {r['pred']} 秒 [{r['lo']}, {r['hi']}]｜"
              f"天真 P-K {r['naive_pk']}")


def predict3s_v1(zpath, out):
    """方法 3′ 的第一版（10/21 21:45 登記，22:10 前被更新版取代，那時還沒有任何新資料；留著當紀錄）：模型（公式 (3)）不變，只改校正
    - b 對忙碌比例的曲線：點 = (校正期間的忙碌比例, 冷的 b)、(50% 那段實際的忙碌比例, 後一半的平均)、(75% 同上)、
      (連續請求後一半的忙碌比例, 熱的 b)；照忙碌比例排序後做單調不減的迴歸，中間線性內插，範圍外用最邊的值
    - 每組的忙碌比例：先用 ρ̂（冷的 b × 量測區間的實際到達率）模擬一次，取模擬出的忙碌比例（量測區間：服務時間總和 ÷ 經過的時間），
      再用它的 b 正式模擬（只更新一次）
    - 服務時間 = b(忙碌比例) ×（1 + e），e 是 AR(1)：連續請求後一半的「服務時間 ÷ 平均 − 1」的前後相關 φ 和標準差 σ；
      兩個請求的開始時間差 Δt 秒，相關用 φ^(Δt ÷ 熱的 b)
    - δ 和方法 3 一樣內插；模擬 2,000 次，每次重新抽樣各段的平均、各秒數的 δ；丟掉前 1/3"""
    z = zipfile.ZipFile(zpath)
    info = json.loads(z.read("run_info.json"))
    cal = [r for r in csv.DictReader(io.StringIO(z.read("calibration.csv").decode("utf-8"))) if not r["error"]]
    res = {"zip": zpath.replace("\\", "/").rsplit("/", 1)[-1], "made_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "method": "3′：b 照忙碌比例在量到的四個點之間內插＋服務時間前後相關＋δ 內插", "runs": []}
    ok = all(n in z.namelist() for n in ("duty_steps.csv", "sustained.csv")) and (info.get("sustained") or {}).get("complete") \
        and info.get("duty_steps") and all(s.get("complete") for s in info["duty_steps"])
    if not ok:
        res["runs"] = [{"tag": r["tag"], "skipped": "照比例忙或連續請求沒有跑完，不做方法 3′"} for r in info["runs"]]
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("方法 3′：照比例忙或連續請求沒有跑完，不做")
        return
    S0 = np.array([float(r["service"]) for r in cal if float(r["gap"]) == 0])
    b_cold = float(S0.mean())
    gaps = sorted({float(r["gap"]) for r in cal if float(r["gap"]) > 0})
    per = {g: np.array([float(r["service"]) for r in cal if float(r["gap"]) == g]) for g in gaps}
    xs = np.array([0.0] + [float(np.mean([float(r["idle_before"]) for r in cal if float(r["gap"]) == g])) for g in gaps])
    iso_s = np.array([float(r["service"]) for r in cal if float(r["gap"]) >= 5])
    # 各點：忙碌比例和後一半的服務時間
    c_start = min(float(r["wall_send"]) for r in cal)
    c_end = max(float(r["wall_send"]) + float(r["service"]) for r in cal)
    pts = [(sum(float(r["service"]) for r in cal) / (c_end - c_start), S0)]
    duty = [r for r in csv.DictReader(io.StringIO(z.read("duty_steps.csv").decode("utf-8"))) if not r["error"]]
    for st in info["duty_steps"]:
        half = np.array([float(r["service"]) for r in duty if float(r["duty"]) == st["duty"] and float(r["t_send"]) >= st["seconds"] / 2])
        pts.append((float(st["busy_fraction"]), half))
    sus = [r for r in csv.DictReader(io.StringIO(z.read("sustained.csv").decode("utf-8"))) if not r["error"]]
    T = float(info["sustained"]["seconds"])
    hot_rows = [r for r in sus if float(r["t_send"]) >= T / 2]
    hot = np.array([float(r["service"]) for r in hot_rows])
    span = (float(hot_rows[-1]["t_send"]) + float(hot_rows[-1]["service"])) - float(hot_rows[0]["t_send"])
    pts.append((float(hot.sum() / span), hot))
    b_hot = float(hot.mean())
    xh = hot / b_hot - 1
    phi = float(np.clip(np.corrcoef(xh[:-1], xh[1:])[0, 1], 0.0, 0.999)) if len(xh) > 3 else 0.0
    sig = float(xh.std())
    order = np.argsort([p[0] for p in pts])
    U = np.array([pts[i][0] for i in order])
    samples = [pts[i][1] for i in order]

    def curve(means):
        """單調不減的 b(u)（四個點的 PAV）"""
        m = list(means)
        blocks = [[v, 1.0, i, i] for i, v in enumerate(m[:1])]
        for i, v in enumerate(m[1:], 1):
            blocks.append([v, 1.0, i, i])
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                m2, w2, s2, _ = blocks.pop(-2)
                m1, w1, _, e1 = blocks.pop()
                blocks.append([(m2 * w2 + m1 * w1) / (w2 + w1), w2 + w1, s2, e1])
        f = np.empty(len(m))
        for mm, ww, s2, e2 in blocks:
            f[s2:e2 + 1] = mm
        return f

    rng = np.random.default_rng(29)
    res.update(points=[{"busy": round(float(u), 3), "b": round(float(s.mean()), 4), "n": int(len(s))} for u, s in zip(U, samples)],
               b_cold=round(b_cold, 4), b_hot=round(b_hot, 4), ar1_phi=round(phi, 3), ar1_sigma=round(sig, 4))

    def simulate(run, u_target, rng):
        n, k = run["n"], run["n"] // 3
        arr = arrivals_of(run)
        means = np.stack([s[rng.integers(0, len(s), (R, len(s)))].mean(1) for s in samples], 1)      # 每次模擬重新抽樣各點
        bu = np.array([np.interp(u_target if np.ndim(u_target) == 0 else u_target[i], U, curve(means[i])) for i in range(R)])
        pd_ = S0[rng.integers(0, len(S0), (R, len(S0)))].mean(1)
        gm = np.stack([per[g][rng.integers(0, len(per[g]), (R, len(per[g])))].mean(1) for g in gaps], 1) - pd_[:, None]
        gm = np.concatenate([np.zeros((R, 1)), gm], 1)
        e = rng.normal(0, sig, R)
        c_prev, s_prev, acc = np.full(R, -np.inf), None, np.zeros(R)
        busy, t0 = np.zeros(R), None
        for i in range(n):
            start = np.maximum(arr[i], c_prev)
            w = start - arr[i]
            if s_prev is not None:
                ph = phi ** (np.maximum(start - s_prev, 0.0) / b_hot)
                e = ph * e + np.sqrt(1 - ph ** 2) * sig * rng.normal(0, 1, R)
            x = np.clip(np.where(np.isfinite(c_prev), arr[i] - c_prev, xs[-1]), 0.0, xs[-1])
            j = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
            f = (x - xs[j]) / (xs[j + 1] - xs[j])
            d = gm[np.arange(R), j] * (1 - f) + gm[np.arange(R), j + 1] * f
            svc = bu * (1 + e) + np.where(w == 0, d, 0.0)
            if i >= k:
                if t0 is None:
                    t0 = start.copy()
                acc += w
                busy += svc
            s_prev = start
            c_prev = start + svc
        return acc / (n - k), busy / (c_prev - t0), bu

    for run in info["runs"]:
        n, k = run["n"], run["n"] // 3
        arr = arrivals_of(run)
        rho_hat = b_cold * (n - k - 1) / (arr[-1] - arr[k])
        _, u_sim, _ = simulate(run, rho_hat, rng)                      # 第一次：用 ρ̂
        s, u2, bu = simulate(run, u_sim, rng)                          # 第二次：用模擬出的忙碌比例
        lo, hi = np.percentile(s, [2.5, 97.5])
        es, es2 = iso_s.mean(), (iso_s ** 2).mean()
        lam = run["lambda"]
        naive = lam * es2 / (2 * (1 - lam * es)) if lam * es < 1 else float("inf")
        res["runs"].append({"tag": run["tag"], "lambda": lam, "n": n, "seed": run["seed"], "rho_hat": round(float(rho_hat), 3),
                            "busy_used": round(float(u_sim.mean()), 3), "b_used": round(float(bu.mean()), 4),
                            "pred": round(float(s.mean()), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3),
                            "naive_pk": round(naive, 3)})
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"方法 3′（{res['made_at']}，只用校正、照比例忙、連續請求的資料）：b 對忙碌比例 " +
          "、".join(f"{p['busy']:.2f} → {p['b']:.4f}（{p['n']} 個）" for p in res["points"]) +
          f"｜前後相關 φ = {phi:.3f}、σ = {sig:.4f}")
    for r in res["runs"]:
        print(f"  {r['tag']}：ρ̂ {r['rho_hat']} → 模擬的忙碌比例 {r['busy_used']}、b {r['b_used']} 秒｜平均等待預測 {r['pred']} 秒 "
              f"[{r['lo']}, {r['hi']}]｜天真 P-K {r['naive_pk']}")


TAU_H = 12.0             # 「最近多忙」的時間常數（10/21 開發資料：第 6 台 7 秒、第 4 台 20 秒，取中間）


def h_mid(rows, tau=TAU_H):
    """一段依序的請求（處理中 = [wall_send, wall_send + service]），每個請求處理到一半時的 H（這段開始時 H = 0）；也回傳結束時的 H"""
    H, t_last, out = 0.0, None, []
    for r in rows:
        s, sv = float(r["wall_send"]), float(r["service"])
        if t_last is not None:
            H *= np.exp(-max(s - t_last, 0.0) / tau)
        out.append(1 + (H - 1) * np.exp(-(sv / 2) / tau))
        H = 1 + (H - 1) * np.exp(-sv / tau)
        t_last = s + sv
    return np.array(out), H


def pav(m):
    """單調不減的迴歸（pool adjacent violators），等權重"""
    blocks = []
    for i, v in enumerate(m):
        blocks.append([v, 1.0, i, i])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, s2, _ = blocks.pop(-2)
            m1, w1, _, e1 = blocks.pop()
            blocks.append([(m2 * w2 + m1 * w1) / (w2 + w1), w2 + w1, s2, e1])
    f = np.empty(len(m))
    for mm, ww, s2, e2 in blocks:
        f[s2:e2 + 1] = mm
    return f


def predict3s(zpath, out):
    """方法 3′ 更新版（10/21 22:10 事前登記；取代 21:45 的第一版 predict3s_v1，那時還沒有任何新資料）：
    - 服務時間跟著「最近 12 秒有多忙」H 隨時變：H 是忙不忙的指數移動平均（處理中往 1、閒著往 0，時間常數 12 秒），用處理到一半時的 H
    - 曲線 b(H) 的四個點 = (那些請求處理到一半時 H 的平均, 它們的服務時間)：校正裡閒置 0 秒的請求、50% 和 75% 兩段的後一半、
      連續請求的後一半；每段各自從頭算 H；照 H 排序做單調不減的迴歸，中間線性內插，範圍外用最邊的值
    - 服務時間 = b(H) ×（1 + e），e 是 AR(1)（連續請求後一半估 φ、σ；相關用 φ^(Δt ÷ 熱的 b)）；δ 照每種閒置秒數內插
    - 照實際跑的順序一路模擬 H，第一組從校正結束時的 H 開始；同一串到達時間；模擬 2,000 次（每次重新抽樣各點的平均、各秒數的 δ）；
      丟掉前 1/3。只讀 calibration、duty_steps、sustained、run_info"""
    z = zipfile.ZipFile(zpath)
    info = json.loads(z.read("run_info.json"))
    cal = [r for r in csv.DictReader(io.StringIO(z.read("calibration.csv").decode("utf-8"))) if not r["error"]]
    res = {"zip": zpath.replace("\\", "/").rsplit("/", 1)[-1], "made_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "method": "3′（更新版）：b 跟著最近 12 秒多忙、曲線用量到的四個點＋前後相關＋δ 內插", "tau_h": TAU_H, "runs": []}
    ok = all(n in z.namelist() for n in ("duty_steps.csv", "sustained.csv")) and (info.get("sustained") or {}).get("complete") \
        and info.get("duty_steps") and all(s.get("complete") for s in info["duty_steps"])
    if not ok:
        res["runs"] = [{"tag": r["tag"], "skipped": "照比例忙或連續請求沒有跑完，不做方法 3′"} for r in info["runs"]]
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("方法 3′：照比例忙或連續請求沒有跑完，不做")
        return
    S0 = np.array([float(r["service"]) for r in cal if float(r["gap"]) == 0])
    gaps = sorted({float(r["gap"]) for r in cal if float(r["gap"]) > 0})
    per = {g: np.array([float(r["service"]) for r in cal if float(r["gap"]) == g]) for g in gaps}
    xs = np.array([0.0] + [float(np.mean([float(r["idle_before"]) for r in cal if float(r["gap"]) == g])) for g in gaps])
    iso_s = np.array([float(r["service"]) for r in cal if float(r["gap"]) >= 5])
    cal_sorted = sorted(cal, key=lambda r: float(r["wall_send"]))
    hc, H_cal_end = h_mid(cal_sorted)
    zero = np.array([float(r["gap"]) == 0 for r in cal_sorted])
    pts = [(float(hc[zero].mean()), np.array([float(r["service"]) for r in cal_sorted if float(r["gap"]) == 0]))]
    duty = [r for r in csv.DictReader(io.StringIO(z.read("duty_steps.csv").decode("utf-8"))) if not r["error"]]
    for st in info["duty_steps"]:
        rows = sorted([r for r in duty if float(r["duty"]) == st["duty"]], key=lambda r: float(r["wall_send"]))
        hh, _ = h_mid(rows)
        sec = np.array([float(r["t_send"]) >= st["seconds"] / 2 for r in rows])
        pts.append((float(hh[sec].mean()), np.array([float(r["service"]) for r, s in zip(rows, sec) if s])))
    sus = sorted([r for r in csv.DictReader(io.StringIO(z.read("sustained.csv").decode("utf-8"))) if not r["error"]],
                 key=lambda r: float(r["wall_send"]))
    hh, _ = h_mid(sus)
    T = float(info["sustained"]["seconds"])
    sec = np.array([float(r["t_send"]) >= T / 2 for r in sus])
    hot = np.array([float(r["service"]) for r, s in zip(sus, sec) if s])
    pts.append((float(hh[sec].mean()), hot))
    b_hot = float(hot.mean())
    xh = hot / b_hot - 1
    phi = float(np.clip(np.corrcoef(xh[:-1], xh[1:])[0, 1], 0.0, 0.999)) if len(xh) > 3 else 0.0
    sig = float(xh.std())
    order = np.argsort([p[0] for p in pts])
    U = np.array([pts[i][0] for i in order])
    samples = [pts[i][1] for i in order]
    res.update(points=[{"H": round(float(u), 3), "b": round(float(s.mean()), 4), "n": int(len(s))} for u, s in zip(U, samples)],
               b_cold=round(float(S0.mean()), 4), b_hot=round(b_hot, 4), ar1_phi=round(phi, 3), ar1_sigma=round(sig, 4),
               H_cal_end=round(float(H_cal_end), 3))

    rng = np.random.default_rng(29)
    Hs = np.full(R, H_cal_end)
    for run in info["runs"]:
        n, k = run["n"], run["n"] // 3
        arr = arrivals_of(run)
        means = np.stack([s[rng.integers(0, len(s), (R, len(s)))].mean(1) for s in samples], 1)
        curves = np.array([pav(means[i]) for i in range(R)])
        pd_ = S0[rng.integers(0, len(S0), (R, len(S0)))].mean(1)
        gm = np.stack([per[g][rng.integers(0, len(per[g]), (R, len(per[g])))].mean(1) for g in gaps], 1) - pd_[:, None]
        gm = np.concatenate([np.zeros((R, 1)), gm], 1)

        def b_of(h):
            j = np.clip(np.searchsorted(U, h, side="right") - 1, 0, len(U) - 2)
            f = np.clip((h - U[j]) / (U[j + 1] - U[j]), 0.0, 1.0)
            return curves[np.arange(R), j] * (1 - f) + curves[np.arange(R), j + 1] * f

        e = rng.normal(0, sig, R)
        c_prev, s_prev, acc, hsum, bsum = np.full(R, -np.inf), None, np.zeros(R), 0.0, 0.0
        for i in range(n):
            start = np.maximum(arr[i], c_prev)
            w = start - arr[i]
            idle = np.where(np.isfinite(c_prev), np.maximum(arr[i] - c_prev, 0.0), arr[i])
            Hs = Hs * np.exp(-idle / TAU_H)
            if s_prev is not None:
                ph = phi ** (np.maximum(start - s_prev, 0.0) / b_hot)
                e = ph * e + np.sqrt(1 - ph ** 2) * sig * rng.normal(0, 1, R)
            x = np.clip(np.where(np.isfinite(c_prev), arr[i] - c_prev, xs[-1]), 0.0, xs[-1])
            j = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
            fr = (x - xs[j]) / (xs[j + 1] - xs[j])
            d = gm[np.arange(R), j] * (1 - fr) + gm[np.arange(R), j + 1] * fr
            Hm = 1 + (Hs - 1) * np.exp(-(b_of(Hs) / 2) / TAU_H)
            bh = b_of(Hm)
            svc = bh * (1 + e) + np.where(w == 0, d, 0.0)
            Hs = 1 + (Hs - 1) * np.exp(-svc / TAU_H)
            s_prev = start
            c_prev = start + svc
            if i >= k:
                acc += w
                hsum += float(Hm.mean())
                bsum += float(bh.mean())
        s = acc / (n - k)
        lo, hi = np.percentile(s, [2.5, 97.5])
        es, es2 = iso_s.mean(), (iso_s ** 2).mean()
        lam = run["lambda"]
        naive = lam * es2 / (2 * (1 - lam * es)) if lam * es < 1 else float("inf")
        res["runs"].append({"tag": run["tag"], "lambda": lam, "n": n, "seed": run["seed"],
                            "H_mean": round(hsum / (n - k), 3), "b_mean": round(bsum / (n - k), 4),
                            "pred": round(float(s.mean()), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3),
                            "naive_pk": round(naive, 3)})
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"方法 3′ 更新版（{res['made_at']}，只用校正、照比例忙、連續請求的資料）：b(H) " +
          "、".join(f"H {p['H']:.2f} → {p['b']:.4f}（{p['n']} 個）" for p in res["points"]) +
          f"｜前後相關 φ = {phi:.3f}、σ = {sig:.4f}｜校正結束時 H = {H_cal_end:.2f}")
    for r in res["runs"]:
        print(f"  {r['tag']}：處理時平均 H {r['H_mean']}、b {r['b_mean']} 秒｜平均等待預測 {r['pred']} 秒 [{r['lo']}, {r['hi']}]｜"
              f"天真 P-K {r['naive_pk']}")


TAUS5 = np.array([5.0, 20.0, 60.0])   # 方法 5 的三條線（使用者：像股市的週、月、季線；10/22 遺傳演算法分兩半檢查也沒有找到更好的）
CROSS_T = 30.0                        # 黃金交叉之後幾秒內算「正在變慢」


def lines_feats(rows, H0=None):
    """方法 5：一段依序的請求，每個請求一列 (處理到一半時的 60 秒線, 是不是在黃金交叉後 30 秒內而且還在上面, 服務時間, 前面閒置幾秒)；
    黃金交叉 = 5 秒線由下往上穿過 20 秒線和 60 秒線。H0 = 這段開始時三條線的值（沒給就是 0）；第一個請求前面閒置多久不知道，記 nan。
    也回傳結束時三條線的值、最後一次黃金交叉在結束前幾秒（負的；沒有就是 -inf）"""
    H = np.zeros(3) if H0 is None else np.array(H0, float)
    t_last, last, out = None, -np.inf, []
    above = lambda h: h[0] > max(h[1], h[2])
    for r in rows:
        s, sv = float(r["wall_send"]), float(r["service"])
        idle = np.nan
        if t_last is not None:
            idle = max(s - t_last, 0.0)
            before = H.copy()
            H = H * np.exp(-idle / TAUS5)
            if not above(before) and above(H):
                last = s
        x = 1.0 if (s - last <= CROSS_T and above(H)) else 0.0
        out.append([1 + (H[2] - 1) * np.exp(-(sv / 2) / TAUS5[2]), x, sv, idle])
        b2 = H.copy()
        H = 1 + (H - 1) * np.exp(-sv / TAUS5)
        if not above(b2) and above(H):
            last = s + sv
        t_last = s + sv
    return np.array(out), H, (last - t_last) if t_last is not None else -np.inf


def predict5(zpath, out):
    """方法 5（10/22 晚事前登記；那時第 6、9 台的 v2b 還沒有人跑，沒有任何新資料）：黃金交叉
    - 三條線 = 忙不忙的指數移動平均，時間常數 5、20、60 秒（處理中往 1、閒著往 0）
    - 服務時間 = c0 + c60 ×（處理到一半時的 60 秒線）+ cx ×［5 秒線由下往上穿過 20、60 秒線之後 30 秒內、而且還在上面］，再 ×（1 + e）
    - 係數用最小平方法配：校正裡閒置 0 秒的請求（校正從三條線 = 0 開始算），加上照比例忙 50%、75% 和連續請求的全部請求
      （這三段是接著跑的，三條線一路接著算，開頭三條線都設成最後一組排隊的負載 ρ = λ × 冷的 b；每個請求先扣掉照它前面實際閒置秒數內插的 δ，
      第一個請求前面閒置多久不知道，不用）。模擬時每次重新抽樣這些請求、重配一次係數（參數的不確定性）
    - e：AR(1)，和方法 3′ 相同（連續請求後一半估 φ、σ；相關用 φ^(Δt ÷ 熱的 b)）；δ 照每種閒置秒數內插
    - 照實際跑的順序一路模擬三條線，第一組從校正結束時的值（和最後一次黃金交叉的時間）開始，組和組之間當作沒有間隔；
      同一串到達時間；模擬 2,000 次；丟掉前 1/3
    - 只讀 calibration、duty_steps、sustained、run_info"""
    z = zipfile.ZipFile(zpath)
    info = json.loads(z.read("run_info.json"))
    cal = [r for r in csv.DictReader(io.StringIO(z.read("calibration.csv").decode("utf-8"))) if not r["error"]]
    res = {"zip": zpath.replace("\\", "/").rsplit("/", 1)[-1], "made_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "method": "5：黃金交叉（5／20／60 秒三條線，交叉後 30 秒內）", "taus": TAUS5.tolist(), "cross_window_s": CROSS_T, "runs": []}
    ok = all(n in z.namelist() for n in ("duty_steps.csv", "sustained.csv")) and (info.get("sustained") or {}).get("complete") \
        and info.get("duty_steps") and all(s.get("complete") for s in info["duty_steps"])
    if not ok:
        res["runs"] = [{"tag": r["tag"], "skipped": "照比例忙或連續請求沒有跑完，不做方法 5"} for r in info["runs"]]
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("方法 5：照比例忙或連續請求沒有跑完，不做")
        return
    S0 = np.array([float(r["service"]) for r in cal if float(r["gap"]) == 0])
    gaps = sorted({float(r["gap"]) for r in cal if float(r["gap"]) > 0})
    per = {g: np.array([float(r["service"]) for r in cal if float(r["gap"]) == g]) for g in gaps}
    xs = np.array([0.0] + [float(np.mean([float(r["idle_before"]) for r in cal if float(r["gap"]) == g])) for g in gaps])
    ds = np.array([0.0] + [per[g].mean() - S0.mean() for g in gaps])
    d_of = lambda x: float(np.interp(min(max(x, 0.0), xs[-1]), xs, ds))
    iso_s = np.array([float(r["service"]) for r in cal if float(r["gap"]) >= 5])
    cal_sorted = sorted(cal, key=lambda r: float(r["wall_send"]))
    Fc, H_cal_end, last_rel = lines_feats(cal_sorted)
    zero = np.array([float(r["gap"]) == 0 for r in cal_sorted])
    duty = [r for r in csv.DictReader(io.StringIO(z.read("duty_steps.csv").decode("utf-8"))) if not r["error"]]
    sus = sorted([r for r in csv.DictReader(io.StringIO(z.read("sustained.csv").decode("utf-8"))) if not r["error"]],
                 key=lambda r: float(r["wall_send"]))
    busy = sorted(duty + sus, key=lambda r: float(r["wall_send"]))     # 50% → 75% → 連續請求，中間沒有停
    rho_last = min(float(info["runs"][-1]["lambda"]) * float(S0.mean()), 1.0)
    Fb, _, _ = lines_feats(busy, np.full(3, rho_last))
    known = ~np.isnan(Fb[:, 3])
    X = np.vstack([Fc[zero, :2], Fb[known, :2]])
    y = np.concatenate([Fc[zero, 2], Fb[known, 2] - np.array([d_of(x) for x in Fb[known, 3]])])
    T = float(info["sustained"]["seconds"])
    sec = np.array([float(r["t_send"]) >= T / 2 for r in sus])
    hot = np.array([float(r["service"]) for r, s in zip(sus, sec) if s])
    b_hot = float(hot.mean())
    xh = hot / b_hot - 1
    phi = float(np.clip(np.corrcoef(xh[:-1], xh[1:])[0, 1], 0.0, 0.999)) if len(xh) > 3 else 0.0
    sig = float(xh.std())
    A = np.column_stack([np.ones(len(X)), X[:, 0], X[:, 1]])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    rng = np.random.default_rng(29)
    boots = np.array([np.linalg.lstsq(A[idx], y[idx], rcond=None)[0] for idx in rng.integers(0, len(y), (R, len(y)))])
    res.update(coef=[round(float(c), 5) for c in coef], n_fit=int(len(y)), n_cross=int(X[:, 1].sum()), b_cold=round(float(S0.mean()), 4),
               b_hot=round(b_hot, 4), ar1_phi=round(phi, 3), ar1_sigma=round(sig, 4), rho_busy_start=round(rho_last, 3),
               lines_cal_end=[round(float(h), 3) for h in H_cal_end])
    Hs = np.tile(H_cal_end, (R, 1))
    last = np.full(R, last_rel)
    above = lambda h: h[:, 0] > np.maximum(h[:, 1], h[:, 2])
    clock = 0.0                                                          # 各組各自的時間軸（0 = 上一組最後一個做完）
    for run in info["runs"]:
        n, k = run["n"], run["n"] // 3
        arr = arrivals_of(run)
        last = last - clock
        pd_ = S0[rng.integers(0, len(S0), (R, len(S0)))].mean(1)
        gm = np.stack([per[g][rng.integers(0, len(per[g]), (R, len(per[g])))].mean(1) for g in gaps], 1) - pd_[:, None]
        gm = np.concatenate([np.zeros((R, 1)), gm], 1)
        e = rng.normal(0, sig, R)
        c_prev, s_prev, acc, xsum = np.full(R, -np.inf), None, np.zeros(R), 0.0
        for i in range(n):
            start = np.maximum(arr[i], c_prev)
            w = start - arr[i]
            idle = np.where(np.isfinite(c_prev), np.maximum(arr[i] - c_prev, 0.0), arr[i])
            before = above(Hs)
            Hs = Hs * np.exp(-idle[:, None] / TAUS5)
            now = above(Hs)
            last = np.where(~before & now, start, last)
            xx = ((start - last) <= CROSS_T) & now
            if s_prev is not None:
                ph = phi ** (np.maximum(start - s_prev, 0.0) / b_hot)
                e = ph * e + np.sqrt(1 - ph ** 2) * sig * rng.normal(0, 1, R)
            x = np.clip(np.where(np.isfinite(c_prev), arr[i] - c_prev, xs[-1]), 0.0, xs[-1])
            j = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
            fr = (x - xs[j]) / (xs[j + 1] - xs[j])
            d = gm[np.arange(R), j] * (1 - fr) + gm[np.arange(R), j + 1] * fr
            h60 = 1 + (Hs[:, 2] - 1) * np.exp(-(b_hot / 2) / TAUS5[2])
            base = boots[:, 0] + boots[:, 1] * h60 + boots[:, 2] * xx
            svc = base * (1 + e) + np.where(w == 0, d, 0.0)
            b2 = above(Hs)
            Hs = 1 + (Hs - 1) * np.exp(-svc[:, None] / TAUS5)
            last = np.where(~b2 & above(Hs), start + svc, last)
            s_prev = start
            c_prev = start + svc
            if i >= k:
                acc += w
                xsum += float(xx.mean())
        clock = c_prev.copy()
        s = acc / (n - k)
        lo, hi = np.percentile(s, [2.5, 97.5])
        es, es2 = iso_s.mean(), (iso_s ** 2).mean()
        lam = run["lambda"]
        naive = lam * es2 / (2 * (1 - lam * es)) if lam * es < 1 else float("inf")
        res["runs"].append({"tag": run["tag"], "lambda": lam, "n": n, "seed": run["seed"], "cross_share": round(xsum / (n - k), 3),
                            "pred": round(float(s.mean()), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3),
                            "naive_pk": round(naive, 3)})
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"方法 5 黃金交叉（{res['made_at']}，只用校正、照比例忙、連續請求的資料）：係數 {res['coef']}（{res['n_fit']} 個請求，"
          f"其中 {res['n_cross']} 個在黃金交叉後 30 秒內）｜前後相關 φ = {phi:.3f}、σ = {sig:.4f}")
    for r in res["runs"]:
        print(f"  {r['tag']}：處理時在交叉後 30 秒內的比例 {r['cross_share']}｜平均等待預測 {r['pred']} 秒 [{r['lo']}, {r['hi']}]｜天真 P-K {r['naive_pk']}")


def compare(zpath, pred_path):
    z = zipfile.ZipFile(zpath)
    info = json.loads(z.read("run_info.json"))
    pred = json.load(open(pred_path, encoding="utf-8"))
    print(f"預測是 {pred['made_at']} 做的；現在才打開排隊的資料")
    usable = {r["tag"]: r for r in info["runs"]}
    out = {"zip": pred["zip"], "pred_made_at": pred["made_at"], "compared_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "runs": []}
    for p in pred["runs"]:
        if "skipped" in p:                                   # 方法 2 沒做的組
            print(f"  {p['tag']}：{p['skipped']}")
            continue
        rows = [r for r in csv.DictReader(io.StringIO(z.read(f"queue_{p['tag']}.csv").decode("utf-8"))) if r["status"] == "ok"]
        rows.sort(key=lambda r: float(r["t_send"]))
        a = np.array([float(r["t_send"]) for r in rows])
        c = np.array([float(r["t_done"]) for r in rows])
        prev = served_prev(a, c)                              # 10/22 改：照做完的順序（原本照送出順序）
        w = np.maximum(a, prev) - a
        k = len(a) // 3
        m = float(w[k:].mean())
        hit = p["lo"] <= m <= p["hi"]
        closer = abs(m - p["pred"]) < abs(m - p["naive_pk"])
        u = usable.get(p["tag"], {})
        err = 100 * (p["pred"] - m) / m                      # 10/21 加：誤差（預測 − 實測，佔實測的 %）
        print(f"  {p['tag']}：實測平均等待 {m:.3f} 秒｜預測 {p['pred']} [{p['lo']}, {p['hi']}] → {'在範圍內' if hit else '在範圍外'}"
              f"（誤差 {err:+.1f}%）｜天真 P-K {p['naive_pk']} → {'比較接近模型' if closer else '比較接近天真 P-K'}｜"
              f"可用：{u.get('usable')}（成功 {u.get('ok')}/{u.get('n')}，送出延誤最大 {u.get('max_lateness_s')} 秒）")
        out["runs"].append({"tag": p["tag"], "measured_wait": round(m, 4), "in_range": bool(hit), "closer_to_model": bool(closer),
                            "usable": u.get("usable"), "error_pct": round(err, 2)})
    if out["runs"]:
        out["mean_abs_error_pct"] = round(float(np.mean([abs(r["error_pct"]) for r in out["runs"]])), 2)
        print(f"  {len(out['runs'])} 組中 {sum(r['in_range'] for r in out['runs'])} 組在範圍內；誤差絕對值的平均 {out['mean_abs_error_pct']}%")
    # 10/20 加：結果也存成 JSON（和預測放在同一個資料夾，檔名把「預測_」換成「結果_」）
    res_path = pred_path.replace("預測_", "結果_")
    if res_path != pred_path:
        json.dump(out, open(res_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"→ {res_path}")


if __name__ == "__main__":
    {"predict": predict, "predict2": predict2, "predict3": predict3, "predict3t": predict3t, "predict3s": predict3s,
     "predict3s_v1": predict3s_v1, "predict5": predict5, "compare": compare}[sys.argv[1]](sys.argv[2], sys.argv[3])
