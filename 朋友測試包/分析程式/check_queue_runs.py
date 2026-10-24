"""朋友排隊測試打開資料之後的檢查（2026-10-20；第 1 項是事先定好的排除規則，其餘是事後檢查）

1. 排除規則（排隊測試/設計說明.md）：每段閒置（前一個完成後 1.2 秒，到下一個送出前 0.2 秒）nvidia-smi 的平均使用率
2. 服務時間：排隊中的請求（一個接一個做）和校正的 b 比；照時間分四段看有沒有越跑越慢
3. 顯示卡狀態：忙的時候的時脈、溫度、功耗（每組、照時間分四段）
4. 用每組自己排隊請求的服務時間重新模擬（其他參數和預測相同），看偏差有多少是服務時間變長造成的

用法：python -I check_queue_runs.py <結果zip> <預測.json>
"""
import csv
import datetime
import io
import json
import random
import sys
import zipfile

import numpy as np

EDGES = [1.0, 2.0, 5.0]


def gpu_log(z):
    rows = []
    for p in csv.reader(io.StringIO(z.read("round_all_gpu_nvidia.csv").decode("utf-8", "replace"))):
        try:
            t = datetime.datetime.strptime(p[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            rows.append((t, p[2].strip(), float(p[3]), float(p[5]), float(p[6]), float(p[7])))
        except (ValueError, IndexError):
            continue
    a = np.array([(r[0], r[2], r[3], r[4], r[5]) for r in rows])
    return a[:, 0], np.array([r[1] for r in rows]), a[:, 1], a[:, 2], a[:, 3], a[:, 4]   # 時間、P-state、時脈、使用率、溫度、功耗


def main(zpath, pred_path):
    z = zipfile.ZipFile(zpath)
    pred = json.load(open(pred_path, encoding="utf-8"))
    cal = [r for r in csv.DictReader(io.StringIO(z.read("calibration.csv").decode("utf-8"))) if not r["error"]]
    S0 = np.array([float(r["service"]) for r in cal if float(r["gap"]) == 0])
    by_bin = [[] for _ in range(4)]
    for r in cal:
        if float(r["gap"]) > 0:
            by_bin[int(np.digitize(float(r["idle_before"]), EDGES))].append(float(r["service"]))
    by_bin = [np.array(x) if x else np.array([S0.mean()]) for x in by_bin]
    ts, ps, gr, ut, tc, pw = gpu_log(z)
    cal_w = np.array([float(r["wall_send"]) for r in cal])
    cal_sel = (ts >= cal_w.min()) & (ts <= cal_w.max() + 10)
    print(f"校正：b = {S0.mean():.4f} 秒（{len(S0)} 個）；校正期間顯示卡溫度中位數 {np.median(tc[cal_sel]):.0f}°C")

    for p in pred["runs"]:
        rows = [r for r in csv.DictReader(io.StringIO(z.read(f"queue_{p['tag']}.csv").decode("utf-8"))) if r["status"] == "ok"]
        rows.sort(key=lambda r: float(r["t_send"]))
        a = np.array([float(r["t_send"]) for r in rows])
        c = np.array([float(r["t_done"]) for r in rows])
        off = np.median([float(r["wall_send"]) - float(r["t_send"]) for r in rows])
        prev = np.empty(len(c)); _cp = -np.inf    # 10/22 改：照做完的順序重建（一次只處理一個請求，做完的順序就是處理的順序；原本照送出順序）
        for _j in np.argsort(c, kind="stable"):
            prev[_j] = _cp; _cp = c[_j]
        start = np.maximum(a, prev)
        svc = c - start
        n, k = len(a), len(a) // 3
        queued = a < prev
        print(f"\n== {p['tag']}（{n} 個，丟掉前 {k} 個）")

        # 1. 排除規則：閒置期間的使用率
        segs = [(c[i] + 1.2, a[i + 1] - 0.2) for i in range(k, n - 1) if a[i + 1] - 0.2 > c[i] + 1.2]
        u_all, w_all, n_hi = [], [], 0
        for s, e in segs:
            sel = (ts >= s + off) & (ts <= e + off)
            if sel.any():
                m = ut[sel].mean()
                u_all.append(m)
                w_all.append(e - s)
                n_hi += m > 10
        if u_all:
            print(f"  1. 閒置期間的顯示卡使用率：{len(u_all)} 段，時間加權平均 {np.average(u_all, weights=w_all):.1f}%，"
                  f"平均超過 10% 的有 {n_hi} 段")
        else:
            print("  1. 沒有夠長的閒置段（每段要超過 1.4 秒）")

        # 2. 服務時間
        sq = svc[k:][queued[k:]]
        print(f"  2. 排隊中請求的服務時間 平均 {sq.mean():.4f} 秒（校正 b 的 {sq.mean() / S0.mean():.3f} 倍，{len(sq)} 個）")
        idx = np.arange(k, n)
        q4 = np.array_split(idx, 4)
        print("     照時間分四段（排隊中的請求）：" + "、".join(
            f"{svc[g][queued[g]].mean():.3f}" if queued[g].any() else "—" for g in q4))
        busy = np.zeros(len(ts), bool)
        sel_win = (ts >= start[k] + off) & (ts <= c[-1] + off)
        for i in range(k, n):
            busy |= (ts >= start[i] + off + 0.1) & (ts <= c[i] + off - 0.05)
        util_real = float(np.sum(c[k:] - start[k:]) / (c[-1] - start[k]))
        print(f"     量測區間實際的忙碌比例（ρ）：{util_real:.3f}")

        # 3. 顯示卡狀態（忙的時候）
        bsel = busy & sel_win
        print(f"  3. 忙的時候：時脈中位數 {np.median(gr[bsel]):.0f} MHz、溫度中位數 {np.median(tc[bsel]):.0f}°C、功耗中位數 {np.median(pw[bsel]):.0f} W")
        tq = np.array_split(np.where(bsel)[0], 4)
        print("     照時間分四段：時脈 " + "、".join(f"{np.median(gr[g]):.0f}" for g in tq) + " MHz｜溫度 " +
              "、".join(f"{np.median(tc[g]):.0f}" for g in tq) + "°C｜功耗 " + "、".join(f"{np.median(pw[g]):.0f}" for g in tq) + " W")

        # 4. 用這組自己排隊請求的服務時間重新模擬（δ 和到達時間跟預測相同）
        r_, t, arr = random.Random(p["seed"]), 0.0, []
        for _ in range(p["n"]):
            t += r_.expovariate(p["lambda"])
            arr.append(t)
        arr = np.array(arr)
        rng = np.random.default_rng(29)
        R = 2000
        own = svc[queued]
        base = own[rng.integers(0, len(own), (R, p["n"]))]
        dv = np.stack([by_bin[i][rng.integers(0, len(by_bin[i]), (R, len(by_bin[i])))].mean(1) for i in range(4)], 1) - \
            S0[rng.integers(0, len(S0), (R, len(S0)))].mean(1)[:, None]
        c_prev, acc = np.full(R, -np.inf), np.zeros(R)
        for i in range(p["n"]):
            st = np.maximum(arr[i], c_prev)
            w = st - arr[i]
            pen = np.where(w == 0, dv[np.arange(R), np.digitize(arr[i] - c_prev, EDGES)], 0.0)
            c_prev = st + base[:, i] + pen
            if i >= k:
                acc += w
        s = acc / (p["n"] - k)
        wm = float((np.maximum(a, prev) - a)[k:].mean())
        lo, hi = np.percentile(s, [2.5, 97.5])
        print(f"  4. 用這組自己排隊請求的服務時間（{own.mean():.4f} 秒）重新模擬：{s.mean():.3f}（{lo:.3f}–{hi:.3f}）｜"
              f"實測 {wm:.3f}｜原本的預測 {p['pred']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
