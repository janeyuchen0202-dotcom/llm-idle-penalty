"""保溫取捨實測的分析（2026-10-19，新第 VI 節）：照 事前預期.md（09:12）寫好的判定標準

- 等待、回應時間：照送出時間排序；開始服務 = max(送出, 前一個完成)（和 analyze_plan.py 相同）；
  等待 = 開始 − 送出，回應時間 = 完成 − 送出；丟掉前 1/3（60 個）
- 平均功耗：量測區間從第 60 個請求做完到最後一個做完；load_gen.py 印的開始時間（UTC）加上相對時間，
  和 nvidia-smi 的時間（本機時間）對齊；區間內讀數的平均（每 0.1 秒一筆，均勻取樣）
- 判定：等待、回應時間落在預測的 95% 範圍內算命中；功耗在預測值 ±10% 內算命中

用法：python analyze_power.py locked default（有哪組就分析哪組）
"""
import csv
import datetime
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def analyze(cond):
    # 兩端是 plan_power_locked／default.json；保溫中間點（run_keepwarm.py）是 plan_kw_tau3.json 這種
    name = f"plan_power_{cond}.json" if cond in ("locked", "default") else f"plan_{cond}.json"
    plan = json.load(open(os.path.join(HERE, name), encoding="utf-8"))
    run = plan["runs"][0]
    tag = run["tag"]
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "out", f"{tag}.csv"), encoding="utf-8")) if r["status"] == "ok"]
    rows.sort(key=lambda r: float(r["t_send"]))
    a = np.array([float(r["t_send"]) for r in rows])
    c = np.array([float(r["t_done"]) for r in rows])
    prev = np.empty(len(c)); _cp = -np.inf    # 10/22 改：照做完的順序重建（一次只處理一個請求，做完的順序就是處理的順序；原本照送出順序）
    for _j in np.argsort(c, kind="stable"):
        prev[_j] = _cp; _cp = c[_j]
    start = np.maximum(a, prev)
    wait, resp = start - a, c - a
    n, k = len(rows), run["n"] // 3
    log = open(os.path.join(HERE, "out", f"{tag}_runlog.txt"), encoding="utf-8").read()
    m = re.search(r"實驗開始時間（UTC）：(\S+)", log)
    t0 = datetime.datetime.fromisoformat(m.group(1)).timestamp()
    T0, T1 = t0 + c[k - 1], t0 + c[-1]
    ts, pw, ps, ck = [], [], [], []
    for p in csv.reader(open(os.path.join(HERE, "out", f"{tag}_gpu.csv"), encoding="utf-8")):
        try:                                     # 整行都讀得到才收（10/19：有一行的欄位是讀不到的值，原本會讓長度對不上）
            t_, s_, c_, w_ = datetime.datetime.strptime(p[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(), p[1].strip(), \
                float(p[2]), float(p[5])
        except (ValueError, IndexError):
            continue
        ts.append(t_)
        ps.append(s_)
        ck.append(c_)
        pw.append(w_)
    ts, pw, ck, ps = np.array(ts), np.array(pw), np.array(ck), np.array(ps)
    sel = (ts >= T0) & (ts <= T1)
    p_mean = float(pw[sel].mean())
    idle_found = float(np.mean((a[k:] >= prev[k:])))
    res = {"cond": cond, "n_ok": n, "n_planned": run["n"], "wait": float(wait[k:].mean()), "resp": float(resp[k:].mean()),
           "service": float((c - start)[k:].mean()), "power_W": p_mean, "energy_per_request_J": p_mean * (T1 - T0) / (n - k),
           "idle_found_frac": idle_found, "gpu_samples": int(sel.sum()),
           "p8_frac": float(np.mean(ps[sel] == "P8")), "clock_median_MHz": float(np.median(ck[sel])),
           "window_min": (T1 - T0) / 60}
    hit_w = run["lo"] <= res["wait"] <= run["hi"]
    hit_r = run["resp_lo"] <= res["resp"] <= run["resp_hi"]
    hit_p = abs(res["power_W"] / run["power_pred_W"] - 1) <= 0.10
    print(f"== {cond}（{tag}）：成功 {n}/{run['n']}，量測區間 {res['window_min']:.1f} 分鐘、GPU 紀錄 {res['gpu_samples']} 筆")
    print(f"   平均等待 {res['wait']:.3f} 秒｜預測 {run['pred']} [{run['lo']}, {run['hi']}] → {'命中' if hit_w else '沒中'}")
    print(f"   平均回應時間 {res['resp']:.3f} 秒｜預測 {run['resp_pred']} [{run['resp_lo']}, {run['resp_hi']}] → {'命中' if hit_r else '沒中'}")
    print(f"   顯示卡平均功耗 {res['power_W']:.1f} W｜預測 {run['power_pred_W']} W（±10%：{run['power_pred_W'] * 0.9:.1f}–{run['power_pred_W'] * 1.1:.1f}）"
          f" → {'命中' if hit_p else '沒中'}（差 {100 * (res['power_W'] / run['power_pred_W'] - 1):+.1f}%）")
    print(f"   每個請求的能量 {res['energy_per_request_J']:.0f} J｜平均服務時間 {res['service']:.3f} 秒｜"
          f"到達時伺服器閒著的比例 {idle_found:.0%}｜P8 的時間比例 {res['p8_frac']:.0%}｜時脈中位數 {res['clock_median_MHz']:.0f} MHz")
    res.update(hit_wait=hit_w, hit_resp=hit_r, hit_power=hit_p)
    return res


def main():
    out = {c: analyze(c) for c in sys.argv[1:]}
    if "locked" in out and "default" in out:
        l, d = out["locked"], out["default"]
        print(f"\n鎖時脈 − 預設：回應時間 {l['resp'] - d['resp']:+.3f} 秒，平均功耗 {l['power_W'] - d['power_W']:+.1f} W，"
              f"每個請求的能量 {l['energy_per_request_J'] - d['energy_per_request_J']:+.0f} J")
    if len(out) > 1:
        print("\n各組：" + "｜".join(f"{c} 回應 {r['resp']:.3f} 秒、{r['power_W']:.1f} W" for c, r in out.items()))
    # 合併進原本的結果檔（10/19：只分析新的幾組時，不要把兩端的結果蓋掉）；試流程（kwq_）不寫
    path = os.path.join(HERE, "out", "power_results.json")
    keep = {c: r for c, r in out.items() if not c.startswith("kwq_")}
    if keep:
        old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        old.update(keep)
        json.dump(old, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
