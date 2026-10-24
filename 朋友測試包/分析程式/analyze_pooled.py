"""同一台跑了很多次：每一份先照排除規則逐輪檢查（analyze_friend.analyze），再依電源計畫分組合併計算。
螢幕開和關的輪次分開比較（排除規則「螢幕狀態」）。用法：python -I analyze_pooled.py <zip> ..."""
import json, os, sys, zipfile
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_friend import analyze, welch


def screen_state(st_, en_):
    if (st_ or 0) >= 1 and (en_ or 0) >= 1:
        return "螢幕開"
    if st_ == 0 and en_ == 0:
        return "螢幕關"
    return "中途變了"


groups, per_run = {}, []
for p in sys.argv[1:]:
    _, res = analyze(p)
    hw = json.loads(zipfile.ZipFile(p).read("hardware.json"))
    plan = ((hw.get("power") or {}).get("active_scheme") or "").split("(")[-1].rstrip(") ") or "?"
    run = p[-17:-4]
    mon = {ri["round"]: (ri.get("monitors_active_start"), ri.get("monitors_active_end")) for ri in res["info"]["rounds_info"]}
    g = groups.setdefault((res["mode"], plan), {"rows": [], "runs": [], "logs": {}, "checks": [], "dropped": 0})
    for r in res["keep"]:
        g["rows"].append(dict(r, run=run, key=(run, r["round"]), screen=screen_state(*mon.get(r["round"], (None, None)))))
    g["runs"].append(run)
    g["dropped"] += len(res["dropped"])
    g["checks"] += [(run, c["round"], c["reasons"]) for c in res["round_check"]]
    for k, L in res["gpu_logs"].items():
        if k in res["valid_rounds"]:                     # 耗電、P-state 也只用算數的輪
            g["logs"][(run, k)] = L
    per_run.append((res["mode"], plan, run, len(res["valid_rounds"]), len(res["round_check"]), res["d"], res["lo"], res["hi"], res["b0"],
                    sorted({screen_state(*v) for v in mon.values()})))


def report(name, rows, logs=None):
    S = lambda f: [r["service"] for r in rows if f(r)]
    b0, g3 = S(lambda r: r["gap"] == 0), S(lambda r: r["gap"] >= 3)
    if len(b0) < 3 or len(g3) < 3:
        print(f"  {name}：資料太少（{len(b0)}／{len(g3)}）")
        return
    d, lo, hi = welch(g3, b0)
    d2, lo2, hi2 = welch(S(lambda r: r["gap"] == 2), b0)
    per = {}
    for r in rows:
        c = "0" if r["gap"] == 0 else "3" if r["gap"] >= 3 else None
        if c:
            per.setdefault(r["key"], {"0": [], "3": []})[c].append(r["service"])
    pr = np.array([np.mean(v["3"]) - np.mean(v["0"]) for v in per.values() if v["0"] and v["3"]])
    tq = stats.t.ppf(0.975, len(pr) - 1) if len(pr) > 1 else float("nan")
    half = tq * pr.std(ddof=1) / np.sqrt(len(pr)) if len(pr) > 1 else float("nan")
    gaps = sorted({r["gap"] for r in rows})
    print(f"  {name}：一個接一個 {np.mean(b0):.3f} 秒（n={len(b0)}）｜**閒置 ≥3 秒 {d:+.3f} 秒（95% CI {lo:+.3f}, {hi:+.3f}），{d / np.mean(b0):+.1%}**｜"
          f"閒置 2 秒 {d2:+.3f}（{lo2:+.3f}, {hi2:+.3f}）")
    print("     每種閒置秒數：" + "、".join(f"{int(x)} 秒 {np.mean(S(lambda r, x=x: r['gap'] == x)):.3f}" for x in gaps) +
          f"｜各輪：{len(pr)} 輪，平均 {pr.mean():+.3f}，標準差 {pr.std(ddof=1):.3f}，輪間 CI ({pr.mean() - half:+.3f}, {pr.mean() + half:+.3f})")
    T = lambda f: [r["ttft"] for r in rows if f(r) and r["ttft"] is not None]
    D = lambda f: [r["service"] - r["ttft"] for r in rows if f(r) and r["ttft"] is not None]
    dt_, lot, hit = welch(T(lambda r: r["gap"] >= 3), T(lambda r: r["gap"] == 0))
    dd_, lod, hid = welch(D(lambda r: r["gap"] >= 3), D(lambda r: r["gap"] == 0))
    print(f"     拆開：讀 prompt {dt_:+.3f}（{lot:+.3f}, {hit:+.3f}）、產生輸出 {dd_:+.3f}（{lod:+.3f}, {hid:+.3f}）")
    if logs:
        offs = [0.5, 1, 1.5, 2, 3, 5, 7.5]
        clk = {o: [] for o in offs}
        first, n8 = [], 0
        for r in rows:
            L = logs.get(r["key"])
            if not L or r["gap"] != 8:
                continue
            n8 += 1
            ts = np.array([x[0] for x in L])
            ws = r["wall_send"] - r["idle_before"]
            for o in offs:
                i = int(np.argmin(np.abs(ts - (ws + o))))
                if abs(ts[i] - (ws + o)) < 0.3:
                    clk[o].append(L[i][2])
            w = [x[0] - ws for x in L if ws <= x[0] <= r["wall_send"] and x[1] == "P8"]
            if w:
                first.append(min(w))
        allp = [x for L in logs.values() for x in L]
        p8 = [x[6] for x in allp if x[1] == "P8"]
        idle = [x[6] for x in allp if x[4] == 0]
        busy = [x[6] for x in allp if x[4] >= 50]
        print("     時脈（閒置 8 秒的請求，前一個結束後）：" + "、".join(f"{o} 秒 {np.median(clk[o]):.0f}" for o in offs if clk[o]) + " MHz" +
              (f"｜進 P8 中位數 {np.median(first):.1f} 秒（{len(first)}／{n8} 個有進）" if first else "｜8 秒內沒進 P8"))
        print(f"     耗電：P8 {np.median(p8):.1f} W｜閒著 {np.median(idle):.1f} W｜忙碌 {np.median(busy):.1f} W" if p8 else
              f"     耗電：沒有 P8｜閒著 {np.median(idle):.1f} W｜忙碌 {np.median(busy):.1f} W")
        print(f"     出現過的 P-state：{sorted({x[1] for x in allp})}｜最高時脈 {max(x[2] for x in allp):.0f} MHz｜溫度 {min(x[5] for x in allp):.0f}–{max(x[5] for x in allp):.0f} °C")


print("## 每一次各自算（照時間）")
for m, plan, run, nv, nr, d, lo, hi, b0, sc in sorted(per_run, key=lambda x: x[2]):
    print(f"- {run}｜{m.upper()}｜{plan}｜{nv}/{nr} 輪算數｜一個接一個 {b0:.3f}｜閒置 ≥3 秒 {d:+.3f}（{lo:+.3f}, {hi:+.3f}）｜{'、'.join(sc)}")

for (mode, plan), g in sorted(groups.items()):
    bad = [c for c in g["checks"] if c[2]]
    print(f"\n## {mode.upper()} 測試｜電源計畫 {plan}｜{len(g['runs'])} 次")
    print(f"  逐輪檢查：{len(g['checks']) - len(bad)}／{len(g['checks'])} 輪算數；不算的：{bad or '沒有'}｜單一請求刪掉 {g['dropped']} 個")
    report("全部", g["rows"], g["logs"] if mode == "gpu" else None)
    for sc in ("螢幕開", "螢幕關", "中途變了"):
        sub = [r for r in g["rows"] if r["screen"] == sc]
        if sub:
            report(f"{sc}（{len({r['key'] for r in sub})} 輪）", sub, g["logs"] if mode == "gpu" else None)
