"""RTX 4050 補充實驗 B 的分析：保溫五個點（同一段時間）。判定標準：事前預期.md 2026-10-21 的事前登記
- 每點用 analyze_power.py 的 analyze()：等待、回應時間落在預測的 95% 範圍內算命中，功耗在預測值 ±10% 內算命中 → 15 項
- 形狀：兩端（τ = 0、∞）實測連成的直線上、同一個功耗的回應時間；中間三點的回應時間 − 連線 ≥ −0.01 秒（三點都要）
  →「部分保溫沒有比切換好」再確認一次
- 只描述：方向（τ 越大，回應時間越短、功耗越高）、和 10/19 五個點的差（out/power_results.json）
用法：python analyze_keepwarm5.py [kw5|kwq5]     （kwq5：試流程的結果）
"""
import json
import os
import sys

import analyze_power as ap

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["0", "3", "6", "12", "inf"]
OLD = {"0": "default", "3": "kw_tau3", "6": "kw_tau6", "12": "kw_tau12", "inf": "locked"}     # 10/19 的五個點
SHAPE_TOL = -0.01


def main():
    pfx = sys.argv[1] if len(sys.argv) > 1 else "kw5"
    res = {n: ap.analyze(f"{pfx}_tau{n}") for n in NAMES}
    print()
    lab = {"0": "τ = 0（不保溫）", "3": "τ = 3 秒", "6": "τ = 6 秒", "12": "τ = 12 秒", "inf": "τ = ∞（一直鎖）"}
    for k, name in (("hit_wait", "平均等待"), ("hit_resp", "平均回應時間"), ("hit_power", "平均功耗")):
        print(f"{name}：{sum(res[n][k] for n in NAMES)}/5 命中（" + "、".join(f"{n} {'✓' if res[n][k] else '✗'}" for n in NAMES) + "）")
    n_hit = sum(res[n][k] for n in NAMES for k in ("hit_wait", "hit_resp", "hit_power"))
    print(f"**15 項中 {n_hit} 項命中**")

    d, l = res["0"], res["inf"]
    dp = l["power_W"] - d["power_W"]
    gaps, shape = {}, None
    if abs(dp) > 1e-6:
        slope = (l["resp"] - d["resp"]) / dp
        gaps = {n: res[n]["resp"] - (d["resp"] + slope * (res[n]["power_W"] - d["power_W"])) for n in ("3", "6", "12")}
        shape = all(g >= SHAPE_TOL for g in gaps.values())
        print(f"\n形狀（回應時間 − 兩端實測連線，同功耗；≥ {SHAPE_TOL} 秒算在連線上或上方）：" +
              "、".join(f"τ = {n} 秒 {g:+.3f}" for n, g in gaps.items()) +
              f" → **{'成立：部分保溫沒有比切換好（再確認）' if shape else '不成立：至少一點在連線下方超過 0.01 秒'}**")
    else:
        print("\n形狀：兩端功耗一樣（試流程沒有鎖時脈），無法判定")
    order = sorted(NAMES, key=lambda n: float(n))
    mono = all(res[a]["resp"] >= res[b]["resp"] and res[a]["power_W"] <= res[b]["power_W"] for a, b in zip(order, order[1:]))
    print(f"方向（只描述）：τ 越大，回應時間越短、功耗越高 → {'五個點都符合' if mono else '有不符合的'}")

    path = os.path.join(HERE, "out", "power_results.json")
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    if all(OLD[n] in old for n in NAMES):
        print("\n和 10/19 比（只描述；10/19 兩端在早上、中間三點在晚上量）：")
        for n in NAMES:
            o, r = old[OLD[n]], res[n]
            print(f"  {lab[n]}：回應時間 {r['resp']:.3f} 秒（10/19 {o['resp']:.3f}，差 {r['resp'] - o['resp']:+.3f}）｜"
                  f"功耗 {r['power_W']:.1f} W（10/19 {o['power_W']:.1f}，差 {r['power_W'] - o['power_W']:+.1f}）｜"
                  f"等待 {r['wait']:.3f} 秒（10/19 {o['wait']:.3f}）")
    prev = os.path.join(HERE, "out", "kw5_results.json")           # 10/22 加：重跑（例如 kw5m）時也和 10/22 凌晨那次比
    if pfx not in ("kw5", "kwq5") and os.path.exists(prev):
        p5 = json.load(open(prev, encoding="utf-8"))
        print("\n和 10/22 凌晨（kw5，沒接外接螢幕）比（只描述）：")
        for n in NAMES:
            o, r = p5["points"][n], res[n]
            print(f"  {lab[n]}：回應時間 {r['resp']:.3f} 秒（kw5 {o['resp']:.3f}，差 {r['resp'] - o['resp']:+.3f}）｜"
                  f"功耗 {r['power_W']:.1f} W（kw5 {o['power_W']:.1f}，差 {r['power_W'] - o['power_W']:+.1f}）｜等待 {r['wait']:.3f} 秒（kw5 {o['wait']:.3f}）")
        print("  形狀（kw5）：" + "、".join(f"τ = {n} 秒 {g:+.3f}" for n, g in p5["shape_gaps_s"].items()))
    out = {"prefix": pfx, "points": res, "hits": n_hit, "shape_gaps_s": gaps, "shape_ok": shape, "direction_ok": mono}
    json.dump(out, open(os.path.join(HERE, "out", f"{pfx}_results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n→ out/{pfx}_results.json")


if __name__ == "__main__":
    main()
