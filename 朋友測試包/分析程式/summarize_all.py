"""所有機器的彙整（2026-10-19）：從原始 zip 重新計算，輸出論文要用的總表、排除清單、圖

- 只收「已經揭曉」的機器（規格表與預測.md 裡有「## 揭曉」那一節），新機器在使用者預測之前不會被打開
- 每一份先用 analyze_friend.analyze 照排除規則逐輪檢查；同一台、同一種測試（第 3 台再分電源設定）的幾份合併算
- 閒置代價＝閒置 ≥3 秒的服務時間 − 一個接一個的服務時間（Welch 95% CI），和論文表 I 相同
輸出（朋友結果/彙整/）：all_results.csv、彙整.md、penalty_by_device.png
用法：python -I summarize_all.py <朋友結果資料夾>
"""
import csv
import glob
import json
import os
import re
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_friend import analyze, welch            # noqa: E402

ROOT = sys.argv[1]
OUT = os.path.join(ROOT, "彙整")
# 每台的基本資料（從規格表整理；只用來標示）
MACHINES = {
    "01_RTX5060Ti_9700X_20261018": ("01", "桌機", "RTX 5060 Ti 16GB", "50", "Ryzen 7 9700X"),
    "02_RTX5060Laptop_20261018": ("02", "筆電", "RTX 5060 Laptop 8GB", "50", "Ryzen 9 8940HX"),
    "03_RTX4060Laptop_20261018": ("03", "筆電", "RTX 4060 Laptop 8GB", "40", "Ryzen 9 7940H"),
    "04_RTX2060_5800X_20261018": ("04", "桌機", "RTX 2060 6GB", "20", "Ryzen 7 5800X"),
    "05_RX6600_3800X_20261018": ("05", "桌機", "（AMD RX 6600，GPU 測試沒跑成）", "-", "Ryzen 7 3800X"),
    "06_RTX2070S_5800X_20261019": ("06", "桌機", "RTX 2070 SUPER 8GB", "20", "Ryzen 7 5800X"),
    "07_RTX5070_U7-265K_20261019": ("07", "桌機", "RTX 5070 12GB", "50", "Intel Core Ultra 7 265K"),
    "08_RTX5070Ti_i7-11700_20261020": ("08", "桌機", "RTX 5070 Ti 16GB", "50", "Intel Core i7-11700"),
    "09_RTX3060Ti_i5-12400_20261020": ("09", "桌機", "RTX 3060 Ti 8GB", "30", "Intel Core i5-12400"),
    "10_RTX4060Laptop_i5-12500H_20261021": ("10", "筆電", "RTX 4060 Laptop 8GB", "40", "Intel Core i5-12500H"),
    "M1_MacBookAir_20261019": ("M1", "筆電（沒風扇）", "Apple M1 7-core GPU", "Apple", "Apple M1"),
    "M1Pro_MacBookPro_20261023": ("M1Pro", "筆電（有風扇）", "Apple M1 Pro 16-core GPU", "Apple", "Apple M1 Pro"),   # 10/23 加
}
# 使用者的事前預測（照各台 規格表與預測.md 的原文；會不會變慢、從幾秒開始、多幾秒）
PRED = {
    ("01", "GPU", ""): ("會", "3（事後說明）", "—"), ("01", "CPU", ""): ("不會", "—", "—"),
    ("02", "GPU", ""): ("會", "2", "0.23"), ("02", "CPU", ""): ("幾乎不會", "4", "0.1"),
    ("03", "GPU", "Turbo"): ("會", "3", "0.08"), ("03", "GPU", "Silent"): ("會", "6", "0.5"),
    ("03", "CPU", "Turbo"): ("幾乎不會", "3", "0.09"), ("03", "CPU", "Silent"): ("幾乎不會", "3", "0.3"),
    ("04", "GPU", ""): ("會", "3", "0.2"), ("04", "CPU", ""): ("幾乎不會", "0", "0"),
    ("05", "CPU", ""): ("幾乎不會", "0", "0"),
    ("06", "GPU", ""): ("會", "3", "0.3"), ("06", "CPU", ""): ("幾乎不會", "0", "0.05"),
    ("07", "GPU", ""): ("會", "2", "0.3"), ("07", "CPU", ""): ("會", "3", "0.5"),
    ("08", "GPU", ""): ("會", "2", "0.5"), ("08", "CPU", ""): ("幾乎不會", "無", "0.02"),
    ("09", "GPU", ""): ("會", "3", "0.2"), ("09", "CPU", ""): ("幾乎不會", "0", "0.02"),
    ("10", "GPU", ""): ("會", "2", "0.35"), ("10", "CPU", ""): ("幾乎不會", "2", "0.02"),
    ("M1", "GPU", ""): ("不會", "沒有", "−0.05"),
    ("M1Pro", "GPU", ""): ("不會（反而變快）", "3", "−0.3"),      # 10/23 20:24 原話「變快但沒快這麼多 3 -0.3 M晶片」
}
MEANINGFUL = 0.02          # 判斷「會不會變慢」：代價 ≥ 20 毫秒算會（CPU 的 3–8 毫秒算幾乎沒有）
# 10/20 使用者決定：「幾乎不會」和「不會」不同。說「幾乎不會」的，實測代價小於一個接一個的 2% 就算對
# （看過第 9 台才定的；只有第 9 台 CPU 這一題因此改變：照原本只分會／不會的算法是錯，現在算對）
ALMOST_NONE_MAX = 0.02


def plan_of(z):
    hw = json.loads(z.read("hardware.json"))
    s = ((hw.get("power") or {}).get("active_scheme") or "")
    return s.split("(")[-1].rstrip(") ") if "(" in s else ""


def main():
    os.makedirs(OUT, exist_ok=True)
    groups, excl = {}, []
    for folder, meta in MACHINES.items():
        d = os.path.join(ROOT, folder)
        spec = os.path.join(d, "規格表與預測.md")
        if not os.path.exists(spec) or "## 揭曉" not in open(spec, encoding="utf-8").read():
            print(f"跳過（還沒揭曉）：{folder}")
            continue
        for p in sorted(glob.glob(os.path.join(d, "閒置測試結果_*.zip"))):
            z = zipfile.ZipFile(p)
            _, res = analyze(p)
            test = "GPU" if res["mode"] == "gpu" else "CPU"
            cfg = plan_of(z) if meta[0] == "03" else ""
            g = groups.setdefault((meta[0], test, cfg), {"meta": meta, "rows": [], "checks": [], "dropped": 0, "files": [],
                                                         "logs": {}, "makeup": 0, "program_invalid": []})
            run = os.path.basename(p)[-17:-4]
            g["files"].append(os.path.basename(p))
            g["rows"] += [dict(r, key=(run, r["round"])) for r in res["keep"]]
            g["checks"] += [(run, c["round"], c["reasons"]) for c in res["round_check"]]
            g["dropped"] += len(res["dropped"])
            ri = res["info"]
            g["makeup"] += ri.get("makeup_rounds") or 0
            g["program_invalid"] += [(run, r["round"], r.get("invalid_reasons")) for r in ri["rounds_info"] if not r.get("valid")]
            for c in res["round_check"]:
                if c["reasons"]:
                    excl.append((meta[0], test, cfg, run, c["round"], "；".join(c["reasons"])))
            for r in res["dropped"]:
                excl.append((meta[0], test, cfg, run, r["round"], f"單一請求慢 3 倍以上（閒置 {r['gap']:g} 秒，{r['service']:.2f} 秒）"))
            for k, L in res["gpu_logs"].items():
                if not any(c["round"] == k and c["reasons"] for c in res["round_check"]):
                    g["logs"][(run, k)] = L

    table = []
    for (mid, test, cfg), g in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
        rows = g["rows"]
        S = lambda f: [r["service"] for r in rows if f(r)]
        b0 = S(lambda r: r["gap"] == 0)
        g3 = S(lambda r: r["gap"] >= 3)
        d, lo, hi = welch(g3, b0)
        per_gap = {}
        for gap in sorted({r["gap"] for r in rows if r["gap"] > 0}):
            per_gap[gap] = welch(S(lambda r, gap=gap: r["gap"] == gap), b0)
        T = lambda f: [r["ttft"] for r in rows if f(r) and r["ttft"] is not None]
        dt = welch(T(lambda r: r["gap"] >= 3), T(lambda r: r["gap"] == 0))[0] if T(lambda r: r["gap"] == 0) else float("nan")
        per = {}
        for r in rows:
            c = "0" if r["gap"] == 0 else "3" if r["gap"] >= 3 else None
            if c:
                per.setdefault(r["key"], {"0": [], "3": []})[c].append(r["service"])
        pr = np.array([np.mean(v["3"]) - np.mean(v["0"]) for v in per.values() if v["0"] and v["3"]])
        # 10/20 加：以「輪」為單位的 95% CI（每輪各自算一個代價，再用 t 分布；輪與輪之間的變異都算進去）
        from scipy import stats
        r_half = stats.t.ppf(0.975, len(pr) - 1) * pr.std(ddof=1) / np.sqrt(len(pr)) if len(pr) > 1 else float("nan")
        valid = sum(1 for c in g["checks"] if not c[2])
        p8 = [x[6] for L in g["logs"].values() for x in L if x[1] == "P8"] if test == "GPU" else []
        onset = next((f"{gap:g}" for gap, (m, l, h) in per_gap.items() if l > 0 and m >= MEANINGFUL), "沒有")
        pred = PRED.get((mid, test, cfg), ("", "", ""))
        slowed = d >= MEANINGFUL
        said = pred[0]
        dir_ok = "" if not said else ("對" if (said == "會") == slowed else "錯")
        if said == "幾乎不會":
            dir_ok = "對" if d / np.mean(b0) < ALMOST_NONE_MAX else "錯"
        table.append({"機器": mid, "類型": g["meta"][1], "裝置": g["meta"][2] if test == "GPU" else g["meta"][4],
                      "世代": g["meta"][3] if test == "GPU" else ("Intel" if "Intel" in g["meta"][4] else "AMD" if "Ryzen" in g["meta"][4] else "Apple"),
                      "測試": test, "設定": cfg, "份數": len(g["files"]), "有效輪": f"{valid}/{len(g['checks'])}",
                      "請求數": len(rows), "b": round(float(np.mean(b0)), 3), "b_CV": round(float(np.std(b0, ddof=1) / np.mean(b0)), 3),
                      "代價": round(d, 3), "CI_lo": round(lo, 3), "CI_hi": round(hi, 3), "比例": round(d / np.mean(b0), 3),
                      **{f"閒置{int(k) if k == int(k) else k}秒": round(v[0], 3) for k, v in per_gap.items()},
                      "讀prompt部分": round(dt, 3), "各輪標準差": round(float(pr.std(ddof=1)), 3) if len(pr) > 1 else "",
                      "輪數": len(pr), "輪平均": round(float(pr.mean()), 3), "輪CI_lo": round(float(pr.mean() - r_half), 3),
                      "輪CI_hi": round(float(pr.mean() + r_half), 3),
                      "明確開始": onset, "P8耗電W": round(float(np.median(p8)), 1) if p8 else "",
                      "預測方向": said, "預測起點": pred[1], "預測大小": pred[2], "方向對嗎": dir_ok,
                      "單一請求刪掉": g["dropped"], "程式補跑": g["makeup"]})
    fields = list(dict.fromkeys(k for t in table for k in t))
    with open(os.path.join(OUT, "all_results.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)

    # ---- 彙整.md
    L = ["# 朋友電腦閒置測試：彙整（summarize_all.py 自動產生，從原始 zip 重新計算）", "",
         "> 閒置代價＝閒置 ≥3 秒的服務時間 − 一個接一個的服務時間（Welch 95% CI），和論文表 I 相同。只收已經揭曉的機器。",
         "> 對照：使用者的 RTX 4050 筆電（論文的閒置測試）+0.29 秒（0.26, 0.31），16%，閒置 2 秒就開始。", "",
         "## GPU", "", "| 機器 | 類型 | 顯示卡 | 世代 | 有效輪 | 一個接一個（秒） | 閒置 ≥3 秒（95% CI） | 以輪算（95% CI） | 比例 | 2 秒 | 3 秒 | 5 秒 | 8 秒 | 明確開始 | 預測 | 方向 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in [x for x in table if x["測試"] == "GPU"]:
        L.append(f"| {t['機器']}{'（' + t['設定'] + '）' if t['設定'] else ''} | {t['類型']} | {t['裝置']} | {t['世代']} | {t['有效輪']} | {t['b']} | "
                 f"{t['代價']:+.3f}（{t['CI_lo']:+.3f}, {t['CI_hi']:+.3f}） | {t['輪平均']:+.3f}（{t['輪CI_lo']:+.3f}, {t['輪CI_hi']:+.3f}） | "
                 f"{100 * t['比例']:+.1f}% | {t.get('閒置2秒', ''):+.3f} | {t.get('閒置3秒', ''):+.3f} | "
                 f"{t.get('閒置5秒', ''):+.3f} | {t.get('閒置8秒', ''):+.3f} | {t['明確開始'] + (' 秒' if t['明確開始'] != '沒有' else '')} | {t['預測方向']}／{t['預測起點']}／{t['預測大小']} | {t['方向對嗎']} |")
    L += ["", "## CPU", "", "| 機器 | CPU | 有效輪 | 一個接一個（秒） | 閒置 ≥3 秒（95% CI） | 以輪算（95% CI） | 比例 | 預測 | 方向 |", "|---|---|---|---|---|---|---|---|---|"]
    for t in [x for x in table if x["測試"] == "CPU"]:
        L.append(f"| {t['機器']}{'（' + t['設定'] + '）' if t['設定'] else ''} | {t['裝置']} | {t['有效輪']} | {t['b']} | "
                 f"{t['代價']:+.3f}（{t['CI_lo']:+.3f}, {t['CI_hi']:+.3f}） | {t['輪平均']:+.3f}（{t['輪CI_lo']:+.3f}, {t['輪CI_hi']:+.3f}） | "
                 f"{100 * t['比例']:+.1f}% | {t['預測方向']}／{t['預測起點']}／{t['預測大小']} | {t['方向對嗎']} |")
    # 10/20 加：以請求算和以輪算，結論（CI 有沒有排除 0）是不是一樣
    sig = lambda lo, hi: "大於 0" if lo > 0 else ("小於 0" if hi < 0 else "包含 0")
    same = [t for t in table if sig(t["CI_lo"], t["CI_hi"]) == sig(t["輪CI_lo"], t["輪CI_hi"])]
    diff = [t for t in table if t not in same]
    L += ["", "## 以請求算 vs 以輪算（95% CI 有沒有排除 0）", "",
          f"- {len(table)} 組中 {len(same)} 組結論相同" + ("" if not diff else "；不同的：" + "、".join(
              f"{t['機器']} {t['測試']}{'（' + t['設定'] + '）' if t['設定'] else ''}（以請求：{sig(t['CI_lo'], t['CI_hi'])}，以輪：{sig(t['輪CI_lo'], t['輪CI_hi'])}）"
              for t in diff)),
          "- 以輪算的 CI 把輪與輪之間的變異（每輪重新啟動伺服器）也算進去，通常比較寬"]
    gpu = [x for x in table if x["測試"] == "GPU" and x["世代"] in ("20", "30", "40", "50")]
    cpu = [x for x in table if x["測試"] == "CPU"]
    has = [x for x in gpu if x["明確開始"] != "沒有"]                 # 有代價：CI 下限 > 0 而且 ≥ 20 毫秒（和「明確開始」同一個標準）
    # 10/21：寫上機器編號（第 3、10 台都是 RTX 4060 Laptop，只有第 10 台沒有代價）
    no = sorted({f"第 {x['機器']} 台 {x['裝置']}" for x in gpu if x["明確開始"] == "沒有"})
    n_has = len({x["機器"] for x in has}) + 1                          # 加上論文的 RTX 4050
    L += ["", "## 摘要", "",
          f"- NVIDIA GPU：{len({x['機器'] for x in gpu})} 張（加上論文的 RTX 4050 共 {len({x['機器'] for x in gpu}) + 1} 張），"
          f"{'全部' if not no else f'其中 {n_has} 張'}有代價，比例 {100 * min(x['比例'] for x in has):.1f}%–{100 * max(x['比例'] for x in has):.1f}%"
          f"（論文的 4050 是 16%）" + (f"；沒有代價：{'、'.join(no)}" if no else "")]
    for gen in ("20", "30", "40", "50"):
        xs = [x for x in gpu if x["世代"] == gen]
        if xs:
            L.append(f"  - {gen} 系列：" + "、".join(
                f"{x['裝置']}{'（' + x['設定'] + '）' if x['設定'] else ''} {100 * x['比例']:.1f}%"
                f"（{x['明確開始'] + ' 秒開始' if x['明確開始'] != '沒有' else '沒有代價'}）" for x in xs))
    L.append(f"- CPU：{len(cpu)} 組（{len({x['機器'] for x in cpu})} 台）都很小：最大 {max(x['代價'] for x in cpu):+.3f} 秒"
             f"（{100 * max(x['比例'] for x in cpu):.1f}%）")
    m1 = [x for x in table if x["機器"] == "M1"]
    if m1:
        L.append(f"- Apple M1 Air（沒風扇）：{m1[0]['代價']:+.3f} 秒（{100 * m1[0]['比例']:+.1f}%），閒置之後反而快（看起來是過熱降速）")
    judged = [x for x in table if x["方向對嗎"]]
    L.append(f"- 使用者的方向預測：{len(judged)} 題中 {sum(x['方向對嗎'] == '對' for x in judged)} 對、"
             f"{sum(x['方向對嗎'] == '錯' for x in judged)} 錯（「會」：代價 ≥ {MEANINGFUL * 1000:.0f} 毫秒；"
             f"「幾乎不會」：代價小於一個接一個的 {100 * ALMOST_NONE_MAX:.0f}%，10/20 看過第 9 台後使用者決定，只有第 9 台 CPU 因此從錯變對）")
    L += ["", "## 排除清單（排除規則第 6 點：每一筆都列出來）", ""]
    if excl:
        L += ["| 機器 | 測試 | 設定 | 哪一份 | 第幾輪 | 原因 |", "|---|---|---|---|---|---|"]
        L += [f"| {a} | {b} | {c} | {d_} | {e} | {f_} |" for a, b, c, d_, e, f_ in excl]
    else:
        L.append("沒有")
    L += ["", "程式自己判定不算、已經自動補跑的輪次（測試包在跑的時候就處理掉了）：", ""]
    prog = [(k[0], k[1], k[2], x) for k, g in groups.items() for x in g["program_invalid"]]
    L += [f"- {a} {b}{'（' + c + '）' if c else ''}：{x[0]} 第 {x[1]} 輪，{x[2]}" for a, b, c, x in prog] or ["- 沒有"]
    open(os.path.join(OUT, "彙整.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    # ---- 圖
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = [("RTX 4050 Laptop 6 GB (paper)", 0.16, 0.26 / 1.83, 0.31 / 1.83, "40")]   # 論文的閒置測試，和 40 系列放在一起
    for x in gpu:
        nm = x["裝置"].replace("GB", " GB") + (f" ({x['設定']})" if x["設定"] else "")
        items.append((nm, x["比例"], x["CI_lo"] / x["b"], x["CI_hi"] / x["b"], x["世代"]))
    items.sort(key=lambda i: (i[4], -i[1]))
    for x in m1:
        items.append(("Apple M1 (fanless)", x["比例"], x["CI_lo"] / x["b"], x["CI_hi"] / x["b"], "Apple"))
    items.append((f"CPUs (n={len(cpu)}), max", max(x["比例"] for x in cpu), 0, 0, "CPU"))
    col = {"20": "#1f77b4", "30": "#ff7f0e", "40": "#d62728", "50": "#2ca02c", "Apple": "#7f7f7f", "CPU": "#9467bd"}
    fig, ax = plt.subplots(figsize=(3.5, 0.22 * len(items) + 0.8), dpi=300)
    y = np.arange(len(items))[::-1]
    for yi, (nm, v, lo_, hi_, gen) in zip(y, items):
        ax.barh(yi, 100 * v, color=col[gen], height=0.6)
        if hi_ > lo_:
            ax.errorbar(100 * v, yi, xerr=[[100 * (v - lo_)], [100 * (hi_ - v)]], fmt="none", ecolor="k", elinewidth=0.6, capsize=1.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col[k], label=lab) for k, lab in (("20", "RTX 20 series"), ("30", "RTX 30 series"), ("40", "RTX 40 series"),
                                                                    ("50", "RTX 50 series"))],
              fontsize=5, frameon=False, loc="lower right")
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=5.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Idle penalty (% of back-to-back service time)", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(axis="x", alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "penalty_by_device.png"))
    fig.savefig(os.path.join(OUT, "penalty_by_device.pdf"))
    print(f"→ {OUT}：all_results.csv（{len(table)} 列）、彙整.md、penalty_by_device.png／.pdf")


if __name__ == "__main__":
    main()
