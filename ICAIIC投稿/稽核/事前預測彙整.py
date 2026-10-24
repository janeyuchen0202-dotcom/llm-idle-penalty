"""全部事前預測的彙整（2026-10-24）：33 組排隊預測，每組的預測、範圍、實測、誤差，和當時一起登記的對照公式

給論文 V-B／V-C 用：總平均誤差、命中數、符號檢定（模型比對照公式接近實測的組數）、Bland–Altman（誤差 % 的平均偏差和 95% 一致性界限）
- 實測：照做完的順序重建等待（和 recompute.py 的 run() 相同），丟掉前 1/3
- 預測和範圍：當時寫下的數字（事前預期.md、plan_*.json、志願者的 預測_*.json），不重算
- 對照公式：每組當時一起登記的那個。RTX 4050 3B（seed 1、seed 2）是名目 λ、平均服務時間的 P-K；
  其他是「天真 P-K」（名目 λ、單獨送的請求的服務時間）。算不出來（ρ ≥ 1）的當作無限大
- 誤差 =（預測 − 實測）÷ 實測
- 輸出：稽核/事前預測彙整.csv（之後畫散佈圖用）和螢幕上的摘要

用法：在 ICAIIC投稿 資料夾執行  python 稽核/事前預測彙整.py
"""
import csv
import glob
import io
import json
import math
import os
import sys
import zipfile

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from recompute import EXP, run, wq   # noqa: E402

FRIENDS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "朋友測試包", "朋友結果")
OUT_CSV = os.path.join(HERE, "事前預測彙整.csv")

# RTX 4050 3B seed 1 的後 5 組（10/6 00:29 寫在 事前預期.md）：(P-K, 預測, 下限, 上限)；和 recompute.py 相同
SEED1 = {"rep_rho30": (0.388, 0.327, 0.30, 0.37), "rep_rho50": (0.880, 0.705, 0.63, 0.80),
         "var2_A_rho50": (0.919, 0.723, 0.68, 0.77), "var2_A_rho75": (2.673, 1.675, 1.54, 1.80),
         "var2_C_rho50": (1.109, 0.874, 0.81, 0.95)}
PLANS = [("plan_seed2.json", "RTX 4050", "3B seed 2", "pk"),
         ("plan_7b.json", "RTX 4050", "7B 分開放", "naive_pk"),
         ("plan_7b_v2.json", "RTX 4050", "7B 全放顯示卡", "naive_pk"),
         ("plan_m1.json", "Apple M1", "M1", "naive_pk"),
         ("plan_win.json", "HP MX250", "HP", "naive_pk")]
QUEUE = [("04_RTX2060_5800X_20261018/排隊測試_20261020", "預測_2*.json", "RTX 2060（第 4 台）"),
         ("06_RTX2070S_5800X_20261019/排隊測試_20261021", "預測_2*.json", "RTX 2070S（第 6 台，10/21）"),
         ("06_RTX2070S_5800X_20261019/排隊測試_20261024", "預測_方法1_*.json", "RTX 2070S（第 6 台，10/24 v2b）"),
         ("09_RTX3060Ti_i5-12400_20261020/排隊測試_20261024", "預測_方法1_*.json", "RTX 3060 Ti（第 9 台）")]


def base_value(x):
    return math.inf if x is None or (isinstance(x, float) and (math.isinf(x) or x > 1e6)) else float(x)


def queue_wait(zp, tag):
    """志願者排隊測試的實測平均等待（照做完的順序重建，丟掉前 1/3；和 analyze_queue.py compare 相同）"""
    z = zipfile.ZipFile(zp)
    q = sorted(csv.DictReader(io.StringIO(z.read(f"queue_{tag}.csv").decode("utf-8"))), key=lambda r: int(r["req_id"]))
    a = np.array([float(r["t_send"]) for r in q])
    c = np.array([float(r["t_done"]) for r in q])
    prev, cp = np.empty(len(c)), -np.inf
    for j in np.argsort(c, kind="stable"):
        prev[j], cp = cp, c[j]
    w = np.maximum(a, prev) - a
    return w[len(q) // 3:].mean()


def collect():
    rows = []
    for tag, (pk, pred, lo, hi) in SEED1.items():
        rows.append(dict(device="RTX 4050", group="3B seed 1", tag=tag, pred=pred, lo=lo, hi=hi, base=pk,
                         base_kind="P-K（平均服務時間）", meas=wq(run(tag))))
    for plan, dev, grp, key in PLANS:
        p = json.load(open(os.path.join(EXP, plan), encoding="utf-8"))
        for r in p["runs"]:
            rows.append(dict(device=dev, group=grp, tag=r["tag"], pred=r["pred"], lo=r["lo"], hi=r["hi"],
                             base=base_value(r.get(key)),
                             base_kind="P-K（平均服務時間）" if key == "pk" else "天真 P-K（單獨請求）",
                             meas=wq(run(r["tag"]))))
    for d, pat, dev in QUEUE:
        p = json.load(open(glob.glob(os.path.join(FRIENDS, d, pat))[0], encoding="utf-8"))
        zp = glob.glob(os.path.join(FRIENDS, d, "*.zip"))[0]
        for r in p["runs"]:
            rows.append(dict(device=dev, group=dev, tag=r["tag"], pred=r["pred"], lo=r["lo"], hi=r["hi"],
                             base=base_value(r.get("naive_pk")), base_kind="天真 P-K（單獨請求）",
                             meas=queue_wait(zp, r["tag"])))
    for r in rows:
        r["err"] = (r["pred"] - r["meas"]) / r["meas"]
        r["hit"] = r["lo"] <= r["meas"] <= r["hi"]
        r["base_err"] = (r["base"] - r["meas"]) / r["meas"]
        r["closer"] = abs(r["pred"] - r["meas"]) < abs(r["base"] - r["meas"])
    return rows


def summary(rows, label):
    e = np.array([r["err"] for r in rows])
    be = np.array([r["base_err"] for r in rows if math.isfinite(r["base_err"])])
    k = sum(r["closer"] for r in rows)
    p = stats.binomtest(k, len(rows), 0.5).pvalue
    bias, sd = e.mean(), e.std(ddof=1)
    print(f"{label}（{len(rows)} 組）：命中 {sum(r['hit'] for r in rows)}｜誤差絕對值 平均 {np.abs(e).mean():.1%}、中位數 "
          f"{np.median(np.abs(e)):.1%}｜≤5% 的 {int(np.sum(np.abs(e) <= 0.05))} 組、≤10% 的 {int(np.sum(np.abs(e) <= 0.10))} 組｜"
          f"Bland–Altman 平均偏差 {bias:+.1%}、95% 界限 {bias - 1.96 * sd:+.1%} 到 {bias + 1.96 * sd:+.1%}｜"
          f"比對照公式接近 {k}/{len(rows)}（符號檢定 p = {p:.1g}）｜對照公式誤差絕對值平均 {np.abs(be).mean():.0%}（{len(be)} 組有限值）")


def main():
    rows = collect()
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['device']:<28}{r['tag']:<14}預測 {r['pred']:.3f}（{r['lo']:.3f}–{r['hi']:.3f}）實測 {r['meas']:.3f}｜"
              f"誤差 {r['err']:+.1%} {'✅' if r['hit'] else '❌'}｜對照 {r['base']:.3f}（{r['base_err']:+.0%}）{'模型較近' if r['closer'] else '對照較近'}")
    print()
    summary(rows, "全部")
    summary([r for r in rows if r["device"] == "RTX 4050"], "RTX 4050")
    summary([r for r in rows if r["device"] != "RTX 4050"], "其他機器")
    heat = [r for r in rows if r["device"].startswith(("RTX 20", "RTX 30", "HP"))]
    summary(heat, "會變熱的機器（HP、第 4、6、9 台）")
    summary([r for r in rows if r not in heat], "其他（RTX 4050、M1）")
    print(f"→ {OUT_CSV}")


if __name__ == "__main__":
    main()
