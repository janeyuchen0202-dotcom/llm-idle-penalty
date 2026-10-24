"""論文數字稽核（2026-10-09）：從原始 CSV 重算 paper_draft.md 裡的數字，和論文寫的並排比對

怎麼用：在 ICAIIC投稿 資料夾執行  python 稽核/recompute.py      （約 1–2 分鐘）
輸出：螢幕上的對照表，同時寫到 稽核/重算結果.md

原則
- 實測值一律從 LLM延遲實驗/out/ 的原始 CSV 重算（每筆請求的送出、完成時間），不讀分析程式的輸出
- 要模擬的數字（第 V-B 節的兩個模型、第 VI 節的卸載門檻）會重新執行原本的程式
  （analyze_strengthen.py、make_offload_v2.py，亂數種子固定，所以結果應該和論文完全一樣）
- 事前預測（plan_*.json、事前預期.md 裡的表）是當時寫下的數字，這裡只重算實測值，再判斷有沒有落在範圍內
- 判定：論文寫的和重算值四捨五入到論文的位數後一致 → ✅；不一致或要人工看 → ⚠️（會寫原因）

每一列的「來源」就是你要抽查的原始檔：用 Excel 打開，挑幾筆照公式 (1) 手算，看是不是同一個數。
"""
import csv
import json
import os
import re
import statistics
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
EXP = os.path.join(os.path.dirname(PAPER_DIR), "LLM延遲實驗")
OUT = os.path.join(EXP, "out")
NL = chr(10)
RESULTS = []
HITS = {}          # 事前預測：組別 → (落在範圍內的組數, 總組數)


# ------------------------------------------------------------------ 共用
OVER = {}   # 單獨請求比一個接一個的服務時間高多少（摘要的 12%–57%）


def add(sec, what, paper, got, src, ok=None, note=""):
    """paper、got 都是字串（已照論文的位數格式化）；ok 沒給就比對字串是否相同"""
    if ok is None:
        ok = paper == got
    RESULTS.append(dict(sec=sec, what=what, paper=paper, got=got, ok=ok, src=src, note=note))


def rows_of(name):
    return list(csv.DictReader(open(os.path.join(OUT, name + ".csv"), encoding="utf-8")))


def run(name):
    """公式 (1)：s_i = max(a_i, c_{i-1})，W_i = s_i − a_i，S_i = c_i − s_i（10/22 起 c_{i-1} 是照做完的順序、前一個被處理的請求做完的時間）"""
    r = sorted([x for x in rows_of(name) if x["status"] == "ok"], key=lambda x: float(x["t_send"]))
    a = np.array([float(x["t_send"]) for x in r])
    c = np.array([float(x["t_done"]) for x in r])
    prev = np.empty(len(c)); _cp = -np.inf    # 10/22 改：照做完的順序重建（一次只處理一個請求，做完的順序就是處理的順序；原本照送出順序）
    for _j in np.argsort(c, kind="stable"):
        prev[_j] = _cp; _cp = c[_j]
    s = np.maximum(a, prev)
    tok = np.array([int(x["out_tokens"]) for x in r])
    return dict(a=a, W=s - a, S=c - s, gap=a - prev, tok=tok, rows=r, n=len(r))


def win(d):
    k = d["n"] // 3
    return d["a"][k:], d["W"][k:], d["S"][k:]


def wq(d):
    return win(d)[1].mean()


def lam_hat(d):
    aw = win(d)[0]
    return (len(aw) - 1) / (aw[-1] - aw[0])


def pk(lam, es, es2):
    return lam * es2 / (2 * (1 - lam * es)) if lam * es < 1 else float("inf")


def probe(name):
    return list(csv.DictReader(open(os.path.join(OUT, name + ".csv"), encoding="utf-8")))


def probe_extra(name, cond="A", gmin=3.0):
    """閒置 g ≥ gmin 的請求比 g = 0 多出的服務時間（平均與 Welch 95% CI）"""
    from scipy import stats
    r = [x for x in probe(name) if x["cond"] == cond]
    b0 = np.array([float(x["service"]) for x in r if float(x["gap"]) == 0])
    x = np.array([float(x["service"]) for x in r if float(x["gap"]) >= gmin])
    d = x.mean() - b0.mean()
    v1, v2 = x.var(ddof=1) / len(x), b0.var(ddof=1) / len(b0)
    se = np.sqrt(v1 + v2)
    t = stats.t.ppf(0.975, se ** 4 / (v1 ** 2 / (len(x) - 1) + v2 ** 2 / (len(b0) - 1)))
    return b0.mean(), d, d - t * se, d + t * se


def per_gap(name, cond="A", key="service"):
    r = [x for x in probe(name) if x["cond"] == cond]
    gaps = sorted({float(x["gap"]) for x in r})
    b0 = np.mean([float(x[key]) for x in r if float(x["gap"]) == 0])
    return {g: np.mean([float(x[key]) for x in r if float(x["gap"]) == g]) - b0 for g in gaps}


def f(x, nd):
    return f"{x:.{nd}f}"


A_RUNS = ["main2_rho30", "main2_rho50", "long70_rho70", "long85_rho85", "mix_A_rho50", "mix_A_rho75",
          "rep_rho30", "rep_rho50", "rep_rho70", "rep_rho85", "var2_A_rho50", "var2_A_rho75"]
C_RUNS = ["var_C_rho50", "var_C_rho70", "var2_C_rho50", "var2_C_rho70", "confirm_C_rho75"]
NOMINAL = {"main2_rho30": 0.14995, "main2_rho50": 0.24991, "long70_rho70": 0.34988, "long85_rho85": 0.42485,
           "mix_A_rho50": 0.259, "mix_A_rho75": 0.389, "var_C_rho50": 0.233, "var_C_rho70": 0.330,
           "rep_rho30": 0.14995, "rep_rho50": 0.24991, "rep_rho70": 0.34988, "rep_rho85": 0.42485,
           "var2_A_rho50": 0.259, "var2_A_rho75": 0.389, "var2_C_rho50": 0.233, "var2_C_rho70": 0.330,
           "confirm_C_rho75": 0.35}
D = {n: run(n) for n in A_RUNS + C_RUNS}


# ------------------------------------------------------------------ III-A 測試環境
def sec_testbed():
    t = open(os.path.join(EXP, "prompts", "token_counts.txt"), encoding="utf-8").read()
    got = dict(re.findall(r"(\w+)\.txt\s+目標 \d+ tokens\s+實際 (\d+) tokens", t))
    add("III-A", "prompt 長度（medium／short／long）", "838／118／1,478",
        f"{got['medium']}／{got['short']}／{int(got['long']):,}", "LLM延遲實驗/prompts/token_counts.txt")
    for name, paper in (("mix_A_rho50", "0.09"), ("var_C_rho50", "0.67")):
        S = win(run(name))[2]
        add("III-A", f"服務時間的變異係數（{name}，分析區間）", paper, f(S.std() / S.mean(), 2), f"out/{name}.csv")


# ------------------------------------------------------------------ III-E 假伺服器驗證
def sec_validation():
    for name, rho, paper_w, paper_th, lam in (("my_rho50", "0.5", "0.261", "0.271", 1.0),
                                               ("my_rho80", "0.8", "1.037", "1.083", 1.6)):
        d = run(name)
        th = lam * 0.2708 / (2 * (1 - lam * 0.5))          # U(0.25, 0.75)：E[S]=0.5、E[S²]=0.2708
        add("III-E", f"假伺服器 ρ={rho}：公式 (1) 重建、丟前 20 筆的平均等待（{d['n'] - 20} 筆）", paper_w,
            f(d["W"][20:].mean(), 3), f"out/{name}.csv")
        add("III-E", f"假伺服器 ρ={rho}：P-K 理論值", paper_th, f(th, 3), "U(0.25, 0.75) 的動差")
    pen = []
    for name in ("my_rho50", "my_rho80"):
        d = run(name)
        idle, q = d["W"] == 0, d["W"] > 0
        idle[0] = False
        pen.append(d["S"][idle].mean() - d["S"][q].mean())
    add("III-E", "假伺服器的閒置後多出時間（兩組）", "0.01–0.03", f"{min(pen):.2f}–{max(pen):.2f}",
        "out/my_rho50.csv、my_rho80.csv")


# ------------------------------------------------------------------ V-A 閒置效應
def sec_idle():
    probes = ["probe_idle_3b", "probe_idle_3b_nolog", "probe_idle_3b_heavy"]
    pg = {p: per_gap(p) for p in probes}
    small = max(abs(v) for p in probes for g, v in pg[p].items() if 0 < g <= 1.5)
    add("V-A", "3B：g ≤ 1.5 秒時沒有額外時間（三次測試的最大絕對值）", "沒有（約 0）", f"最大 {small:.3f} 秒",
        "out/probe_idle_3b*.csv", ok=small < 0.05)
    g2 = [pg[p][2.0] for p in probes if 2.0 in pg[p]]          # nolog 那次沒有測 g = 2
    add("V-A", "3B：g = 2 秒的額外時間（有測 g=2 的兩次測試）", "0.17–0.20", f"{min(g2):.2f}–{max(g2):.2f}", "同上")
    big = [v for p in probes for g, v in pg[p].items() if g >= 3]
    avg = np.mean([np.mean([v for g, v in pg[p].items() if g >= 3]) for p in probes])
    add("V-A", "3B：g ≥ 3 秒的平均額外時間", "0.29", f(avg, 2), "同上")
    add("V-A", "3B：g ≥ 3 秒各閒置時間、各次測試的範圍", "0.22–0.36", f"{min(big):.2f}–{max(big):.2f}", "同上")
    tt = [per_gap(p, key="ttft") for p in probes]
    share = np.mean([tt[i][g] for i in range(3) for g in pg[probes[i]] if g >= 3]) / np.mean(big)
    add("V-A", "額外時間落在第一個 token 之前的比例", "幾乎全部", f"{100 * share:.0f}%", "同上", ok=share > 0.85)
    # 顯示卡時脈（probe_idle_3b 的 nvidia-smi 紀錄）
    from datetime import datetime
    g = []
    for line in open(os.path.join(OUT, "probe_idle_3b_gpu.csv"), encoding="utf-8", errors="replace"):
        p = [x.strip() for x in line.split(",")]
        try:
            g.append((datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp(), p[1], float(p[2])))
        except (ValueError, IndexError):
            continue
    g.sort()
    gt, gp, gc = np.array([x[0] for x in g]), np.array([x[1] for x in g]), np.array([x[2] for x in g])
    rows = probe("probe_idle_3b")

    def clk(cond, gap):
        sel = [r for r in rows if r["cond"] == cond and float(r["gap"]) == gap]
        return np.mean([gc[max(0, np.searchsorted(gt, float(r["wall_send"])) - 1)] for r in sel])
    add("V-A", "送出時的時脈：g ≤ 1.5 秒（A）", "約 2,550 MHz", f"{np.mean([clk('A', x) for x in (0, 0.5, 1, 1.5)]):,.0f} MHz",
        "out/probe_idle_3b_gpu.csv", ok=abs(np.mean([clk('A', x) for x in (0, 0.5, 1, 1.5)]) - 2550) < 100)
    add("V-A", "送出時的時脈：g = 3 秒（A）", "約 400 MHz", f"{clk('A', 3):,.0f} MHz", "同上", ok=abs(clk("A", 3) - 400) < 60)
    drops = []
    for r in rows:
        if r["cond"] != "A" or float(r["gap"]) < 3:
            continue
        t_end = float(r["wall_send"]) - float(r["idle_before"])
        j = np.searchsorted(gt, t_end)
        k = next((i for i in range(j, len(gt)) if gt[i] < float(r["wall_send"]) and gp[i] == "P8"), None)
        if k is not None:
            drops.append(gt[k] - t_end)
    add("V-A", "閒置後進入 P8 的時間（中位數）", "3.6 秒", f"{np.median(drops):.1f} 秒", "同上")
    kc = [clk("K", x) for x in (3, 5, 8)]
    add("V-A", "1 token 保溫：g ≥ 3 秒送出時的時脈", "1,000–1,300 MHz", f"{min(kc):,.0f}–{max(kc):,.0f} MHz", "同上",
        ok=min(kc) > 900 and max(kc) < 1400)
    ka = np.mean([v for gg, v in per_gap("probe_idle_3b", "K").items() if gg >= 3])
    aa = np.mean([v for gg, v in pg["probe_idle_3b"].items() if gg >= 3])
    add("V-A", "1 token 保溫：g ≥ 3 秒的平均額外時間（不保溫 → 輕保溫）", "0.27 → 0.15", f"{aa:.2f} → {ka:.2f}", "out/probe_idle_3b.csv")
    hv = per_gap("probe_idle_3b_heavy", "H")
    hbig = np.mean([v for gg, v in hv.items() if gg >= 3])
    add("V-A", "重保溫：g ≥ 3 秒的平均額外時間", "−0.007", f"{hbig:+.3f}".replace("-", "−"), "out/probe_idle_3b_heavy.csv")
    # 7B
    p7 = per_gap("probe_idle_7b16")
    add("V-A", "7B 分開放：g = 5 秒多出的時間（16 token 測試，和表 I 同一組）", "0.81", f(per_gap("probe_idle_7b16")[5.0], 2),
        "out/probe_idle_7b16.csv（16 tokens）")
    b, d, lo, hi = probe_extra("probe_b7v2")
    add("V-A", "7B 全放顯示卡：g ≥ 3 秒的平均額外時間", "0.58", f(d, 2), "out/probe_b7v2.csv")
    # 17 組排隊實驗裡的閒置效應
    gaps, res, pools = [], [], {}
    for name in A_RUNS + C_RUNS:
        dd = D[name]
        for key in sorted(set(dd["tok"])):
            q = (dd["W"] > 0) & (dd["tok"] == key)
            pools.setdefault(key, []).extend(dd["S"][q])
            idle = (dd["W"] == 0) & (dd["tok"] == key)
            idle[0] = False
            gaps.extend(dd["gap"][idle])
            res.extend(dd["S"][idle] - dd["S"][q].mean())
    gaps, res = np.array(gaps), np.array(res)
    for lab, m, pp, pn in (("< 1 秒", gaps < 1, "0.04", "715"), ("1–2 秒", (gaps >= 1) & (gaps < 2), "0.09", "505"),
                           ("≥ 2 秒", gaps >= 2, "0.34", "1,549")):
        add("V-A", f"排隊實驗：閒置 {lab} 後的額外時間（筆數）", f"{pp}（{pn}）", f"{res[m].mean():.2f}（{m.sum():,}）",
            "17 組 out/*.csv")
    add("V-A", "排隊實驗：一個接一個做的服務時間（A）", "1.80", f(np.mean(pools[128]), 2), "同上")
    # 單獨請求的校正（公式 (1) 重建的 S）
    ca = run("mix_cal_A")
    cc = run("var_cal_C")
    iso_c = (cc["S"][cc["tok"] == 32].mean() + cc["S"][cc["tok"] == 224].mean()) / 2
    btb_c = (np.mean(pools[32]) + np.mean(pools[224])) / 2
    add("V-A", "單獨請求的服務時間（A）", "2.01", f(ca["S"].mean(), 2), "out/mix_cal_A.csv")
    add("V-A", "單獨請求高估一個接一個的服務時間（A／C）", "12%／13%",
        f"{ca['S'].mean() / np.mean(pools[128]) - 1:.0%}／{iso_c / btb_c - 1:.0%}", "out/mix_cal_A.csv、var_cal_C.csv")
    OVER["3B A"], OVER["3B C"] = ca["S"].mean() / np.mean(pools[128]) - 1, iso_c / btb_c - 1


# ------------------------------------------------------------------ V-B 等待時間（重跑 analyze_strengthen.py）
def sec_waiting():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    txt = subprocess.run([sys.executable, os.path.join(EXP, "analyze_strengthen.py")], cwd=EXP, env=env,
                         capture_output=True, text=True, encoding="utf-8").stdout
    m = re.search(r"一般 M/G/1 (\d+)/17，閒置效應模型 (\d+)/17", txt)
    add("V-B", "落在 95% 範圍內：一般 M/G/1／閒置效應模型", "15／17", f"{m.group(1)}／{m.group(2)}", "重跑 analyze_strengthen.py")
    m = re.search(r"一般 M/G/1（同到達時間）(\d+)%，閒置效應模型 (\d+)%", txt)
    add("V-B", "平均絕對誤差：一般 M/G/1／閒置效應模型", "7%／3%", f"{m.group(1)}%／{m.group(2)}%", "同上")
    # III-D 暖機敏感度：改丟前 1/4、1/2（WARM_DIV=4、2；不會覆蓋畫圖用的檔案）
    sens = {}
    for dv in (4, 2):
        t2 = subprocess.run([sys.executable, os.path.join(EXP, "analyze_strengthen.py")], cwd=EXP,
                            env=dict(env, WARM_DIV=str(dv)), capture_output=True, text=True, encoding="utf-8").stdout
        h = re.search(r"一般 M/G/1 (\d+)/17，閒置效應模型 (\d+)/17", t2)
        e = re.search(r"一般 M/G/1（同到達時間）(\d+)%，閒置效應模型 (\d+)%", t2)
        sens[dv] = dict(iid_in=int(h.group(1)), mod_in=int(h.group(2)), iid_err=int(e.group(1)), mod_err=int(e.group(2)))
    rng = lambda key, base: f"{min([sens[4][key], sens[2][key]])}–{max([sens[4][key], sens[2][key]])}"         if sens[4][key] != sens[2][key] else str(sens[4][key])
    add("III-D", "改丟前 1/4 或 1/2：考慮準備時間的模型／標準模型的平均絕對誤差", "3%／7–8%",
        f"{rng('mod_err', 0)}%／{rng('iid_err', 0)}%", "重跑 analyze_strengthen.py（WARM_DIV=4、2）")
    add("III-D", "改丟前 1/4 或 1/2：落在 95% 範圍內的組數（考慮準備時間／標準）", "16–17／15–17",
        f"{rng('mod_in', 0)}／{rng('iid_in', 0)}", "同上")
    summ = {r["run"]: r for r in csv.DictReader(open(os.path.join(OUT, "strengthen_summary.csv"), encoding="utf-8"))}
    # 「實測比預測低幾 %」＝ 1 − 實測／預測（和 V-C 的寫法相同）
    over = [1 - float(summ[n]["W"]) / float(summ[n]["iid_mean"]) for n in A_RUNS]
    add("V-B", "一般 M/G/1 高估了 12 組 A 中的幾組；實測平均比預測低", "12／12，6%", f"{sum(o > 0 for o in over)}／12，{np.mean(over):.0%}", "同上")
    mod = [1 - float(r["W"]) / float(r["mod_mean"]) for r in summ.values()]
    add("V-B", "閒置效應模型：實測平均比預測低", "2%", f"{np.mean(mod):.0%}", "同上")
    dev = [wq(D[n]) / pk(NOMINAL[n], win(D[n])[2].mean(), (win(D[n])[2] ** 2).mean()) - 1 for n in A_RUNS + C_RUNS]
    add("V-B", "單次實驗和名目 P-K 的偏差範圍", "−39% 到 +28%", f"{min(dev):.0%} 到 {max(dev):+.0%}".replace("-", "−"), "17 組 out/*.csv")
    util = [lam_hat(D[n]) * win(D[n])[2].mean() for n in A_RUNS + C_RUNS]
    add("摘要", "17 組的利用率範圍（λ̂E[S]）", "0.26–0.80", f"{min(util):.2f}–{max(util):.2f}", "17 組 out/*.csv")
    lr = [lam_hat(D[n]) / NOMINAL[n] for n in A_RUNS + C_RUNS]
    add("V-B", "實際到達率／名目到達率", "0.89–1.16", f"{min(lr):.2f}–{max(lr):.2f}", "同上")
    add("V-B", "ρ≈0.8 時到達率少 5%，等待少多少（P-K 公式）", "約 20%", f"{1 - 0.95 * 0.2 / (1 - 0.8 * 0.95):.0%}", "公式",
        ok=True)
    add("V-B", "第一次實驗兩個低負載點的到達率偏高", "16%", f"{lam_hat(D['main2_rho30']) / NOMINAL['main2_rho30'] - 1:.0%}",
        "out/main2_rho30.csv、main2_rho50.csv")
    return summ


# ------------------------------------------------------------------ V-C 事前預測
def sec_predictions():
    exp = open(os.path.join(EXP, "事前預期.md"), encoding="utf-8").read()
    # (1) 3B seed 1 後 5 組（10/6 00:29 寫下；數字照 事前預期.md）
    mid = {"rep_rho30": (0.388, 0.327, 0.30, 0.37), "rep_rho50": (0.880, 0.705, 0.63, 0.80),
           "var2_A_rho50": (0.919, 0.723, 0.68, 0.77), "var2_A_rho75": (2.673, 1.675, 1.54, 1.80),
           "var2_C_rho50": (1.109, 0.874, 0.81, 0.95)}
    ins = [lo <= wq(D[n]) <= hi for n, (_, _, lo, hi) in mid.items()]
    HITS["seed1"] = (sum(ins), len(ins))
    below = mid["var2_C_rho50"][2] - wq(D["var2_C_rho50"])
    near = [abs(wq(D[n]) - p) < abs(wq(D[n]) - k) for n, (k, p, _, _) in mid.items()]
    bias = np.mean([wq(D[n]) / p - 1 for n, (_, p, _, _) in mid.items()])
    add("V-C", "3B seed 1：5 組中幾組在範圍內／落空那組低多少", "4／0.008 秒", f"{sum(ins)}／{below:.3f} 秒",
        "事前預期.md（10/6 00:29）＋ out/*.csv")
    add("V-C", "3B seed 1：比較接近模型的組數／平均偏差", "5／−5%", f"{sum(near)}／{bias:.0%}".replace("-", "−"), "同上")
    # (2)–(4) plan JSON
    for plan, label, paper_hits, paper_meas, paper_pred in (
            ("plan_7b.json", "7B（分開放，10/6）", "3／3", "0.95／1.72／3.48", "0.94／1.73／3.65"),
            ("plan_seed2.json", "3B seed 2", "3／4", None, None),
            ("plan_7b_v2.json", "7B 全放顯示卡（10/8）", "3／3", "0.34／0.77／1.46", "0.33／0.77／1.52"),
            ("plan_m1.json", "M1", "2／3", None, None),
            ("plan_win.json", "HP 筆電（MX250）", "1／3", None, None)):
        p = json.load(open(os.path.join(EXP, plan), encoding="utf-8"))
        meas = [wq(run(r["tag"])) for r in p["runs"]]
        hits = sum(r["lo"] <= m <= r["hi"] for r, m in zip(p["runs"], meas))
        HITS[plan] = (hits, len(meas))
        add("V-C／V-D", f"{label}：落在範圍內", paper_hits, f"{hits}／{len(meas)}", f"LLM延遲實驗/{plan} ＋ out/*.csv")
        order = sorted(range(len(meas)), key=lambda i: p["runs"][i]["rate"])
        if paper_meas:
            add("V-C", f"{label}：實測（ρ 由低到高）", paper_meas, "／".join(f(meas[i], 2) for i in order), "out/*.csv")
            add("V-C", f"{label}：當時寫下的預測", paper_pred, "／".join(f(p["runs"][i]["pred"], 2) for i in order), plan)
    p7 = json.load(open(os.path.join(EXP, "plan_7b.json"), encoding="utf-8"))
    add("V-C", "7B（分開放）：天真 P-K（單獨請求 5.06 秒）", "1.41／3.72／12.69",
        "／".join(f(r["naive_pk"], 2) for r in sorted(p7["runs"], key=lambda r: r["rate"])), "plan_7b.json")
    worst = max(r["naive_pk"] / wq(run(r["tag"])) for r in p7["runs"])
    add("V-C", "7B（分開放）：天真 P-K 最多是實測的幾倍", "3.6", f(worst, 1), "同上")
    pv = json.load(open(os.path.join(EXP, "plan_7b_v2.json"), encoding="utf-8"))
    r128 = [x for x in probe("probe_idle_7b") if x["cond"] == "A"]
    b128 = np.mean([float(x["service"]) for x in r128 if float(x["gap"]) == 0])
    d5 = np.mean([float(x["service"]) for x in r128 if float(x["gap"]) > 0 and float(x["idle_before"]) >= 5]) - b128
    r16 = [x for x in probe("probe_idle_7b16") if x["cond"] == "A"]
    b16 = np.mean([float(x["service"]) for x in r16 if float(x["gap"]) == 0])
    iso16 = np.mean([float(x["service"]) for x in r16 if float(x["gap"]) == 5])
    add("V-C", "7B 分開放：單獨請求（g = 5 秒）／一個接一個的服務時間", "5.06／4.24", f"{iso16:.2f}／{b16:.2f}", "out/probe_idle_7b16.csv")
    add("V-C", "7B 分開放：預測用的閒置效應（128 token 測試、實際閒置 ≥ 5 秒；佔 b 的 %）", "0.84（20%）",
        f"{d5:.2f}（{d5 / b16:.0%}）", "out/probe_idle_7b.csv、probe_idle_7b16.csv（predict_7b.py 的分段）")
    OVER["7B 分開放"] = iso16 / b16 - 1
    add("V-C", "7B 全放顯示卡：一個接一個／單獨請求的服務時間", "1.20／1.88", f"{pv['b']:.2f}／{pv['iso']:.2f}", "plan_7b_v2.json（來自 out/probe_b7v2.csv）")
    OVER["7B 全放顯示卡"] = pv["iso"] / pv["b"] - 1
    add("V-C", "單獨請求比一個接一個的服務時間高幾 %（7B 分開放／7B 全放顯示卡）", "19%／57%",
        f"{OVER['7B 分開放']:.0%}／{OVER['7B 全放顯示卡']:.0%}", "同上")
    add("摘要", "RTX 4050 上單獨請求高估負載下的服務時間（3B A／3B C／7B 兩種放法）：範圍", "12% to 57%",
        f"{min(OVER.values()):.0%} to {max(OVER.values()):.0%}", "V-A、V-C 四個值")
    nk = sorted(pv["runs"], key=lambda r: r["rate"])
    add("V-C", "7B 全放顯示卡：天真 P-K（兩個低負載點；最高負載）", "0.86／3.52；不穩定",
        f"{nk[0]['naive_pk']:.2f}／{nk[1]['naive_pk']:.2f}；{'不穩定' if nk[2]['naive_pk'] in (float('inf'), None) or nk[2]['naive_pk'] > 1e6 else nk[2]['naive_pk']}",
        "同上")
    s2 = json.load(open(os.path.join(EXP, "plan_seed2.json"), encoding="utf-8"))
    miss = [r["lo"] - wq(run(r["tag"])) for r in s2["runs"] if not r["lo"] <= wq(run(r["tag"])) <= r["hi"]]
    add("V-C", "3B seed 2：落空那組比下限低多少", "0.005 秒", f"{miss[0]:.3f} 秒" if miss else "沒有落空", "plan_seed2.json ＋ out/rep2_*.csv")
    btb = [run(r["tag"]) for r in s2["runs"]]
    vals = [x["S"][x["W"] > 0].mean() for x in btb]
    # 9 組 3B 事前預測（seed1 5 組＋seed2 4 組）：實測平均比預測低多少
    preds = {n: p for n, (_, p, _, _) in mid.items()}
    preds.update({r["tag"]: r["pred"] for r in s2["runs"]})
    bias9 = np.mean([1 - wq(run(t)) / p for t, p in preds.items()])
    add("V-C", "9 組 out-of-sample 3B：實測平均比預測低", "5%", f"{bias9:.0%}", "事前預期.md、plan_seed2.json ＋ out/*.csv")
    # 漂移分析：8 組 A 用最終模型重新模擬（照 10/6 的做法，亂數種子 23）
    rng23 = np.random.default_rng(23)
    dv = np.loadtxt(os.path.join(OUT, "idle_penalty_bins.csv"), delimiter=",", skiprows=1)[:, 0]
    pool = np.concatenate([D[t]["S"][D[t]["W"] > 0] for t in A_RUNS])

    def resim(a, bp):
        n, k = len(a), len(a) // 3
        base = rng23.choice(bp, size=(2000, n))
        c_prev, acc = np.full(2000, -np.inf), np.zeros(2000)
        for i in range(n):
            st = np.maximum(a[i], c_prev)
            w = st - a[i]
            pen = np.where(w == 0, dv[np.digitize(a[i] - c_prev, [1.0, 2.0, 5.0])], 0.0)
            c_prev = st + base[:, i] + pen
            if i >= k:
                acc += w
        return acc.mean() / (n - k)
    r1, r2 = [], []
    for t in ["rep_rho30", "rep_rho50", "var2_A_rho50", "var2_A_rho75", "rep2_rho30", "rep2_rho50", "rep2_rho70", "rep2_rho85"]:
        d = run(t)
        own = d["S"][d["W"] > 0]
        r1.append(1 - wq(d) / resim(d["a"], pool))
        r2.append(1 - wq(d) / resim(d["a"], pool - pool.mean() + own.mean()))
    add("V-C", "8 組 A 重新模擬的偏差 → 改用各組自己的服務時間", "4.3% → 3.4%",
        f"{np.mean(r1):.1%} → {np.mean(r2):.1%}", "重新模擬（10/6 的做法，種子 23）")
    rtx = [HITS[k] for k in ("seed1", "plan_7b.json", "plan_seed2.json", "plan_7b_v2.json")]
    add("V-C", "總計：RTX 4050 上事前預測在範圍內", "13／15", f"{sum(h for h, _ in rtx)}／{sum(n for _, n in rtx)}", "上面幾列加總")


# ------------------------------------------------------------------ V-D 其他硬體（表 I）
def sec_hardware():
    # RTX 4050 3B：三次測試各自的差，用反變異數加權合併
    est, n0, n3 = [], 0, 0
    for p in ("probe_idle_3b", "probe_idle_3b_nolog", "probe_idle_3b_heavy"):
        r = [x for x in probe(p) if x["cond"] == "A"]
        s0 = np.array([float(x["service"]) for x in r if float(x["gap"]) == 0])
        s3 = np.array([float(x["service"]) for x in r if float(x["gap"]) >= 3])
        se = np.sqrt(s3.var(ddof=1) / len(s3) + s0.var(ddof=1) / len(s0))
        est.append((s0.mean(), s3.mean() - s0.mean(), se))
        n0, n3 = n0 + len(s0), n3 + len(s3)
    w = np.array([1 / se ** 2 for _, _, se in est])
    dm = sum(wi * d for wi, (_, d, _) in zip(w, est)) / w.sum()
    sem = 1 / np.sqrt(w.sum())
    add("V-D 表 I", "RTX 4050 3B：一個接一個的服務時間（三次測試平均）", "1.82", f(np.mean([b0 for b0, _, _ in est]), 2), "out/probe_idle_3b*.csv")
    add("V-D 表 I", "RTX 4050 3B：閒置 ≥3 秒多出的時間（三次合併，95% CI）", "0.29 (0.26, 0.31)",
        f"{dm:.2f} ({dm - 1.96 * sem:.2f}, {dm + 1.96 * sem:.2f})", "out/probe_idle_3b*.csv")
    add("V-D 表 I", "RTX 4050 3B：筆數（g = 0／g ≥ 3 秒，三次合計）", "32/96", f"{n0}/{n3}", "out/probe_idle_3b*.csv")
    add("摘要", "閒置後多出的時間佔一個接一個服務時間的 %（表 I 第一列 0.29 ÷ 1.82）", "16%",
        f"{dm / np.mean([b0 for b0, _, _ in est]):.0%}", "表 I 第一列")
    PN = {"probe_idle_7b16": "16/8*"}
    global UPPER
    UPPER = []
    LOW3B = dm - 1.96 * sem
    for name, label, pb, pe in (("probe_idle_7b16", "RTX 4050 7B 分開放", "4.24", "0.81 (0.71, 0.92)"),
                                ("probe_b7v2", "RTX 4050 7B 全放顯示卡", "1.20", "0.58 (0.44, 0.73)"),
                                ("probe_m1", "Apple M1", "3.88", "0.01 (−0.05, 0.07)"),
                                ("probe_m4", "Apple M4", "2.47", "0.03 (−0.01, 0.06)"),
                                ("probe_wincpu", "i7-10510U 純 CPU", "4.94", "0.06 (−0.12, 0.24)"),
                                ("probe_wingpu", "MX250", "3.12", "−0.08 (−0.25, 0.09)")):
        b, d, lo, hi = probe_extra(name)
        got = f"{d:.2f} ({lo:.2f}, {hi:.2f})".replace("-", "−")
        add("V-D 表 I", f"{label}：一個接一個的服務時間", pb, f(b, 2), f"out/{name}.csv")
        add("V-D 表 I", f"{label}：閒置 ≥3 秒多出的時間（95% CI）", pe, got, f"out/{name}.csv")
        if name not in ("probe_idle_7b16", "probe_b7v2"):
            UPPER.append(hi)
        r = [x for x in probe(name) if x["cond"] == "A"]
        g3 = sorted({float(x["gap"]) for x in r if float(x["gap"]) >= 3})
        nn = f"{sum(float(x['gap']) == 0 for x in r)}/{sum(float(x['gap']) >= 3 for x in r)}" + ("*" if g3 == [5.0] else "")
        add("V-D 表 I", f"{label}：筆數（g = 0／g ≥ 3 秒；* = 只有 5 秒）", PN.get(name, "12/18"), nn, f"out/{name}.csv")
    add("V-D", "其他四種設定 95% CI 上限的範圍；都低於 RTX 4050 3B 合併 CI 的下限", "0.06–0.24 < 0.26",
        f"{min(UPPER):.2f}–{max(UPPER):.2f} {'<' if max(UPPER) < LOW3B else '≥'} {LOW3B:.2f}", "表 I 各列")
    net = []
    for name in ("netidle_probe_m4_before", "netidle_probe_wincpu_before", "netidle_probe_wingpu_before"):
        r = probe(name)
        base = np.mean([float(x["rtt_ms"]) for x in r if float(x["gap"]) == 0])
        net.append(max(np.mean([float(x["rtt_ms"]) for x in r if float(x["gap"]) == g]) - base
                       for g in {float(x["gap"]) for x in r} if g > 0))
    add("V-D", "Wi-Fi 本身閒置後多出的時間（最多）", "約 0.05 秒", f"{max(net) / 1000:.3f} 秒", "out/netidle_*.csv",
        ok=abs(max(net) / 1000 - 0.05) < 0.01)
    for name, paper in (("probe_m1", "6%"), ("probe_m4", "10%")):
        r = probe(name)
        a = np.mean([float(x["service"]) for x in r if x["cond"] == "A"])
        last = np.mean([float(x["service"]) for x in r if int(x["block"]) == 4])
        add("V-D", f"{name[6:].upper()}：最後一個重保溫區塊比 A 區塊慢", paper, f"{last / a - 1:.0%}", f"out/{name}.csv")
    pw = json.load(open(os.path.join(EXP, "plan_win.json"), encoding="utf-8"))
    vals = []
    for r in sorted(pw["runs"], key=lambda r: r["rate"]):
        d = run(r["tag"])
        vals.append((d["S"][d["W"] > 0].mean(), r, d))
    add("V-D", "HP：排隊後開始的服務時間（ρ≈0.3／0.5／0.7）", "3.07／3.41／3.71", "／".join(f(v[0], 2) for v in vals), "out/win_rho*.csv")
    top = vals[2]
    util = lam_hat(top[2]) * win(top[2])[2].mean()
    add("V-D", "HP：最高負載的利用率（λ̂E[S]，論文 III-D 的定義）", "約 0.92", f(util, 3), "out/win_rho70.csv",
        ok=abs(util - 0.92) < 0.006)
    # 遠端：主筆電記的時間戳，排隊的請求最多多算一個網路來回
    meds, worst = [], 0.0
    for plan in ("plan_m1", "plan_win"):
        for L in csv.DictReader(open(os.path.join(OUT, f"{plan}_runlog.csv"), encoding="utf-8")):
            rtt = max(float(L["rtt_before_med"]), float(L["rtt_after_med"])) / 1000
            meds.append(rtt)
            d = run(L["tag"])
            w = win(d)[1]
            worst = max(worst, (w > 0).mean() * rtt / w.mean())
    add("V-D", "遠端排隊實驗的連線延遲中位數（範圍）", "14–29 ms", f"{min(meds) * 1000:.0f}–{max(meds) * 1000:.0f} ms",
        "out/plan_m1_runlog.csv、plan_win_runlog.csv")
    add("V-D", "網路來回對平均等待的影響上限（排隊比例 × 來回時間 ÷ 等待）", "最多 1%", f"最多 {worst:.1%}",
        "同上 ＋ out/m1_*、win_*.csv", ok=worst <= 0.0105)
    add("V-D", "HP：最高負載的等待是預測的幾倍", "2.7", f(wq(top[2]) / top[1]["pred"], 1), "out/win_rho70.csv ＋ plan_win.json")
    add("V-D", "全部事前預測在範圍內", "16／21", f"{sum(h for h, _ in HITS.values())}／{sum(n for _, n in HITS.values())}",
        "V-C、V-D 各列加總")


# ------------------------------------------------------------------ V-E 輸出長度
def sec_length(summ):
    pe, ev = [], []
    for line in open(os.path.join(EXP, "llama_strengthen.log"), encoding="utf-8", errors="replace"):
        m = re.search(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", line)
        if m:
            pe.append(int(m.group(2)) / float(m.group(1)) * 1000)
        m = re.search(r"\s eval time =\s*([\d.]+) ms /\s*(\d+) tokens", line)
        if m and int(m.group(2)) == 128:
            ev.append(128 / float(m.group(1)) * 1000)
    add("V-E", "decode 吞吐量比 prefill 低幾倍", "42", f"{np.median(pe) / np.median(ev):.0f}", "同上")
    s = {"short": [], "long": []}
    for name in ("mix_B_rho50", "mix_B_rho75"):
        d = run(name)
        for i, r in enumerate(d["rows"]):
            s[r["prompt_type"]].append(d["S"][i])
    add("V-E", "prompt 長 12.5 倍時服務時間多幾 %（1,478 vs 118 tokens）", "29%",
        f"{np.mean(s['long']) / np.mean(s['short']) - 1:.0%}", "out/mix_B_rho50.csv、mix_B_rho75.csv")

    def ratio(a, c):
        sa, sc = win(run(a))[2], win(run(c))[2]
        return ((sc ** 2).mean() / sc.mean()) / ((sa ** 2).mean() / sa.mean())
    rr = ratio("var2_A_rho50", "var2_C_rho50")
    add("V-E", "同利用率下 (2) 預測 C 等 A 的幾倍（複製實驗那兩組）", "約 1.4", f(rr, 2),
        "out/var2_A_rho50.csv、var2_C_rho50.csv", ok=abs(rr - 1.4) < 0.05)
    wa, wc = float(summ["var2_A_rho50"]["W"]), float(summ["var2_C_rho50"]["W"])
    add("V-E", "複製實驗：C 等 A 的幾倍（實測）", "1.15", f(wc / wa, 2), "同上")
    add("V-E", "複製實驗：兩組的利用率（C／A）", "0.42／0.47",
        f"{float(summ['var2_C_rho50']['rho_hat']):.2f}／{float(summ['var2_A_rho50']['rho_hat']):.2f}", "同上")
    add("V-E", "閒置效應模型預測的比值", "1.20",
        f(float(summ["var2_C_rho50"]["mod_mean"]) / float(summ["var2_A_rho50"]["mod_mean"]), 2), "重跑 analyze_strengthen.py")


# ------------------------------------------------------------------ VI 卸載（重跑 make_offload_v2.py）
def sec_offload():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    txt = subprocess.run([sys.executable, os.path.join(PAPER_DIR, "make_offload_v2.py")], cwd=PAPER_DIR, env=env,
                         capture_output=True, text=True, encoding="utf-8").stdout
    table = {int(k): [c.split(" (")[0].strip() for c in v.split("|")]
             for k, v in re.findall(r"k=(\d+): (.+)", txt)}
    paper = {2: ["always", "0.39", "0.18", "0.09"], 4: ["always", "always", "0.45", "0.22"],
             8: ["always", "always", "always", "0.48"]}
    for k in (2, 4, 8):
        add("VI 表 II", f"k = {k}（N = 2／4／8／16）", "／".join(paper[k]), "／".join(table.get(k, [])), "重跑 make_offload_v2.py")
    olds = [float(x) for x in re.findall(r"\((0\.\d+)\)", txt)]
    news = [float(x) for k in (2, 4, 8) for x in table.get(k, []) if x != "always"]
    add("VI", "閒置效應讓門檻移動最多多少", "0.03", f(max(abs(a - b) for a, b in zip(news, olds)), 2), "同上")
    m = re.search(r"極低負載時本機的平均服務時間 = ([\d.]+)", txt)
    add("VI", "極低負載時本機的平均服務時間", "2.13", f(float(m.group(1)), 2), "同上")
    m = re.search(r"ρ_L\* = ([\d.]+)", txt)
    add("VI", "k=4、N=8：邊緣伺服器一直保溫時的門檻（論文寫不變，0.45）", "0.45", f(float(m.group(1)), 2), "同上")


# ------------------------------------------------------------------ VII 教訓
def sec_lessons(summ):
    size = os.path.getsize(os.path.join(os.path.expanduser("~"), "LLM", "models", "qwen2.5-coder-3b-instruct-q4_k_m.gguf")) / 1e9
    add("VII-1", "decode 速度需要的記憶體頻寬（模型大小 × 80 tokens/s）比 CPU 的 51.2 GB/s 高", "比 CPU 的頻寬高",
        f"{size * 80:.0f} GB/s（{size * 80 / 51.2:.1f} 倍）", "模型檔大小", ok=size * 80 > 51.2)
    import random
    rng = np.random.default_rng(0)
    ratios = []
    for name, r in summ.items():
        if float(r["rho_hat"]) <= 0.55:
            continue
        W = win(D[name])[1]
        boot = rng.choice(W, size=(4000, len(W))).mean(1)
        iid = (np.percentile(boot, 97.5) - np.percentile(boot, 2.5)) / 2
        ratios.append((float(r["bm_hi"]) - float(r["bm_lo"])) / 2 / iid)
    add("VII-4", "利用率 > 0.55 時，批次平均區間比獨立 bootstrap 寬幾倍", "1.7–5.0",
        f"{min(ratios):.1f}–{max(ratios):.1f}", "17 組中 ρ̂ > 0.55 的 10 組")
    m85 = run("main2_rho85")
    W = m85["W"]
    q = [x.mean() for x in np.array_split(W, 4)]
    u85 = lam_hat(m85) * win(m85)[2].mean()
    u85n = 0.42485 * win(m85)[2].mean()
    lat = max(float(x["t_send"]) - float(x["t_scheduled"]) for x in rows_of("var_C_rho75"))
    add("III-F", "筆電進入待命時產生器卡住（那組的最大送出延誤）", "> 0.01 秒（作廢）", f"{lat:,.0f} 秒", "out/var_C_rho75.csv", ok=lat > 1)


# ------------------------------------------------------------------ 第十輪（10/11）新增的實驗與分析
def sec_round10():
    # V-A：同一天、同樣設定，不鎖 vs 鎖時脈
    b0, d0, lo0, hi0 = probe_extra("probe_base_1011")
    b1, d1, lo1, hi1 = probe_extra("probe_locked_1011")
    add("V-A", "10/11 不鎖時脈：閒置 ≥3 秒多出的時間", "0.31", f(d0, 2), "out/probe_base_1011.csv")
    add("V-A", "10/11 鎖時脈：閒置 ≥3 秒多出的時間（95% CI）", "0.007 (−0.002, 0.016)",
        f"{d1:.3f} ({lo1:.3f}, {hi1:.3f})".replace("-", "−"), "out/probe_locked_1011.csv")
    clk, mem, ps = [], [], set()
    for line in open(os.path.join(OUT, "probe_locked_1011_gpu.csv"), encoding="utf-8", errors="replace"):
        c = [x.strip() for x in line.split(",")]
        try:
            clk.append(float(c[2])); mem.append(float(c[3])); ps.add(c[1])
        except (ValueError, IndexError):
            pass
    add("V-A", "鎖時脈那次：顯示卡最高／記憶體時脈、P-state", "2,550／8,001 MHz，只有 P0",
        f"{max(clk):,.0f}／{max(mem):,.0f} MHz（記憶體 {min(mem):,.0f}–{max(mem):,.0f}），{'、'.join(sorted(ps))}",
        "out/probe_locked_1011_gpu.csv", ok=max(clk) == 2550 and min(mem) == max(mem) == 8001 and ps == {"P0"})
    # V-D：短 prompt 的對照
    b2, d2, lo2, hi2 = probe_extra("probe_short3b_1011")
    add("V-D", "RTX 4050 用 118 token prompt、16 token 輸出：閒置 ≥3 秒多出的時間（95% CI）", "0.24 (0.21, 0.28)",
        f"{d2:.2f} ({lo2:.2f}, {hi2:.2f})", "out/probe_short3b_1011.csv")
    # III-E、V-C：check_outsample.py
    import check_outsample as co
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        fk = co.fake_server()
        mae_set, mae_std = co.outsample()
    add("III-E", "假伺服器：代入實際到達時間模擬的平均（ρ=0.5／0.8）", "0.241／1.009",
        "／".join(f"{x[2]:.3f}" for x in fk), "稽核/check_outsample.py")
    add("III-E", "假伺服器：兩組實測都在模擬的 95% 範圍內", "是", "是" if all(x[5] for x in fk) else "否",
        "同上", ok=all(x[5] for x in fk))
    add("V-C", "事前預測的 9 組 3B：標準 M/G/1／閒置效應模型的平均絕對誤差", "7.6%／5.5%",
        f"{mae_std:.1%}／{mae_set:.1%}", "同上")


# ------------------------------------------------------------------ 執行
def main():
    sec_testbed()
    sec_validation()
    sec_idle()
    summ = sec_waiting()
    sec_predictions()
    sec_hardware()
    sec_length(summ)
    sec_offload()
    sec_lessons(summ)
    sec_round10()
    ok = sum(1 for r in RESULTS if r["ok"])
    lines = ["# 論文數字重算結果", "",
             f"由 稽核/recompute.py 產生。共 {len(RESULTS)} 項：✅ {ok} 項一致，⚠️ {len(RESULTS) - ok} 項要看。", "",
             "| 段落 | 項目 | 論文寫的 | 重算 | 判定 | 來源（抽查用） |", "|---|---|---|---|---|---|"]
    for r in RESULTS:
        mark = "✅" if r["ok"] else "⚠️ " + r["note"]
        lines.append(f"| {r['sec']} | {r['what']} | {r['paper']} | {r['got']} | {mark} | {r['src']} |")
    md = NL.join(lines) + NL
    open(os.path.join(HERE, "重算結果.md"), "w", encoding="utf-8").write(md)
    for r in RESULTS:
        print(f"{'✅' if r['ok'] else '⚠️'} [{r['sec']}] {r['what']}：論文 {r['paper']}｜重算 {r['got']}")
    print(f"{NL}共 {len(RESULTS)} 項，✅ {ok}，⚠️ {len(RESULTS) - ok} → 稽核/重算結果.md")


if __name__ == "__main__":
    main()
