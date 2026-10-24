"""補強實驗的正式分析（2026-10-06）

做三件事：
1. 對照 10/5 寫下的事前預期（用當時的方法：名目 λ 的 P-K、獨立抽樣 bootstrap 信賴區間）
2. 對照 10/6 00:29 寫下的期中預期（閒置效應模型對後 5 組的預測）
3. 用全部 17 組資料重新估計閒置效應模型，給論文用：
   - 每組實際的到達率 λ̂、批次平均信賴區間（考慮前後請求的等待時間相關）
   - 兩種模型在「同一組到達時間」下的預測：
       一般 M/G/1：服務時間獨立抽樣（和 P-K 的假設相同）
       閒置效應模型：排隊後開始的請求用一般服務時間，閒置後的第一個請求再加 δ（依閒置時間）
   - 輸出 out/strengthen_summary.csv 給畫圖用

用法：python analyze_strengthen.py
"""
import csv
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WARM_DIV = int(os.environ.get("WARM_DIV", "3"))   # 暖機：丟掉每組前 1/WARM_DIV（論文用 3；稽核用 4、2 檢查敏感度）
sys.path.insert(0, HERE)
import analyze_main as am  # noqa: E402  rebuild() 與 boot() 和主實驗的分析完全相同

R = 2000
EDGES = [1.0, 2.0, 5.0]
LABELS = ["<1 s", "1–2 s", "2–5 s", "≥5 s"]
BATCHES = 10
T975_9 = 2.262

# (檔名, 名目 λ, 組別, 新/舊)
RUNS = [("main2_rho30", 0.14995, "main", "舊"), ("main2_rho50", 0.24991, "main", "舊"),
        ("long70_rho70", 0.34988, "main", "舊"), ("long85_rho85", 0.42485, "main", "舊"),
        ("mix_A_rho50", 0.259, "A", "舊"), ("mix_A_rho75", 0.389, "A", "舊"),
        ("var_C_rho50", 0.233, "C", "舊"), ("var_C_rho70", 0.330, "C", "舊"),
        ("rep_rho30", 0.14995, "main", "新"), ("rep_rho50", 0.24991, "main", "新"),
        ("rep_rho70", 0.34988, "main", "新"), ("rep_rho85", 0.42485, "main", "新"),
        ("var2_A_rho50", 0.259, "A", "新"), ("var2_A_rho75", 0.389, "A", "新"),
        ("var2_C_rho50", 0.233, "C", "新"), ("var2_C_rho70", 0.330, "C", "新"),
        ("confirm_C_rho75", 0.35, "C", "新")]
OLD_OF = {"rep_rho30": "main2_rho30", "rep_rho50": "main2_rho50", "rep_rho70": "long70_rho70",
          "rep_rho85": "long85_rho85", "var2_A_rho50": "mix_A_rho50", "var2_A_rho75": "mix_A_rho75",
          "var2_C_rho50": "var_C_rho50", "var2_C_rho70": "var_C_rho70"}
# 期中預期（事前預期.md，2026-10-06 00:29）：組別 → (P-K 名目, 模型預測, 範圍下限, 範圍上限)
MIDRUN = {"rep_rho30": (0.388, 0.327, 0.30, 0.37), "rep_rho50": (0.880, 0.705, 0.63, 0.80),
          "var2_A_rho50": (0.919, 0.723, 0.68, 0.77), "var2_A_rho75": (2.673, 1.675, 1.54, 1.80),
          "var2_C_rho50": (1.109, 0.874, 0.81, 0.95)}


def load(name):
    rows = sorted([r for r in csv.DictReader(open(os.path.join(HERE, "out", name + ".csv"), encoding="utf-8"))
                   if r["status"] == "ok"], key=lambda r: float(r["t_send"]))
    a = np.array([float(r["t_send"]) for r in rows])
    c = np.array([float(r["t_done"]) for r in rows])
    cls = np.array(["A" if int(r["out_tokens"]) == 128 else f"C{r['out_tokens']}" for r in rows])
    prev = np.empty(len(c)); _cp = -np.inf    # 10/22 改：照做完的順序重建（一次只處理一個請求，做完的順序就是處理的順序；原本照送出順序）
    for _j in np.argsort(c, kind="stable"):
        prev[_j] = _cp; _cp = c[_j]
    start = np.maximum(a, prev)
    return a, start - a, c - start, a - prev, cls


DATA = {name: load(name) for name, *_ in RUNS}


def pk(lam, es, es2):
    return lam * es2 / (2 * (1 - lam * es))


def window(name):
    a, W, S, gap, cls = DATA[name]
    k = len(a) // WARM_DIV
    return a[k:], W[k:], S[k:]


# ======================================================================
print("=" * 100)
print("一、10/5 事前預期的判定（當時的方法：名目 λ 的 P-K、獨立抽樣 bootstrap）")
print("=" * 100)
res10 = {}
for name, lam, grp, new in RUNS:
    aw, Ww, Sw = window(name)
    th = pk(lam, Sw.mean(), (Sw ** 2).mean())
    lo, hi = am.boot(list(Ww))
    res10[name] = dict(W=Ww.mean(), th=th, lo=lo, hi=hi, inside=lo <= th <= hi, dev=Ww.mean() / th - 1,
                       rho=lam * Sw.mean())
cover = [n for n in ("rep_rho30", "rep_rho50", "rep_rho70", "rep_rho85") if res10[n]["inside"]]
for n in ("rep_rho30", "rep_rho50", "rep_rho70", "rep_rho85"):
    r = res10[n]
    print(f"  {n:<13} ρ={r['rho']:.3f}  W_q={r['W']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]  P-K={r['th']:.3f}  "
          f"{'涵蓋' if r['inside'] else '未涵蓋'}  偏差 {100 * r['dev']:+.0f}%")
print(f"  → 主實驗重跑：{len(cover)}/4 個點的區間涵蓋 P-K（預期至少 3/4）："
      f"{'符合' if len(cover) >= 3 else '不符合'}")
for n in ("var2_A_rho50", "var2_A_rho75", "var2_C_rho50", "var2_C_rho70"):
    r, o = res10[n], res10[OLD_OF[n]]
    print(f"  {n:<13} 偏差 {100 * r['dev']:+.0f}%（上次 {OLD_OF[n]} {100 * o['dev']:+.0f}%）  "
          f"{'比上次小' if abs(r['dev']) < abs(o['dev']) else '沒有比上次小'}，"
          f"{'在 30% 以內' if abs(r['dev']) < 0.30 else '超過 30%'}")
ratio = res10["var2_C_rho50"]["W"] / res10["var2_A_rho50"]["W"]
print(f"  ρ≈0.5：C 的平均等待是 A 的 {ratio:.2f} 倍（預期 1.5–3 倍）："
      f"{'符合' if 1.5 <= ratio <= 3 else '不符合'}")
log = {r["tag"]: r for r in csv.DictReader(open(os.path.join(HERE, "out", "strengthen_runlog.csv"), encoding="utf-8"))}
c = log["confirm_C_rho75"]
print(f"  確認組：成功 {c['ok']}/{c['total']}，送出延誤最大 {c['max_lateness_s']} 秒（預期全部成功、0.01 秒以內）："
      f"{'符合' if c['ok'] == c['total'] and float(c['max_lateness_s']) < 0.01 else '不符合'}")

# ======================================================================
print("\n" + "=" * 100)
print("二、10/6 期中預期的判定（閒置效應模型對後 5 組的預測，數字照當時寫下的）")
print("=" * 100)
hits, closer = 0, 0
for n, (pkn, pred, lo, hi) in MIDRUN.items():
    m = res10[n]["W"]
    inside = lo <= m <= hi
    near = abs(m - pred) < abs(m - pkn)
    hits += inside
    closer += near
    print(f"  {n:<13} 實測 {m:.3f}  模型 {pred:.3f} [{lo:.2f}, {hi:.2f}]  P-K {pkn:.3f}  "
          f"{'在範圍內' if inside else '在範圍外'}，{'較接近模型' if near else '較接近 P-K'}")
print(f"  → {hits}/5 在範圍內（預期至少 4），{closer}/5 較接近模型（預期 5）："
      f"{'符合' if hits >= 4 and closer == 5 else '不符合'}")

# ======================================================================
print("\n" + "=" * 100)
print("三、閒置效應模型（用全部 17 組重新估計）")
print("=" * 100)
pools, gaps, res = {}, [], []
for name, *_ in RUNS:
    a, W, S, gap, cls = DATA[name]
    for k in sorted(set(cls)):          # 固定順序：set 的順序每次執行可能不同，會讓結果有微小差異
        q = (W > 0) & (cls == k)
        pools.setdefault(k, []).extend(S[q])
        idle = (W == 0) & (cls == k)
        idle[0] = False
        gaps.extend(gap[idle])
        res.extend(S[idle] - S[q].mean())
pools = {k: np.array(v) for k, v in pools.items()}
gaps, res = np.array(gaps), np.array(res)
bins = np.digitize(gaps, EDGES)
dvals = np.array([res[bins == i].mean() for i in range(4)])
dse = np.array([res[bins == i].std() / np.sqrt(np.sum(bins == i)) for i in range(4)])
for i in range(4):
    print(f"  閒置 {LABELS[i]:>6}：n={np.sum(bins == i):>4}  δ = {dvals[i]:+.3f} 秒（SE {dse[i]:.3f}）")
print("  排隊後開始的服務時間：" + "，".join(f"{k} 平均 {v.mean():.3f} 秒（n={len(v)}）" for k, v in sorted(pools.items())))

rng = np.random.default_rng(11)


def sim(a, cls, penalty):
    n, k = len(a), len(a) // WARM_DIV
    base = np.empty((R, n))
    for key in sorted(set(cls)):        # 同上，固定抽樣順序
        idx = np.where(cls == key)[0]
        base[:, idx] = rng.choice(pools[key], size=(R, len(idx)))
    c_prev, acc = np.full(R, -np.inf), np.zeros(R)
    for i in range(n):
        start = np.maximum(a[i], c_prev)
        w = start - a[i]
        pen = np.where(w == 0, dvals[np.digitize(a[i] - c_prev, EDGES)], 0.0) if penalty else 0.0
        c_prev = start + base[:, i] + pen
        if i >= k:
            acc += w
    return acc / (n - k)


def sim_iid(a, S_all):
    """一般 M/G/1：同樣的到達時間，服務時間從該組全部的實測值獨立抽樣（不分閒置或排隊）"""
    n, k = len(a), len(a) // WARM_DIV
    s = rng.choice(S_all, size=(R, n))
    c_prev, acc = np.full(R, -np.inf), np.zeros(R)
    for i in range(n):
        start = np.maximum(a[i], c_prev)
        if i >= k:
            acc += start - a[i]
        c_prev = start + s[:, i]
    return acc / (n - k)


rows = []
print(f"\n  {'組別':<16}{'新舊':>3}{'λ̂/λ':>7}{'實測W_q':>9}{'批次CI':>16}{'P-K名目':>9}{'P-K(λ̂)':>9}"
      f"{'一般MG1':>9}{'百分位':>7}{'閒置模型':>9}{'95%範圍':>15}{'百分位':>7}")
for name, lam, grp, new in RUNS:
    a, W, S, gap, cls = DATA[name]
    k = len(a) // WARM_DIV
    aw, Ww, Sw = a[k:], W[k:], S[k:]
    lh = (len(aw) - 1) / (aw[-1] - aw[0])
    es, es2 = Sw.mean(), (Sw ** 2).mean()
    m = Ww.mean()
    bm = np.array([b.mean() for b in np.array_split(Ww, BATCHES)])
    half = T975_9 * bm.std(ddof=1) / np.sqrt(BATCHES)
    s_iid = sim_iid(a, S)
    s_mod = sim(a, cls, True)
    lo, hi = np.percentile(s_mod, [2.5, 97.5])
    row = dict(run=name, group=grp, new=new, lam=lam, lam_hat=lh, rho_nom=lam * es, rho_hat=lh * es, n=len(Ww),
               W=m, bm_lo=m - half, bm_hi=m + half, pk_nom=pk(lam, es, es2), pk_hat=pk(lh, es, es2),
               iid_mean=s_iid.mean(), iid_lo=np.percentile(s_iid, 2.5), iid_hi=np.percentile(s_iid, 97.5),
               iid_pct=(s_iid < m).mean() * 100, mod_mean=s_mod.mean(), mod_lo=lo, mod_hi=hi,
               mod_pct=(s_mod < m).mean() * 100)
    rows.append(row)
    print(f"  {name:<16}{new:>3}{lh / lam:>7.3f}{m:>9.3f}{f'[{m - half:.2f}, {m + half:.2f}]':>16}"
          f"{row['pk_nom']:>9.3f}{row['pk_hat']:>9.3f}{row['iid_mean']:>9.3f}{row['iid_pct']:>7.0f}"
          f"{row['mod_mean']:>9.3f}{f'[{lo:.2f}, {hi:.2f}]':>15}{row['mod_pct']:>7.0f}")

inside_iid = sum(1 for r in rows if r["iid_lo"] <= r["W"] <= r["iid_hi"])
inside_mod = sum(1 for r in rows if r["mod_lo"] <= r["W"] <= r["mod_hi"])
err = lambda key: np.mean([abs(r["W"] / r[key] - 1) for r in rows]) * 100
print(f"\n  落在 95% 範圍內：一般 M/G/1 {inside_iid}/17，閒置效應模型 {inside_mod}/17")
print(f"  平均絕對誤差：P-K 名目 {err('pk_nom'):.0f}%，P-K(λ̂) {err('pk_hat'):.0f}%，"
      f"一般 M/G/1（同到達時間）{err('iid_mean'):.0f}%，閒置效應模型 {err('mod_mean'):.0f}%")

if WARM_DIV != 3:                       # 敏感度檢查只印結果，不覆蓋畫圖用的檔案
    sys.exit(0)
with open(os.path.join(HERE, "out", "strengthen_summary.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
np.savetxt(os.path.join(HERE, "out", "idle_penalty_bins.csv"), np.column_stack([dvals, dse]), delimiter=",",
           header="delta_s,se_s  (bins: <1, 1-2, 2-5, >=5 s idle)", comments="")
print("\n→ out/strengthen_summary.csv、out/idle_penalty_bins.csv")
