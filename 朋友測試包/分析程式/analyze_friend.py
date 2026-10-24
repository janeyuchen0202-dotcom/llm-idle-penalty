"""朋友電腦閒置測試的分析：照 朋友測試包/排除規則.md，和論文相同的算法
用法：python -I analyze_friend.py <結果zip> [<結果zip> ...]   （只讀 zip，不解壓）"""
import csv
import datetime
import io
import json
import re
import sys
import zipfile

import numpy as np
from scipy import stats

GUARD_START, GUARD_END = 1.2, 0.2


def welch(a, b):
    """平均差 a−b 和 Welch 95% 信賴區間（論文 recompute.py 的算法）"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    d = a.mean() - b.mean()
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(0.975, df)
    return d, d - t * se, d + t * se


def parse_wall(s):
    m = re.match(r"(\d{4})[-/](\d\d)[-/](\d\d)[T ](\d\d):(\d\d):(\d\d)(\.\d+)?\s*(Z|[+-]\d\d:\d\d)?$", s.strip())
    if not m:
        return None
    y, mo, d, h, mi, se = (int(x) for x in m.groups()[:6])
    tz = m.group(8)
    if tz:
        off = 0 if tz == "Z" else (1 if tz[0] == "+" else -1) * (int(tz[1:3]) * 3600 + int(tz[4:6]) * 60)
        dt = datetime.datetime(y, mo, d, h, mi, se, tzinfo=datetime.timezone(datetime.timedelta(seconds=off)))
    else:
        dt = datetime.datetime(y, mo, d, h, mi, se)
    return dt.timestamp() + float(m.group(7) or 0)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def analyze(path):
    zf = zipfile.ZipFile(path)
    info = json.loads(zf.read("run_info.json"))
    rows = list(csv.DictReader(io.StringIO(zf.read("requests.csv").decode("utf-8"))))
    mode = info["mode"]
    out = [f"## {path.rsplit('/', 1)[-1]}", ""]
    for r in rows:
        for k in ("gap", "idle_before", "wall_send", "ttft", "service", "cpu_wait", "stall"):
            r[k] = fnum(r.get(k))
        r["round"] = int(r["round"])
        r["out_tokens"] = int(r["out_tokens"]) if (r.get("out_tokens") or "").strip() else None
        r["monitor"] = int(r.get("monitor") or 0)

    # ---- 排除規則第 1 點：逐輪重新檢查（用原始紀錄算，不只看程式當時的判斷）
    gpu_logs = {}
    if mode == "gpu":
        for n in zf.namelist():
            m = re.match(r"round(\d+)_gpu_nvidia\.csv$", n)
            if m:
                pts = []
                for line in zf.read(n).decode("utf-8", "replace").splitlines():
                    p = [x.strip() for x in line.split(",")]
                    if len(p) != 8:
                        continue
                    t = parse_wall(p[0])
                    vals = [fnum(x) for x in p[3:8]]
                    if t is not None and None not in vals:
                        pts.append((t, p[2], *vals))       # 時間, P-state, 時脈, 記憶體時脈, 使用率, 溫度, 功耗
                gpu_logs[int(m.group(1))] = sorted(pts)
    round_check = []
    for ri in info["rounds_info"]:
        k = ri["round"]
        rr = [r for r in rows if r["round"] == k]
        reasons = []
        if mode == "gpu" and ri.get("placement") == "partial":
            reasons.append("placement=partial")
        used = None
        if mode == "gpu" and info.get("platform") != "macOS" and f"round{k}_server.log" in zf.namelist():   # 10/18 加：這輪實際用的是不是要測的那張顯示卡（Mac 只有一個 GPU，不用查）
            log = zf.read(f"round{k}_server.log").decode("utf-8", "replace")
            m_dev = re.search(r"--device ((?:Vulkan|CUDA)\d+)", log)       # 10/21：CUDA 也認（RTX 4050 補充實驗 A）
            names = dict(re.findall(r"- ((?:Vulkan|CUDA)\d+)\s+:\s+(.+?) \(\d+ MiB", log))   # CUDA 版編號後面有對齊用的空格
            if m_dev and m_dev.group(1) in names:
                used = names[m_dev.group(1)].strip()
                if used != (info.get("device_name") or "").strip():
                    reasons.append(f"wrong_device {used}")
        if any((r["error"] or "").strip() or r["out_tokens"] == 0 for r in rr) or ri.get("requests_done", len(rr)) < len(rr):
            reasons.append("request_failed")
        waits = [r for r in rr if r["gap"] and r["gap"] >= 2 and r["cpu_wait"] is not None]
        cpu = (sum(r["cpu_wait"] * r["idle_before"] for r in waits) / sum(r["idle_before"] for r in waits)) if waits else None
        if cpu is not None and cpu > 30:
            reasons.append(f"cpu_busy {cpu:.1f}%")
        if any((r["stall"] or 0) > 2 or (r["service"] or 0) > 120 for r in rr):
            reasons.append("sleep")
        gpu_u = gpu_n = req_u = None
        if mode == "gpu" and k in gpu_logs:
            L = gpu_logs[k]
            ts = np.array([x[0] for x in L])
            util = np.array([x[4] for x in L])
            wins = [(r["wall_send"] - r["idle_before"] + GUARD_START, r["wall_send"] - GUARD_END)
                    for r in rr if r["gap"] and r["gap"] >= 2 and r["wall_send"]]
            sel = np.zeros(len(ts), bool)
            for a, b in wins:
                if b > a:
                    sel |= (ts >= a) & (ts <= b)
            gpu_n = int(sel.sum())
            gpu_u = float(util[sel].mean()) if gpu_n else None
            busy = np.zeros(len(ts), bool)                       # 對時檢查：請求進行中，顯示卡應該很忙
            for r in rr:
                if r["wall_send"] and r["service"]:
                    busy |= (ts >= r["wall_send"] + 0.2) & (ts <= r["wall_send"] + r["service"] - 0.1)
            req_u = float(util[busy].mean()) if busy.any() else None
            if gpu_u is not None and gpu_u > 10:
                reasons.append(f"gpu_busy {gpu_u:.1f}%")
        round_check.append(dict(round=k, monitor=ri.get("monitor"), program_valid=ri.get("valid"), reasons=reasons, used=used,
                                cpu=cpu, gpu_u=gpu_u, gpu_n=gpu_n, req_u=req_u))
    if mode == "gpu":
        b0_round = {c["round"]: np.mean([r["service"] for r in rows if r["round"] == c["round"] and r["gap"] == 0 and r["service"]])
                    for c in round_check}
        for c, ri in zip(round_check, info["rounds_info"]):
            if ri.get("placement") in (None, "unknown"):
                others = [v for k, v in b0_round.items() if k != c["round"]]
                c["placement_unknown"] = True
                if others and b0_round[c["round"]] > 1.3 * np.median(others):
                    c["reasons"].append("placement=unknown, slow")
    valid_rounds = [c["round"] for c in round_check if not c["reasons"]]

    # ---- 排除規則第 3 點：單一請求比同一輪、同樣閒置秒數的中位數慢 3 倍以上 → 只刪那一個
    keep, dropped = [], []
    for k in valid_rounds:
        for g in sorted({r["gap"] for r in rows if r["round"] == k}):
            grp = [r for r in rows if r["round"] == k and r["gap"] == g and r["service"] is not None]
            med = np.median([r["service"] for r in grp])
            for r in grp:
                (dropped if r["service"] > 3 * med else keep).append(r)

    out.append(f"- 測試：{mode}｜裝置：{info.get('device_name') or 'CPU ' + str(info.get('threads')) + ' 執行緒'}｜工作負載：{info.get('workload')}")
    out.append(f"- 模型放置確認不了（用服務時間檢查）的輪：{[c['round'] for c in round_check if c.get('placement_unknown')] or '沒有'}")
    if mode == "gpu":
        out.append(f"- 每輪實際用的顯示卡：{sorted({str(c['used']) for c in round_check})}")
    out.append(f"- 逐輪重新檢查：{len(valid_rounds)}／{len(round_check)} 輪算數；不算的："
               f"{[(c['round'], c['reasons']) for c in round_check if c['reasons']] or '沒有'}")
    out.append(f"- 等待時 CPU 使用率（每輪）：{[round(c['cpu'], 1) if c['cpu'] is not None else None for c in round_check]}")
    if mode == "gpu":
        out.append(f"- 等待時顯示卡使用率（每輪，有記錄的輪）：{[(c['round'], round(c['gpu_u'], 1), c['gpu_n']) for c in round_check if c['gpu_u'] is not None]}")
        out.append(f"- 對時檢查（請求進行中顯示卡的使用率，應該很高）：{[round(c['req_u']) for c in round_check if c['req_u'] is not None]}")
    out.append(f"- 單一請求刪掉的（慢 3 倍以上）：{len(dropped)} 個 {[(r['round'], r['gap'], round(r['service'], 2)) for r in dropped]}")

    # ---- 閒置效應（和論文表 I 相同：閒置 ≥3 秒減一個接一個）
    S = lambda cond: [r["service"] for r in keep if cond(r)]
    T = lambda cond: [r["ttft"] for r in keep if cond(r) and r["ttft"] is not None]
    D = lambda cond: [r["service"] - r["ttft"] for r in keep if cond(r) and r["ttft"] is not None]
    b0 = S(lambda r: r["gap"] == 0)
    g3 = S(lambda r: r["gap"] >= 3)
    out.append(f"- 一個接一個的服務時間：平均 {np.mean(b0):.3f} 秒（n = {len(b0)}，CV {np.std(b0, ddof=1) / np.mean(b0):.3f}）")
    means = {g: (np.mean(S(lambda r, g=g: r['gap'] == g)), len(S(lambda r, g=g: r['gap'] == g))) for g in sorted({r['gap'] for r in keep})}
    out.append("- 每種閒置秒數的平均服務時間：" + "、".join(f"{int(g)} 秒 {m:.3f}（n={n}）" for g, (m, n) in means.items()))
    d, lo, hi = welch(g3, b0)
    out.append(f"- **閒置 ≥3 秒多出的時間：{d:+.3f} 秒（95% CI {lo:+.3f}, {hi:+.3f}）**，n = {len(b0)} / {len(g3)}，"
               f"是一個接一個的 {d / np.mean(b0):+.1%}")
    d2, lo2, hi2 = welch(S(lambda r: r["gap"] == 2), b0)
    out.append(f"- 閒置 2 秒多出的時間：{d2:+.3f} 秒（{lo2:+.3f}, {hi2:+.3f}）")
    dt_, lot, hit = welch(T(lambda r: r["gap"] >= 3), T(lambda r: r["gap"] == 0))
    dd_, lod, hid = welch(D(lambda r: r["gap"] >= 3), D(lambda r: r["gap"] == 0))
    out.append(f"- 拆開看：第一個 token 之前（讀 prompt）多 {dt_:+.3f} 秒（{lot:+.3f}, {hit:+.3f}）；之後（產生輸出）多 {dd_:+.3f} 秒（{lod:+.3f}, {hid:+.3f}）")
    per_round = []
    for k in valid_rounds:
        a = [r["service"] for r in keep if r["round"] == k and r["gap"] >= 3]
        b = [r["service"] for r in keep if r["round"] == k and r["gap"] == 0]
        per_round.append((k, np.mean(a) - np.mean(b)))
    pr = np.array([x[1] for x in per_round])
    tq = stats.t.ppf(0.975, len(pr) - 1)
    out.append(f"- 每一輪各自算（每輪都重新啟動伺服器）：{[round(x, 3) for _, x in per_round]}")
    out.append(f"  - {len(pr)} 輪的平均 {pr.mean():+.3f} 秒，標準差 {pr.std(ddof=1):.3f}；用輪與輪之間的差異算的 95% CI "
               f"({pr.mean() - tq * pr.std(ddof=1) / np.sqrt(len(pr)):+.3f}, {pr.mean() + tq * pr.std(ddof=1) / np.sqrt(len(pr)):+.3f})")
    mon = [x for k, x in per_round if dict((c['round'], c['monitor']) for c in round_check)[k]]
    nolog = [x for k, x in per_round if not dict((c['round'], c['monitor']) for c in round_check)[k]]
    out.append(f"  - 有記錄狀態的輪 {np.mean(mon):+.3f} 秒（{len(mon)} 輪）｜不記錄的對照輪 {np.mean(nolog):+.3f} 秒（{len(nolog)} 輪）")

    # ---- 顯示卡時脈：前一個請求結束後 0–8 秒（閒置 8 秒的請求，像論文 Fig. 1b）
    if mode == "gpu" and gpu_logs:
        offs = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5]
        clk, p8 = {o: [] for o in offs}, {o: [] for o in offs}
        first_p8 = []
        for r in keep:
            L = gpu_logs.get(r["round"])
            if not L or r["gap"] != 8:
                continue
            ts = np.array([x[0] for x in L])
            ws = r["wall_send"] - r["idle_before"]
            for o in offs:
                i = int(np.argmin(np.abs(ts - (ws + o))))
                if abs(ts[i] - (ws + o)) < 0.3:
                    clk[o].append(L[i][2])
                    p8[o].append(L[i][1] == "P8")
            w = [(x[0] - ws) for x in L if ws <= x[0] <= r["wall_send"] and x[1] == "P8"]
            if w:
                first_p8.append(min(w))
        out.append("- 顯示卡時脈（閒置 8 秒的請求，前一個結束後幾秒：時脈中位數 MHz／在 P8 的比例）：" +
                   "、".join(f"{o} 秒 {np.median(clk[o]):.0f}／{np.mean(p8[o]):.0%}" for o in offs if clk[o]))
        out.append(f"- 進入 P8 的時間（中位數）：{np.median(first_p8):.1f} 秒（{len(first_p8)} 個請求裡 {len(first_p8)} 個有進入）" if first_p8 else "- 閒置 8 秒內沒有進入 P8")
        allp = [x for L in gpu_logs.values() for x in L]
        p8w = [x[6] for x in allp if x[1] == "P8"]
        idlew = [x[6] for x in allp if x[4] == 0]
        busyw = [x[6] for x in allp if x[4] >= 50]
        out.append(f"- 耗電：P8 時中位數 {np.median(p8w):.1f} W（{len(p8w)} 筆）｜顯示卡閒著（使用率 0）{np.median(idlew):.1f} W｜忙碌時 {np.median(busyw):.1f} W"
                   if p8w else f"- 耗電：沒有 P8 的紀錄｜閒著 {np.median(idlew):.1f} W")
        out.append(f"- 出現過的 P-state：{sorted({x[1] for x in allp})}｜最高時脈 {max(x[2] for x in allp):.0f} MHz")
    out.append("")
    return out, dict(mode=mode, d=d, lo=lo, hi=hi, b0=np.mean(b0), keep=keep, rows=rows, info=info,
                     round_check=round_check, valid_rounds=valid_rounds, gpu_logs=gpu_logs, dropped=dropped)


if __name__ == "__main__":
    lines = []
    for p in sys.argv[1:]:
        o, _ = analyze(p)
        lines += o
    print("\n".join(lines))
