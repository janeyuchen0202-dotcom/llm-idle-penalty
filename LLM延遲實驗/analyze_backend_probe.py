"""RTX 4050 補充實驗 A 的分析：CUDA 版 vs Vulkan 版（判定標準：事前預期.md 2026-10-21 的事前登記）
- 兩個後端各用 analyze_friend.py 的算法：排除規則逐輪重新檢查、單一請求慢 3 倍以上只刪那一個、Welch 95% CI
- 判定 1：CUDA 的閒置 ≥3 秒代價，95% CI 下限 > 0.02 秒 →「CUDA 也有閒置代價」（和朋友的總表同一個標準）
- 判定 2：CUDA 代價 ÷ Vulkan 代價（點估計）在 0.5–1.5 之間 →「兩個後端的代價差不多」
- 只描述、不判定：以輪算的 CI、同一對（同一個種子）兩個後端的代價差、比值的 bootstrap 範圍、
  每種閒置秒數的代價、明確開始的閒置秒數、兩個後端一個接一個的服務時間
用法：python analyze_backend_probe.py [out/backend_<時間>]    沒給就用最新的正式結果
"""
import glob
import importlib.util
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "analyze_friend", os.path.join(os.path.dirname(HERE), "朋友測試包", "分析程式", "analyze_friend.py"))
AF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AF)
MEANINGFUL = 0.02                  # 和朋友的總表相同：代價 ≥ 20 毫秒才算「會變慢」
RATIO_RANGE = (0.5, 1.5)
GAPS = (2, 3, 5, 8)


def services(res, cond):
    return [r["service"] for r in res["keep"] if cond(r)]


def penalty_by_round(res):
    """每一輪各自算的代價（閒置 ≥3 秒 − 一個接一個），只用算數的輪"""
    out = {}
    for k in res["valid_rounds"]:
        a = services(res, lambda r: r["round"] == k and r["gap"] >= 3)
        b = services(res, lambda r: r["round"] == k and r["gap"] == 0)
        if a and b:
            out[k] = float(np.mean(a) - np.mean(b))
    return out


def t_ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() - h), float(x.mean() + h)


def boot_penalty(res, rng, n=5000):
    b0 = np.array(services(res, lambda r: r["gap"] == 0))
    g3 = np.array(services(res, lambda r: r["gap"] >= 3))
    if not len(b0) or not len(g3):                    # 沒有算數的輪：無法估計
        return np.full(n, np.nan)
    return g3[rng.integers(0, len(g3), (n, len(g3)))].mean(1) - b0[rng.integers(0, len(b0), (n, len(b0)))].mean(1)


def main():
    if len(sys.argv) > 1:
        base = os.path.abspath(sys.argv[1].rstrip("/\\"))
    else:
        dirs = sorted(d for d in glob.glob(os.path.join(HERE, "out", "backend_*")) if os.path.isdir(d))
        if not dirs:
            sys.exit("找不到 out/backend_* 的結果。")
        base = dirs[-1]
    res, lines, summary = {}, [], {}
    for be in ("cuda", "vulkan"):
        o, r = AF.analyze(f"{base}_{be}.zip")
        lines += [f"# {be.upper()}", ""] + o
        res[be] = r
    for be, r in res.items():
        b0 = services(r, lambda x: x["gap"] == 0)
        per_gap = {g: AF.welch(services(r, lambda x, g=g: x["gap"] == g), b0) for g in GAPS}
        onset = next((g for g, (m, lo, hi) in per_gap.items() if lo > 0 and m >= MEANINGFUL), None)
        pr = penalty_by_round(r)
        rlo, rhi = t_ci(list(pr.values()))
        summary[be] = {"b0": float(np.mean(b0)), "n_b0": len(b0), "penalty": float(r["d"]), "ci": [float(r["lo"]), float(r["hi"])],
                       "pct_of_b0": float(r["d"] / np.mean(b0)), "valid_rounds": len(r["valid_rounds"]),
                       "rounds": len(r["round_check"]), "dropped": len(r["dropped"]),
                       "per_gap": {str(g): [round(float(v), 4) for v in per_gap[g]] for g in GAPS},
                       "onset_s": onset, "round_penalties": {str(k): round(v, 4) for k, v in pr.items()},
                       "round_mean": float(np.mean(list(pr.values()))) if pr else float("nan"), "round_ci": [rlo, rhi]}
    c, v = summary["cuda"], summary["vulkan"]
    hit1 = c["ci"][0] > MEANINGFUL
    ratio = c["penalty"] / v["penalty"] if v["penalty"] > 0 else float("nan")
    hit2 = bool(RATIO_RANGE[0] <= ratio <= RATIO_RANGE[1])
    rng = np.random.default_rng(2026)
    bc, bv = boot_penalty(res["cuda"], rng), boot_penalty(res["vulkan"], rng)
    ok = bv > 0
    blo, bhi = np.percentile(bc[ok] / bv[ok], [2.5, 97.5]) if ok.any() else (float("nan"), float("nan"))
    rounds = res["cuda"]["info"].get("rounds", 10)
    pc, pv = penalty_by_round(res["cuda"]), penalty_by_round(res["vulkan"])
    pairs = sorted(k for k in pc if k in pv and k <= rounds)        # 同一對：同一個輪次（同一個種子）兩邊都算數
    diff = [pc[k] - pv[k] for k in pairs]
    dlo, dhi = t_ci(diff)
    b0d = AF.welch(services(res["cuda"], lambda r: r["gap"] == 0), services(res["vulkan"], lambda r: r["gap"] == 0))

    L = ["# 判定（事前登記的標準）", ""]
    for be, s in summary.items():
        L.append(f"- {be.upper()}：一個接一個 {s['b0']:.3f} 秒（n = {s['n_b0']}）｜閒置 ≥3 秒的代價 {s['penalty']:+.3f} 秒"
                 f"（95% CI {s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}），是一個接一個的 {100 * s['pct_of_b0']:+.1f}%｜"
                 f"以輪算 {s['round_mean']:+.3f}（{s['round_ci'][0]:+.3f}, {s['round_ci'][1]:+.3f}）｜算數的輪 {s['valid_rounds']}/{s['rounds']}")
        L.append("  - 每種閒置秒數：" + "、".join(f"{g} 秒 {s['per_gap'][str(g)][0]:+.3f}（{s['per_gap'][str(g)][1]:+.3f}, "
                                              f"{s['per_gap'][str(g)][2]:+.3f}）" for g in GAPS) +
                 f"｜明確開始：{str(s['onset_s']) + ' 秒' if s['onset_s'] is not None else '沒有'}")
    L.append(f"- **判定 1（CUDA 也有閒置代價：CI 下限 > {MEANINGFUL} 秒）：{'成立' if hit1 else '不成立'}**（下限 {c['ci'][0]:+.3f} 秒）")
    L.append(f"- **判定 2（CUDA ÷ Vulkan 在 {RATIO_RANGE[0]}–{RATIO_RANGE[1]} 倍）：{'成立' if hit2 else '不成立'}**（{ratio:.2f} 倍；"
             f"bootstrap 95% 範圍 {blo:.2f}–{bhi:.2f}，只描述）")
    L.append(f"- 只描述：同一對（{len(pairs)} 對）CUDA − Vulkan 的代價差 平均 {np.mean(diff) if diff else float('nan'):+.3f} 秒"
             f"（{dlo:+.3f}, {dhi:+.3f}）；一個接一個的服務時間 CUDA − Vulkan {b0d[0]:+.3f} 秒（{b0d[1]:+.3f}, {b0d[2]:+.3f}）")
    text = "\n".join(lines + L)
    print(text)
    out = {"base": os.path.relpath(base, HERE), "summary": summary, "hit1_cuda_penalty": bool(hit1), "ratio": ratio,
           "ratio_boot95": [float(blo), float(bhi)], "hit2_ratio": hit2, "pairs": pairs,
           "pair_diff_mean": float(np.mean(diff)) if diff else None, "pair_diff_ci": [dlo, dhi],
           "b0_diff_cuda_minus_vulkan": [float(x) for x in b0d]}
    json.dump(out, open(f"{base}_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open(f"{base}_分析.md", "w", encoding="utf-8").write(text + "\n")
    print(f"\n→ {os.path.relpath(base, HERE)}_results.json、_分析.md")


if __name__ == "__main__":
    main()
