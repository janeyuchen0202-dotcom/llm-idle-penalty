"""連線檢查（2026-10-08）：資料品質規則 4 用

1. 連續 20 次 /health（每次間隔 0.1 秒）：中位數、最大值 → 附加到 out/rtt_log.csv
   中位數超過 20 ms 或最大超過 0.5 秒 → 結果照放，但標「網路不穩」
2. --idle：網路本身會不會「閒置後變慢」（Wi-Fi 省電）。閒置 g 秒後送一個 /health，
   g = 0、1、2、3、5、8 秒各 4 次、順序隨機 → out/netidle_<label>.csv。
   /health 幾乎不用伺服器運算，所以這裡多出的時間只可能來自網路；用來排除「網路造成的假閒置效應」

用法：python rtt_check.py <label> <base_url> [--idle]
"""
import csv
import os
import random
import statistics as st
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def rtt(base):
    t = time.perf_counter()
    urllib.request.urlopen(base + "/health", timeout=10).read()
    return (time.perf_counter() - t) * 1000


def check20(label, base):
    r = []
    for _ in range(20):
        r.append(rtt(base))
        time.sleep(0.1)
    med, mx = st.median(r), max(r)
    ok = med <= 20 and mx <= 500
    print(f"{label}：連線延遲 20 次，中位數 {med:.1f} ms，最大 {mx:.1f} ms → {'合格' if ok else '標「網路不穩」'}",
          flush=True)
    with open(os.path.join(HERE, "out", "rtt_log.csv"), "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), label, base, round(med, 1), round(mx, 1)])
    return med, mx, ok


def idle_check(label, base):
    gaps = [0, 1, 2, 3, 5, 8] * 4
    random.Random(7).shuffle(gaps)
    rows = []
    for g in gaps:
        time.sleep(g)
        rows.append((g, rtt(base)))
    with open(os.path.join(HERE, "out", f"netidle_{label}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gap", "rtt_ms"])
        w.writerows([(g, round(x, 2)) for g, x in rows])
    for g in sorted(set(gaps)):
        xs = [x for gg, x in rows if gg == g]
        print(f"  網路閒置 {g} 秒後：平均 {st.mean(xs):.1f} ms，中位數 {st.median(xs):.1f} ms，最大 {max(xs):.1f} ms",
              flush=True)


if __name__ == "__main__":
    label, base = sys.argv[1], sys.argv[2]
    check20(label, base)
    if "--idle" in sys.argv:
        idle_check(label, base)
