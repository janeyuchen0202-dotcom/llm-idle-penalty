"""各台 GPU 閒置時的核心時脈、記憶體時脈、P-state、功耗（2026-10-20，第 8 台沒有閒置代價之後做的事後比較）

只用閒置 8 秒的請求：前一個請求結束後 0.5、1、2、3、5、7.5 秒，各台所有有效輪的中位數
用法：python -I idle_clock_compare.py <GPU結果zip> [<GPU結果zip> ...]
"""
import csv
import datetime
import io
import json
import sys
import zipfile

import numpy as np

POINTS = [0.5, 1, 2, 3, 5, 7.5]


def load_gpu(z, k):
    out = []
    for p in csv.reader(io.StringIO(z.read(f"round{k}_gpu_nvidia.csv").decode("utf-8", "replace"))):
        try:
            out.append((datetime.datetime.strptime(p[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(), p[2].strip(),
                        float(p[3]), float(p[4]), float(p[7])))
        except (ValueError, IndexError):
            continue
    return out


for path in sys.argv[1:]:
    z = zipfile.ZipFile(path)
    info = json.loads(z.read("run_info.json"))
    reqs = list(csv.DictReader(io.StringIO(z.read("requests.csv").decode("utf-8", "replace"))))
    acc = {t: [] for t in POINTS}
    for r in info["rounds_info"]:
        k = r["round"]
        if not r.get("valid") or f"round{k}_gpu_nvidia.csv" not in z.namelist():
            continue
        g = load_gpu(z, k)
        if not g:
            continue
        ts = np.array([x[0] for x in g])
        for q in reqs:
            if str(q.get("round")) != str(k) or q.get("gap") in (None, "") or float(q["gap"]) != 8:
                continue
            end_prev = float(q["wall_send"]) - float(q["idle_before"])
            for t in POINTS:
                i = int(np.argmin(np.abs(ts - (end_prev + t))))
                if abs(ts[i] - (end_prev + t)) < 0.15:
                    acc[t].append(g[i])
    name = info.get("device_name", "?")
    print(f"== {name}（{path.rsplit(chr(92), 1)[-1]}）")
    for t in POINTS:
        v = acc[t]
        if not v:
            continue
        ps = [x[1] for x in v]
        top = max(sorted(set(ps)), key=ps.count)                 # 一樣多時取編號小的，每次執行結果相同
        print(f"   {t:>4} 秒：核心 {np.median([x[2] for x in v]):>5.0f} MHz｜記憶體 {np.median([x[3] for x in v]):>6.0f} MHz｜"
              f"P-state 最多是 {top}（{ps.count(top)}/{len(ps)}）｜功耗 {np.median([x[4] for x in v]):5.1f} W")
