"""第 6、9 台排隊結果的資料檢查（10/24，看過結果之後才做；獨立重算，不用 analyze_queue.py）

用法：python -I integrity_check.py <結果zip> [<舊結果zip>]
"""
import csv
import io
import json
import random
import statistics as st
import sys
import zipfile

import numpy as np

EDGES = [1.0, 2.0, 5.0]


def rows(z, name):
    return list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8", "replace"))))


def waits_completion_order(q):
    """照做完的順序重建；回傳照 req_id 排的 (W, S, 有沒有排隊)"""
    q = sorted(q, key=lambda r: int(r["req_id"]))
    send = np.array([float(r["t_send"]) for r in q])
    done = np.array([float(r["t_done"]) for r in q])
    order = np.argsort(done, kind="stable")
    W, S = np.zeros(len(q)), np.zeros(len(q))
    prev = -np.inf
    swaps = 0
    for k, i in enumerate(order):
        start = max(send[i], prev)
        W[i], S[i] = start - send[i], done[i] - start
        prev = done[i]
    # 送出順序和做完順序不一致的次數
    swaps = int(np.sum(np.diff(send[order]) < 0))
    return W, S, swaps


def main(zp, old=None):
    z = zipfile.ZipFile(zp)
    info = json.loads(z.read("run_info.json"))
    cal = [r for r in rows(z, "calibration.csv") if not r["error"]]
    S0 = np.array([float(r["service"]) for r in cal if float(r["gap"]) == 0])
    print(f"== {info['device_name']}｜校正 {len(cal)} 個，閒置 0 秒的 b = {S0.mean():.4f}（{len(S0)} 個）")
    print(f"   校正的輸出 token：{sorted(set(r['out_tokens'] for r in cal))}；TTFT 中位數 {st.median(float(r['ttft']) for r in cal):.3f} 秒")
    for run in info["runs"]:
        q = rows(z, f"queue_{run['tag']}.csv")
        q = sorted(q, key=lambda r: int(r["req_id"]))
        # 1. 到達時間：照種子重建 vs 檔案記的排定時間
        r_, t, arr = random.Random(run["seed"]), 0.0, []
        for _ in range(run["n"]):
            t += r_.expovariate(run["lambda"])
            arr.append(t)
        sched = np.array([float(r["t_scheduled"]) for r in q])
        sent = np.array([float(r["t_send"]) for r in q])
        d_arr = np.max(np.abs(np.array(arr) - sched))
        late = sent - sched
        # 2. 請求本身
        toks = sorted(set(r["out_tokens"] for r in q))
        status = sorted(set(r["status"] for r in q))
        ttft = np.array([float(r["ttft"]) for r in q])
        # 3. 等待（照做完的順序）
        W, S, swaps = waits_completion_order(q)
        k = len(q) // 3
        Wm = W[k:].mean()
        queued = W > 0
        Sq = S[k:][queued[k:]]
        print(f"   {run['tag']}：重建的到達 vs 排定 最大差 {d_arr:.5f} 秒｜送出晚 最大 {late.max():.4f}、最小 {late.min():.4f} 秒｜"
              f"輸出 token {toks}｜狀態 {status}｜做完順序對調 {swaps} 次")
        print(f"      實測平均等待（丟前 1/3，獨立重算）{Wm:.3f} 秒｜排隊中的服務時間 中位數 {np.median(Sq):.3f}、"
              f"最大 {Sq.max():.3f}、超過中位數 +0.3 秒的 {int(np.sum(Sq > np.median(Sq) + 0.3))} 個｜"
              f"TTFT 中位數 {np.median(ttft):.3f}、最小 {ttft.min():.3f}、最大 {ttft.max():.3f}")
    # 4. CPU（工作管理員的計數器）：每組期間
    mon = rows(z, "round_all_monitor.csv")
    for run in info["runs"]:
        end = run["end"][11:19]
        seg = [r for r in mon if r["time"][11:19] <= end]
        mon = [r for r in mon if r["time"][11:19] > end]
        cpu = [float(r["cpu_utility"]) for r in seg if r["cpu_utility"]]
        if cpu:
            print(f"   {run['tag']} 期間 CPU 使用率：中位數 {st.median(cpu):.0f}%、90 百分位 {np.percentile(cpu, 90):.0f}%、最大 {max(cpu):.0f}%（{len(cpu)} 筆）")
    if old:
        zo = zipfile.ZipFile(old)
        io_ = json.loads(zo.read("run_info.json"))
        print(f"   和舊的那次比（{io_.get('kit_version')}，{io_['start'][:16]}）：")
        for run in io_["runs"]:
            qo = rows(zo, f"queue_{run['tag']}.csv")
            Wo, So, _ = waits_completion_order(qo)
            k = len(qo) // 3
            qn = rows(z, f"queue_{run['tag']}.csv")
            Wn, Sn, _ = waits_completion_order(qn)
            kn = len(qn) // 3
            print(f"      {run['tag']}：舊 {Wo[k:].mean():.3f} 秒（排隊中服務 {np.median(So[k:][Wo[k:] > 0]):.3f}）"
                  f"｜新 {Wn[kn:].mean():.3f} 秒（{np.median(Sn[kn:][Wn[kn:] > 0]):.3f}）｜λ 舊 {run['lambda']} 新 "
                  f"{next(r['lambda'] for r in info['runs'] if r['tag'] == run['tag'])}")


if __name__ == "__main__":
    main(*sys.argv[1:])
