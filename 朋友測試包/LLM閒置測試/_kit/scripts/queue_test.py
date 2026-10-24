"""LLM 排隊測試（2026-10-19；10/21 v2：方法 3 用）：在朋友的電腦上驗證論文的排隊模型

做法和論文 10/11–10/12 在 RTX 4050 上的事前預測相同：
1. 校正（約 7 分鐘）：閒置 0、0.5、1、1.5、1.75、2、2.25、2.5、3、5、8 秒各 3 個請求為一個區塊，3 個區塊，
   量一般服務時間 b 和閒置後多出的時間（v2 加了 1.75、2.25、2.5 秒：代價剛開始的那一段）
2. 排隊（約 25–40 分鐘）：同一個伺服器，請求照 Poisson 隨機到達、時間到就送，不等前一個做完（open-loop）。
   負載 ρ = 0.3、0.5、0.7，λ 照校正量到的 b 換算；到達時間用固定的亂數種子，事後可以完全重建
3. 照比例忙（v2b 加）：50%、75% 忙各 3 分鐘（每個請求做完，休息「服務時間 ×（1 ÷ 比例 − 1）」），量中間負載的服務時間（方法 3′）
4. 連續請求 8 分鐘（v2 加）：一個接一個送、中間不閒置，量一直忙、顯示卡變熱時的服務時間（方法 3 的「熱的 b」）
5. 結果存到桌面；全部正常跑完就在 2 分鐘後自動關機（10/21 使用者要求，和閒置測試的睡前跑相同：
   倒數時按任意鍵可以取消；有地方沒跑完或出錯就不關機，留著視窗）
預測由 Claude 收到結果後、只用校正和連續請求的資料先算好並記下時間，之後才打開排隊的資料。
設計和事前登記：朋友測試包/排隊測試/設計說明.md（v2 是 10/21 20:10 那一節）

這一輪不算的條件（和閒置測試相同的部分）：模型沒有完全放進顯示卡、用錯顯示卡、請求失敗、用電池、睡著或卡住
排隊的部分照論文：每組的送出延誤最大 ≤ 1 秒、全部成功

用法：python queue_test.py [--quick]（--quick 只給作者試跑：校正 1 區塊、排隊 1 組 20 個請求）
"""
import csv
import datetime
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import idle_test as it                                       # 沿用閒置測試的伺服器、選卡、打包、個資處理

KIT_VERSION = "2026-10-21-queue-v2b"                             # v2b：連續請求前加 50%、75% 兩段（方法 3′）
CAL_GAPS = [0, 0.5, 1, 1.5, 1.75, 2, 2.25, 2.5, 3, 5, 8]        # v2 加 1.75、2.25、2.5
CAL_REPS, CAL_BLOCKS = 3, 3
LOADS = [("rho30", 0.3, 150), ("rho50", 0.5, 200), ("rho70", 0.7, 300)]
SEED = 2026
SUSTAIN_S = 480                                                  # v2：排隊之後連續請求 8 分鐘，取後一半的平均
DUTY_STEPS = [(0.5, 180), (0.75, 180)]                           # v2b：連續請求之前，先 50%、75% 忙各 3 分鐘
QUICK = "--quick" in sys.argv
if QUICK:
    CAL_BLOCKS, LOADS, SUSTAIN_S, DUTY_STEPS = 1, [("rho50", 0.5, 20)], 60, [(0.5, 20), (0.75, 20)]
CAL_FIELDS = ["req_id", "block", "gap", "idle_before", "t_send", "wall_send", "ttft", "service", "out_tokens", "error", "stall"]
Q_FIELDS = ["req_id", "t_scheduled", "t_send", "wall_send", "t_first_token", "t_done", "ttft", "total", "out_tokens",
            "status", "error", "stall"]
S_FIELDS = ["req_id", "t_send", "wall_send", "ttft", "service", "out_tokens", "error"]
D_FIELDS = ["duty", "req_id", "t_send", "wall_send", "ttft", "service", "out_tokens", "error", "pause"]


def arrivals(lam, n, seed):
    """Poisson 到達時間（秒，從 0 開始）；分析時用同樣的種子和 λ 重建"""
    r, t, out = random.Random(seed), 0.0, []
    for _ in range(n):
        t += r.expovariate(lam)
        out.append(t)
    return out


def run_load(server, prompt, max_tokens, lam, n, seed, tag, path):
    """open-loop：照排好的時間送出，每個請求一個執行緒，不等前一個回應"""
    sched = arrivals(lam, n, seed)
    rows, lock, threads = [], threading.Lock(), []

    def one(i, t_s, t0):
        row = {"req_id": i, "t_scheduled": round(t_s, 4), "status": "ok", "error": ""}
        wall = time.time()
        try:
            ts, tf, td, wall, k = server.request(f"[{tag} {i:04d}] " + prompt, max_tokens)
            wall_end = time.time()
            row.update(t_send=round(ts - t0, 4), wall_send=round(wall, 4), t_first_token=round(tf - t0, 4) if tf else "",
                       t_done=round(td - t0, 4), ttft=round(tf - ts, 4) if tf else "", total=round(td - ts, 4),
                       out_tokens=k, stall=round((wall_end - wall) - (td - ts), 3))
            if k == 0:
                row.update(status="error", error="no output tokens")
        except Exception as e:
            row.update(status="error", error=f"{type(e).__name__}: {e}"[:200], wall_send=round(wall, 4))
        with lock:
            rows.append(row)

    t0 = time.perf_counter()
    for i, t_s in enumerate(sched):
        it.wait_until(t0 + t_s)
        th = threading.Thread(target=one, args=(i, t_s, t0), daemon=True)
        th.start()
        threads.append(th)
        if (i + 1) % 50 == 0:
            it.say(f"  {tag}：已送出 {i + 1}/{n}")
    for th in threads:
        th.join(timeout=900)
    rows.sort(key=lambda r: r["req_id"])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=Q_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in Q_FIELDS})
    late = [float(r["t_send"]) - float(r["t_scheduled"]) for r in rows if r.get("t_send") not in (None, "")]
    ok = sum(1 for r in rows if r["status"] == "ok")
    stall = max([float(r["stall"]) for r in rows if r.get("stall") not in (None, "")] or [0.0])
    return {"tag": tag, "lambda": lam, "n": n, "seed": seed, "ok": ok, "max_lateness_s": round(max(late or [0.0]), 3),
            "late_gt_1s": sum(1 for x in late if x > 1), "max_stall_s": round(stall, 3),
            "minutes": round(sched[-1] / 60, 1)}


def run_sustained(server, prompt, max_tokens, seconds, path):
    """v2（10/21）：一個接一個連續送 seconds 秒，中間不閒置，量一直忙的時候的服務時間。每筆即時寫入"""
    rows, t0 = [], time.perf_counter()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=S_FIELDS)
        w.writeheader()
        while time.perf_counter() - t0 < seconds:
            rid = len(rows)
            ts, tf, td, wall, n = server.request(f"[sustain {rid:04d}] " + prompt, max_tokens)
            rows.append({"req_id": rid, "t_send": round(ts - t0, 4), "wall_send": round(wall, 4),
                         "ttft": round(tf - ts, 4) if tf else "", "service": round(td - ts, 4), "out_tokens": n,
                         "error": "" if n else "no output tokens"})
            w.writerow(rows[-1])
            fh.flush()
            if len(rows) % 50 == 0:
                it.say(f"  連續請求：{len(rows)} 個，{(time.perf_counter() - t0) / 60:.1f}/{seconds / 60:.0f} 分鐘")
    late = [r["service"] for r in rows if r["t_send"] >= seconds / 2 and not r["error"]]     # 後一半
    return {"seconds": seconds, "n": len(rows), "failed": sum(1 for r in rows if r["error"]),
            "n_second_half": len(late), "b_second_half": round(sum(late) / len(late), 4) if late else None,
            "complete": time.perf_counter() - t0 >= seconds}


def run_duty_steps(server, prompt, max_tokens, steps, path):
    """v2b（10/21）：照比例忙——每個請求做完，休息「上一個的服務時間 ×（1 ÷ 比例 − 1）」再送下一個；
    50% 時休息和服務一樣久，75% 時休息服務的 1/3。每一段取後一半的請求算平均；每筆即時寫入 duty_steps.csv"""
    out = []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=D_FIELDS)
        w.writeheader()
        for duty, seconds in steps:
            rows, t0 = [], time.perf_counter()
            busy = 0.0
            while time.perf_counter() - t0 < seconds:
                rid = len(rows)
                ts, tf, td, wall, n = server.request(f"[duty{int(duty * 100)} {rid:04d}] " + prompt, max_tokens)
                pause = (td - ts) * (1 / duty - 1)
                rows.append({"duty": duty, "req_id": rid, "t_send": round(ts - t0, 4), "wall_send": round(wall, 4),
                             "ttft": round(tf - ts, 4) if tf else "", "service": round(td - ts, 4), "out_tokens": n,
                             "error": "" if n else "no output tokens", "pause": round(pause, 4)})
                w.writerow(rows[-1])
                fh.flush()
                busy += td - ts
                it.wait_until(td + pause)
            el = time.perf_counter() - t0
            half = [r["service"] for r in rows if r["t_send"] >= seconds / 2 and not r["error"]]
            out.append({"duty": duty, "seconds": seconds, "n": len(rows), "failed": sum(1 for r in rows if r["error"]),
                        "busy_fraction": round(busy / el, 4) if el > 0 else None, "n_second_half": len(half),
                        "b_second_half": round(sum(half) / len(half), 4) if half else None, "complete": el >= seconds})
            it.say(f"  {int(duty * 100)}% 忙：{len(rows)} 個請求" + (f"，b = {out[-1]['b_second_half']:.3f} 秒" if half else ""))
    return out


def shutdown_countdown(seconds=120):
    """10/21：跑完自動關機（和閒置測試 idle_test.py 的睡前跑相同，只換訊息）；倒數期間按任意鍵可以取消。
    試跑模式（--quick）只倒數，不會真的關機"""
    import msvcrt
    if QUICK:
        seconds = 15
        it.say(f"（試跑模式：不會真的關機）倒數 {seconds} 秒，按任意鍵可以取消。")
    else:
        subprocess.run(["shutdown", "/s", "/t", str(seconds), "/c",
                        "LLM 排隊測試完成，結果已經存到桌面。電腦將在 2 分鐘後關機。"], capture_output=True)
        it.say(f"電腦將在 {seconds // 60} 分鐘後關機。醒著的話，按任意鍵可以取消。")
    t0 = time.time()
    while time.time() - t0 < seconds:
        if msvcrt.kbhit():
            msvcrt.getwch()
            if not QUICK:
                subprocess.run(["shutdown", "/a"], capture_output=True)
            it.say("已取消關機。")
            return
        time.sleep(0.2)
    if QUICK:
        it.say("（試跑模式：倒數結束。正式跑的話，這時候電腦會關機。）")


def main():
    it.disable_quick_edit()
    it.say("=" * 60)
    it.say("  LLM 排隊測試（約 1 小時，跑完會自動關機）")
    it.say("=" * 60)
    it.say("這次量的是「請求一個接一個隨機進來、要排隊時」的等待時間，用來驗證論文的模型。")
    it.say("跑的時候請：")
    it.say("  1. 關掉遊戲、影片、瀏覽器、動態桌布（例如 Wallpaper Engine）等其他程式（有沒存的檔案請先存好）")
    it.say("  2. 筆電要插著電源，不要闔上螢幕；桌機可以直接關掉螢幕")
    it.say("  3. 不要點這個視窗，也不要用電腦")
    it.say("  4. 全部跑完、結果存到桌面後，電腦會在 2 分鐘後【自動關機】（醒著的話按任意鍵可以取消）。")
    it.say("     關機時其他程式會被強制關閉，所以請先存檔、關掉其他程式。")
    it.say("收集的只有時間紀錄、GPU／CPU 狀態和硬體型號，使用者名稱和電腦名稱會遮掉。")
    it.say()
    it.pause("準備好了按 Enter 開始……")

    need = [os.path.join(it.KIT, "model", "qwen2.5-coder-3b-instruct-q4_k_m.gguf"),
            os.path.join(it.KIT, "llama-vulkan", "llama-server.exe")]
    if not all(os.path.exists(p) for p in need):
        it.say("找不到測試需要的檔案。請先把下載的 zip 整個解壓縮，再到解壓縮後的資料夾裡執行。")
        it.pause()
        return
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(it.WORK, f"queue_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    info = {"kit_version": KIT_VERSION, "mode": "queue", "start": datetime.datetime.now().isoformat(),
            "cal_gaps": CAL_GAPS, "cal_reps": CAL_REPS, "cal_blocks": CAL_BLOCKS, "loads": LOADS, "seed": SEED,
            "notes": [], "quick": QUICK}
    server, mon, label, unfinished = None, None, "排隊_GPU", True
    it.keep_awake(True)
    try:
        has_batt, ac = it.on_ac_power()
        info["laptop_battery"] = has_batt
        while has_batt and not ac:
            it.say("偵測到筆電沒有插電。請插上電源（每 3 秒檢查一次；要放棄請關掉這個視窗）。")
            time.sleep(3)
            has_batt, ac = it.on_ac_power()
        it.say("記錄硬體資訊……")
        hw = it.collect_hw()
        json.dump(hw, open(os.path.join(run_dir, "hardware.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        exe = need[1]
        dev, dev_name, integrated, listing = it.pick_gpu(exe)
        open(os.path.join(run_dir, "devices.txt"), "w", encoding="utf-8").write(listing)
        if not dev:
            it.say("這台電腦找不到可以用的 GPU（Vulkan），排隊測試沒辦法跑。")
            info["notes"].append("no Vulkan device")
            return
        info.update(device=dev, device_name=dev_name, integrated=integrated)
        label = f"排隊_GPU_{it.short_name(dev_name, 'GPU')}"
        nvidia = it.nvidia_smi() if "nvidia" in dev_name.lower() else None
        it.say("使用的顯示卡：" + dev_name)
        args = ["-m", os.path.basename(need[0]), "-c", "4096", "--parallel", "1", "-t", "4", "--cache-ram", "0",
                "--device", dev, "-ngl", "all"]
        it.say("確認模型能不能全部放進顯示卡……")
        probe_log = os.path.join(run_dir, "placement_probe.log")
        probe = it.Server(exe, args + ["-lv", "4"], probe_log, cwd=os.path.dirname(need[0]))
        probe_ok = probe.start()
        probe_mem = it.gpu_process_memory(probe.proc.pid) if probe_ok else None
        probe.stop()
        pr = it.offload_info(probe_log)
        pr.update(started=probe_ok, gpu_mem_gb=probe_mem)
        info["placement_probe"] = pr
        info["llama_version"] = it.tool_version(exe)
        info["python_version"] = sys.version
        if not (probe_ok and pr.get("offloaded") and pr["offloaded"] == pr.get("layers")):
            it.say("這張顯示卡放不下整個模型（或伺服器沒有啟動），排隊測試不跑。")
            info["notes"].append("model does not fit on the GPU")
            return
        it.nvidia_snapshot(run_dir, "start")
        cpu_u, gpu_u = it.busy_check(nvidia)
        info["busy_before"] = {"cpu_percent": cpu_u, "gpu_percent": gpu_u}
        if (cpu_u is not None and cpu_u > 30) or (gpu_u is not None and gpu_u > 15):
            it.say(f"電腦好像還在忙（CPU {cpu_u}%、GPU {gpu_u}%）。請關掉其他程式。")
            it.pause("關好之後按 Enter 繼續……")

        now = it.find_device(exe, dev_name, dev)                # 照名稱再確認一次編號
        if now and now != dev:
            info["notes"].append(f"{dev_name} moved from {dev} to {now}")
            dev = now
            args[args.index("--device") + 1] = dev
        log_path = os.path.join(run_dir, "server.log")
        server = it.Server(exe, args, log_path, cwd=os.path.dirname(need[0]))
        it.say()
        it.say("啟動伺服器……")
        if not server.start():
            info["notes"].append("server failed to start")
            it.say("伺服器沒有啟動成功，停止。")
            return
        mem = it.gpu_process_memory(server.proc.pid)
        info["gpu_mem_gb"] = mem
        if probe_mem and mem is not None and mem < (1 - it.PLACEMENT_DROP) * probe_mem:
            info["notes"].append("model only partly on GPU")
        mon = it.Monitors("gpu", run_dir, "_all", nvidia)       # 整段時間的狀態記錄（round_all_gpu_nvidia.csv 等）
        info["monitors_active_start"] = it.active_monitors()
        prompt_file, max_tokens = it.WORKLOADS["main"]
        prompt = open(os.path.join(it.KIT, "prompts", prompt_file), encoding="utf-8").read()
        for i in range(it.WARMUP):                              # 暖機，不記錄
            server.request(f"[warmup {i}] " + prompt, max_tokens)

        # ---- 1. 校正
        it.say("第 1 步：校正（約 7 分鐘）……")
        cal_rows, rng = [], random.Random(SEED)
        last_end, last_wall = time.perf_counter(), time.time()
        batt_seen = False
        for b in range(1, CAL_BLOCKS + 1):
            has_batt, ac = it.on_ac_power()
            batt_seen = batt_seen or (has_batt and not ac)
            order = CAL_GAPS * CAL_REPS
            rng.shuffle(order)
            for g in order:
                it.wait_until(last_end + g)
                rid = len(cal_rows)
                ts, tf, td, wall, n = server.request(f"[cal {rid:04d}] " + prompt, max_tokens)
                wall_end = time.time()
                stall = max((ts - last_end) - g, (wall - last_wall) - g, (wall_end - wall) - (td - ts))
                cal_rows.append({"req_id": rid, "block": b, "gap": g, "idle_before": round(ts - last_end, 4),
                                 "t_send": round(ts, 4), "wall_send": round(wall, 4), "ttft": round(tf - ts, 4) if tf else "",
                                 "service": round(td - ts, 4), "out_tokens": n, "error": "" if n else "no output tokens",
                                 "stall": round(stall, 3)})
                last_end, last_wall = td, wall_end
        with open(os.path.join(run_dir, "calibration.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CAL_FIELDS)
            w.writeheader()
            w.writerows(cal_rows)
        s0 = [r["service"] for r in cal_rows if r["gap"] == 0]
        b_cal = sum(s0) / len(s0)
        info["calibration"] = {"b": round(b_cal, 4), "n": len(cal_rows), "battery_seen": batt_seen,
                               "max_stall_s": max(r["stall"] for r in cal_rows),
                               "failed": sum(1 for r in cal_rows if r["error"])}
        it.say(f"  一個接一個的服務時間 b = {b_cal:.3f} 秒")

        # ---- 2. 排隊（順序用固定種子打亂，和論文 run_plan.py 相同的做法）
        order = list(enumerate(LOADS))
        random.Random(SEED + 2000).shuffle(order)
        info["runs"] = []
        for k, (idx, (tag, rho, n)) in enumerate(order, 1):
            lam = round(rho / b_cal, 5)
            it.say()
            it.say(f"第 2 步（{k}/{len(order)}）：排隊測試 {tag}，λ = {lam}/秒，{n} 個請求，約 {n / lam / 60:.0f} 分鐘……")
            has_batt, ac = it.on_ac_power()
            res = run_load(server, prompt, max_tokens, lam, n, SEED + 10 + idx, tag, os.path.join(run_dir, f"queue_{tag}.csv"))
            has_batt2, ac2 = it.on_ac_power()
            res["battery_seen"] = (has_batt and not ac) or (has_batt2 and not ac2)
            res["usable"] = (res["ok"] == n and res["max_lateness_s"] <= 1.0 and res["late_gt_1s"] == 0
                             and not res["battery_seen"] and res["max_stall_s"] <= it.STALL_MAX)
            res["end"] = datetime.datetime.now().isoformat()
            info["runs"].append(res)
            it.say(f"  完成：成功 {res['ok']}/{n}，送出延誤最大 {res['max_lateness_s']} 秒，{'可用' if res['usable'] else '不可用'}")
            json.dump(info, open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        unfinished = False                                      # 排隊三組都跑完了；下面的連續請求沒跑完也照樣打包

        # ---- 3. 照比例忙（v2b，10/21）：50%、75% 忙的時候的服務時間（方法 3′）
        it.say()
        it.say(f"第 3 步：照比例忙（{'、'.join(f'{int(d * 100)}% {s / 60:.0f} 分鐘' for d, s in DUTY_STEPS)}）……")
        try:
            info["duty_steps"] = run_duty_steps(server, prompt, max_tokens, DUTY_STEPS, os.path.join(run_dir, "duty_steps.csv"))
        except Exception as e:
            info["duty_steps"] = [{"error": f"{type(e).__name__}: {e}"[:200]}]
            info["notes"].append("duty steps failed")
        json.dump(info, open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

        # ---- 4. 連續請求（v2，10/21）：一直忙、顯示卡變熱時的服務時間
        it.say()
        it.say(f"第 4 步：連續請求 {SUSTAIN_S / 60:.0f} 分鐘（量一直忙的時候的速度）……")
        has_batt, ac = it.on_ac_power()
        info["sustained"] = {"started": datetime.datetime.now().isoformat()}
        try:
            info["sustained"].update(run_sustained(server, prompt, max_tokens, SUSTAIN_S, os.path.join(run_dir, "sustained.csv")))
        except Exception as e:
            info["sustained"]["error"] = f"{type(e).__name__}: {e}"[:200]
            info["notes"].append("sustained segment failed")
        has_batt2, ac2 = it.on_ac_power()
        info["sustained"]["battery_seen"] = (has_batt and not ac) or (has_batt2 and not ac2)
        info["sustained"]["end"] = datetime.datetime.now().isoformat()
        b_hot = info["sustained"].get("b_second_half")
        it.say(f"  完成：{info['sustained'].get('n', 0)} 個請求" + (f"，一直忙的時候 b = {b_hot:.3f} 秒" if b_hot else ""))
        json.dump(info, open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    except KeyboardInterrupt:
        info["notes"].append("interrupted by user")
    except Exception:
        info["notes"].append("error: " + traceback.format_exc())
        it.say("發生錯誤，已經跑完的部分還是會打包。")
    finally:
        if mon:
            mon.stop()
        info["monitors_active_end"] = it.active_monitors()
        if server:
            server.stop()
            used = it.device_in_log(os.path.join(run_dir, "server.log"), info.get("device"))
            info["device_used"] = used
            if used and used != info.get("device_name"):
                info["notes"].append(f"wrong device: {used}")
        it.keep_awake(False)
        it.nvidia_snapshot(run_dir, "end")
        info["end"] = datetime.datetime.now().isoformat()
        all_ok = False
        try:
            json.dump(info, open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            dest = it.package(run_dir, label, unfinished)
            it.say()
            it.say("=" * 60)
            it.say("完成！結果已經存到桌面：" if not unfinished else "測試沒有全部跑完，已經跑完的部分存到桌面：")
            it.say("  " + os.path.basename(dest))
            it.say("請把這個檔案傳給請你幫忙測試的人（LINE 或 email 都可以）。")
            it.say("=" * 60)
            # 10/21：全部正常跑完才自動關機；有地方沒跑完或出錯就留著視窗，朋友回來看得到發生什麼事
            all_ok = (not unfinished and info.get("runs") and all(r.get("usable") for r in info["runs"])
                      and (info.get("sustained") or {}).get("complete")
                      and info.get("duty_steps") and all(s.get("complete") for s in info["duty_steps"])
                      and not any(str(n).startswith(("error", "interrupted")) for n in info["notes"]))
        except Exception:
            it.say("打包結果時出了問題，請把這個視窗截圖傳給請你幫忙測試的人。")
            it.say(traceback.format_exc())
        if all_ok:
            shutdown_countdown()
        else:
            it.say("有地方沒有正常跑完，為了安全，這次不自動關機。請把桌面上的結果 zip 和這個視窗的截圖傳給請你幫忙測試的人。")
            it.pause("按 Enter 關閉……")


if __name__ == "__main__":
    main()
