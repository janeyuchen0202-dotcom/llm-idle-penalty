"""RTX 4050 補充實驗 A（2026-10-21，設計：RTX4050補充實驗_設計草稿_20261021.md）：CUDA 版和 Vulkan 版輪流跑閒置測試
判定標準：事前預期.md（2026-10-21 的事前登記）

量法和朋友的測試包相同，伺服器、請求、狀態記錄、排除規則都直接用測試包 idle_test.py 裡的函式：
- 閒置 0、2、3、5、8 秒，每輪 60 個請求（5 種 × 3 次 × 4 區塊，順序用種子 2026 + 輪次打亂），每輪重新啟動伺服器、先暖機 3 個
- 伺服器參數：-c 4096 --parallel 1 -t 4 --cache-ram 0 --device <RTX 4050 的編號> -ngl all
- 第 5、10 輪不記錄顯示卡狀態（對照）；每輪結束照排除規則檢查，不算的輪次在最後補跑（每個後端最多 5 輪）
只換後端，llama.cpp 都是 b9549（04eb4c446）：
- CUDA：%USERPROFILE%/LLM-cuda-b9549/bin（官方 CUDA 12.4 版和 cudart，SHA-256 和 GitHub 公布的相同）
- Vulkan：%USERPROFILE%/LLM（論文用的）
輪流：CUDA 第 1 輪 → Vulkan 第 1 輪 → CUDA 第 2 輪 → ……各 10 輪；同一對的閒置順序一樣
結果：out/backend_<時間>/cuda、vulkan（格式和測試包相同），另外各打包成 out/backend_<時間>_cuda.zip、_vulkan.zip
     分析：analyze_backend_probe.py

用法：python run_backend_probe.py                  正式（run_rtx4050_supplement.py 會呼叫）
      python run_backend_probe.py --quick          試跑：真的伺服器，每個後端 1 輪、15 個請求（不算數）
      python run_backend_probe.py --dry --quick    試流程：假伺服器（fake_server.py），不用顯示卡
      加 --force-invalid：把 CUDA 第 1 輪當成不算，確認補跑會動（只給試跑、試流程用）
"""
import csv
import datetime
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
KIT_SCRIPT = os.path.join(os.path.dirname(HERE), "朋友測試包", "LLM閒置測試", "_kit", "scripts", "idle_test.py")
_spec = importlib.util.spec_from_file_location("idle_test", KIT_SCRIPT)
IT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(IT)                         # 只用測試包裡的函式，不執行它的主程式

DRY, QUICK, FORCE_INVALID = "--dry" in sys.argv, "--quick" in sys.argv, "--force-invalid" in sys.argv
UP = os.environ["USERPROFILE"]
MODEL = os.path.join(UP, "LLM", "models", "qwen2.5-coder-3b-instruct-q4_k_m.gguf")
BACKENDS = {"cuda": os.path.join(UP, "LLM-cuda-b9549", "bin", "llama-server.exe"),
            "vulkan": os.path.join(UP, "LLM", "llama-server.exe")}
GPU_NAME = "NVIDIA GeForce RTX 4050 Laptop GPU"
ROUNDS, BLOCKS = (1, 1) if QUICK else (10, IT.BLOCKS)
STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M")
PFX = "backendq" if (DRY or QUICK) else "backend"   # 試跑、試流程的資料夾和正式的分開
OUT = os.path.join(HERE, "out", f"{PFX}_{STAMP}")
FIELDS = ["round", "req_id", "block", "gap", "idle_before", "t_send", "wall_send", "ttft", "service",
          "out_tokens", "workload", "monitor", "cpu_wait", "error", "stall"]


def say(m=""):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


# ---------------------------------------------------------------- 顯示卡的編號（測試包的版本只認 Vulkan，這裡 CUDA 也認）
def list_devices(exe):
    r = subprocess.run([exe, "--list-devices"], capture_output=True, timeout=120, cwd=os.path.dirname(exe))
    out = (r.stdout + r.stderr).decode("utf-8", "replace")
    devs = [(m.group(1), m.group(2), int(m.group(3)))
            for m in re.finditer(r"^\s*((?:CUDA|Vulkan)\d+):\s*(.+?)\s*\((\d+) MiB", out, flags=re.M)]
    return devs, out


def find_gpu(exe, prefer=None):
    """RTX 4050 現在的編號：照名稱找，同名的有好幾張就優先用原本的編號；找不到回傳 None"""
    try:
        devs, _ = list_devices(exe)
    except Exception:
        return None
    same = [d[0] for d in devs if d[1] == GPU_NAME]
    return (prefer if prefer in same else same[0]) if same else None


def device_in_log(log_path, dev):
    """伺服器紀錄開頭的裝置清單裡，編號 dev 對到的顯示卡名稱（讀不到回傳 None）"""
    try:
        txt = open(log_path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    # CUDA 版的編號後面有對齊用的空格（「- CUDA0   : …」），Vulkan 版只有一個
    return dict(re.findall(r"- ((?:CUDA|Vulkan)\d+)\s+:\s+(.+?) \(\d+ MiB", txt)).get(dev, "").strip() or None


class FakeServer(IT.Server):
    """試流程用：開 fake_server.py 代替 llama-server（不用顯示卡）；送請求、關閉都和測試包的 Server 相同"""
    def start(self, timeout=60):
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        cmd = [sys.executable, os.path.join(HERE, "fake_server.py"), "--dist", "uniform", "--mean", "0.4", "--spread", "0.05",
               "--port", str(self.port), "--log", self.log_path + ".fake.csv"]
        self.log.write("COMMAND: " + " ".join(cmd) + "\n")
        self.log.flush()
        self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=self.log, stderr=subprocess.STDOUT, creationflags=0x08000000)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                if urllib.request.urlopen(self.base + "/health", timeout=3).status == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False


# ---------------------------------------------------------------- 一個後端
class Arm:
    """一個後端：自己的資料夾、run_info.json、requests.csv（和測試包 run_test 的格式相同，analyze_friend.py 可以直接讀）"""

    def __init__(self, name, exe, hw):
        self.name, self.exe = name, exe
        self.dir = os.path.join(OUT, name)
        os.makedirs(self.dir, exist_ok=True)
        json.dump(hw, open(os.path.join(self.dir, "hardware.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        self.plan = [r % 5 != 0 for r in range(1, ROUNDS + 1)]        # 第 5、10 輪不記錄狀態（和測試包相同）
        self.cap, self.makeups, self.rnd, self.stopped = (ROUNDS + 1) // 2, 0, 0, False
        self.probe_full, self.probe_mem, self.full_seen, self.mon = False, None, False, None
        self.workload, self.reps = "main", IT.REPS
        self.nvidia = IT.nvidia_smi()
        self.t_begin = time.time()
        self.info = {"kit_version": IT.KIT_VERSION, "kit_script": os.path.relpath(KIT_SCRIPT, os.path.dirname(HERE)),
                     "backend": name, "mode": "gpu", "start": datetime.datetime.now().isoformat(), "gaps": IT.GAPS,
                     "reps": IT.REPS, "blocks": BLOCKS, "rounds": ROUNDS, "warmup": IT.WARMUP, "rounds_done": 0,
                     "notes": [], "rounds_info": [], "dry": DRY, "quick": QUICK}
        self.fh = open(os.path.join(self.dir, "requests.csv"), "w", newline="", encoding="utf-8")   # 每筆即時寫入
        self.writer = csv.DictWriter(self.fh, fieldnames=FIELDS)
        self.writer.writeheader()

    def save_info(self):
        json.dump(self.info, open(os.path.join(self.dir, "run_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def setup(self):
        devs, listing = list_devices(self.exe)
        open(os.path.join(self.dir, "devices.txt"), "w", encoding="utf-8").write(listing)
        self.dev = next((d[0] for d in devs if d[1] == GPU_NAME), None)
        if not self.dev:
            raise RuntimeError(f"{self.name}：llama.cpp 找不到 {GPU_NAME}\n{listing[-500:]}")
        self.args = ["-m", os.path.basename(MODEL), "-c", "4096", "--parallel", "1", "-t", "4", "--cache-ram", "0",
                     "--device", self.dev, "-ngl", "all"]
        self.info.update(device=self.dev, device_name=GPU_NAME, integrated=False, threads=4, workload=self.workload,
                         label=f"GPU_RTX-4050-Laptop_{self.name}", exe=self.exe, llama_version=IT.tool_version(self.exe),
                         python_version=sys.version)
        say(f"{self.name}：{self.info['llama_version']}｜{self.dev} = {GPU_NAME}")
        if DRY:
            pr = {"started": None, "note": "dry run: no placement probe"}
        else:
            # 和測試包相同：開始前用詳細紀錄（-lv 4）啟動一次，確認模型全部放進顯示卡、占多少顯示卡記憶體
            probe_log = os.path.join(self.dir, "placement_probe.log")
            probe = IT.Server(self.exe, self.args + ["-lv", "4"], probe_log, cwd=os.path.dirname(MODEL))
            ok = probe.start()
            self.probe_mem = IT.gpu_process_memory(probe.proc.pid) if ok else None
            probe.stop()
            pr = IT.offload_info(probe_log)
            pr.update(started=ok, gpu_mem_gb=self.probe_mem)
            self.probe_full = bool(pr.get("offloaded")) and pr["offloaded"] == pr.get("layers")
            say(f"  模型放進顯示卡：{pr.get('offloaded')}/{pr.get('layers')} 層，占 {self.probe_mem} GB")
            if not self.probe_full:
                self.info["placement_probe"] = pr
                raise RuntimeError(f"{self.name}：模型沒有全部放進顯示卡（{pr.get('offloaded')}/{pr.get('layers')} 層），停止。")
        self.info["placement_probe"] = pr
        self.full_seen = self.probe_full
        IT.nvidia_snapshot(self.dir, "start")
        cpu_u, gpu_u = IT.busy_check(self.nvidia)
        self.info["busy_before"] = {"cpu_percent": cpu_u, "gpu_percent": gpu_u}
        self.save_info()

    def pending(self):
        return not self.stopped and self.rnd < len(self.plan)

    def run_round(self):
        """一輪：和測試包 run_test 裡的一輪相同（暖機 3 個 → 4 個區塊 → 排除規則檢查 → 不算就排補跑）"""
        self.rnd += 1
        rnd, monitor_on = self.rnd, self.plan[self.rnd - 1]
        tag = f"{self.name.upper()} 第 {rnd} 輪" + ("（補跑）" if rnd > ROUNDS else "")
        say()
        say(f"{tag}：啟動伺服器……")
        log_path = os.path.join(self.dir, f"round{rnd}_server.log")
        if not DRY:
            now = find_gpu(self.exe, self.dev)                  # 每輪照名稱重新找編號（測試包 10/18 的修正）
            if now and now != self.dev:
                self.info["notes"].append(f"round {rnd}: {GPU_NAME} moved from {self.dev} to {now}")
                say(f"  顯示卡的編號變了，改用 {now}")
                self.dev = now
                self.args[self.args.index("--device") + 1] = now
        server = (FakeServer if DRY else IT.Server)(self.exe, self.args, log_path, cwd=os.path.dirname(MODEL))
        if not server.start():
            server.stop()
            self.info["notes"].append(f"round {rnd}: server failed to start")
            say("  伺服器沒有啟動成功，這個後端停止（已經跑完的輪次照樣保留）。")
            self.stopped = True
            self.save_info()
            return
        try:
            rinfo = {"round": rnd, "makeup": rnd > ROUNDS, "monitor": int(monitor_on), "device": self.dev,
                     "wall_start": datetime.datetime.now().isoformat(), "gpu_mem_gb": IT.gpu_process_memory(server.proc.pid),
                     "monitors_active_start": IT.active_monitors()}     # 螢幕開或關：每輪開始、結束各記一次
            placement_ok = True
            if self.probe_full and self.probe_mem and rinfo["gpu_mem_gb"] is not None:
                placement_ok = rinfo["gpu_mem_gb"] >= (1 - IT.PLACEMENT_DROP) * self.probe_mem
                rinfo["placement"] = "full" if placement_ok else "partial"     # 少 5% 以上：這次放得比較少
            else:
                rinfo["placement"] = "unknown"         # 確認不了：分析時用服務時間檢查（排除規則第 1 點）
            if not placement_ok:
                self.info["notes"].append(f"round {rnd}: model only partly on GPU")
            errors, on_batt, waits, stalls = [], False, [], []
            cpu_idle = cpu_all = 0
            prompt_file, max_tokens = IT.WORKLOADS[self.workload]
            prompt = open(os.path.join(IT.KIT, "prompts", prompt_file), encoding="utf-8").read()
            try:
                warm = []
                for i in range(IT.WARMUP):                              # 暖機，不記錄
                    ts, tf, td, wall, n = server.request(f"[warmup {rnd}-{i}] " + prompt, max_tokens)
                    warm.append(td - ts)
                rinfo["warmup_service"] = [round(x, 3) for x in warm]
            except Exception as e:
                errors.append(f"warmup: {type(e).__name__}: {e}"[:200])

            self.mon = IT.Monitors("gpu", self.dir, rnd, self.nvidia) if monitor_on and not errors else None
            rng = random.Random(2026 + rnd)                             # 和測試包相同：同一輪次的兩個後端順序一樣
            n_req, done = BLOCKS * len(IT.GAPS) * self.reps, 0
            last_end = time.perf_counter()
            last_wall = time.time()                                     # 上一個請求結束的牆上時間（偵測睡著用）
            for b in range(1, BLOCKS + 1):
                if errors:                                              # 出了問題：這輪停下來，最後補跑
                    break
                has_batt, ac = IT.on_ac_power()
                on_batt = on_batt or (has_batt and not ac)
                order = IT.GAPS * self.reps
                rng.shuffle(order)
                for g in order:
                    st0 = IT.sys_times() if g >= 2 else None             # 等待前後各讀一次 CPU 時間
                    IT.wait_until(last_end + g)
                    st1 = IT.sys_times() if st0 else None
                    cpu_w = ""
                    if st0 and st1 and st1[1] > st0[1]:
                        cpu_idle += st1[0] - st0[0]
                        cpu_all += st1[1] - st0[1]
                        cpu_w = round(100 * (1 - (st1[0] - st0[0]) / (st1[1] - st0[1])), 1)
                    rid = (rnd - 1) * 1000 + done
                    row = dict(round=rnd, req_id=rid, block=b, gap=g, workload=self.workload, monitor=int(monitor_on),
                               cpu_wait=cpu_w, error="")
                    try:
                        ts, tf, td, wall, n = server.request(f"[request {rid:05d}] " + prompt, max_tokens)
                    except Exception as e:
                        row["error"] = f"{type(e).__name__}: {e}"[:200]
                        errors.append(f"request {rid}: {row['error']}")
                        self.writer.writerow(row)
                        self.fh.flush()
                        break
                    wall_end = time.time()
                    # 睡著或卡住：等待比設定的久、牆上時間和程式計時對不上，或一個請求久得不正常
                    stall = max((ts - last_end) - g, (wall - last_wall) - g, (wall_end - wall) - (td - ts))
                    if stall > IT.STALL_MAX or td - ts > IT.LONG_REQ_S:
                        stalls.append(rid)
                    row.update(idle_before=round(ts - last_end, 4), t_send=round(ts, 4), wall_send=round(wall, 4),
                               ttft=round(tf - ts, 4) if tf else "", service=round(td - ts, 4), out_tokens=n,
                               stall=round(stall, 3))
                    self.writer.writerow(row)
                    self.fh.flush()
                    if n == 0:
                        errors.append(f"request {rid}: no output tokens")
                    if g >= 2:
                        waits.append((wall - (ts - last_end), wall))    # 這段等待的開始和結束（牆上時間）
                    last_end, last_wall = td, wall_end
                    done += 1
                    if done % 20 == 0 or done == n_req:
                        say(f"  {tag}：{done}/{n_req}")
            if self.mon:
                self.mon.stop()
                self.mon = None
            has_batt, ac = IT.on_ac_power()
            on_batt = on_batt or (has_batt and not ac)
            IT.nvidia_snapshot(self.dir, f"round{rnd}_end")
        finally:
            if self.mon:
                self.mon.stop()
                self.mon = None
            server.stop()
        rinfo["monitors_active_end"] = IT.active_monitors()

        # ---- 這一輪算不算（排除規則第 1 點，和測試包相同）；分析時會用同樣的紀錄再算一次
        reasons = []
        if not placement_ok:
            reasons.append("placement")
        rinfo["device_used"] = device_in_log(log_path, self.dev)        # 伺服器紀錄裡這個編號對到的顯示卡
        if rinfo["device_used"] and rinfo["device_used"] != GPU_NAME:
            reasons.append("wrong_device")
        if errors or done < n_req:
            reasons.append("request_failed")
        if on_batt:
            reasons.append("battery")
        if stalls:
            reasons.append("sleep")
            rinfo["stalled_requests"] = stalls[:20]
        rinfo["cpu_wait_percent"] = round(100 * (1 - cpu_idle / cpu_all), 1) if cpu_all else None
        if rinfo["cpu_wait_percent"] is not None and rinfo["cpu_wait_percent"] > IT.CPU_BUSY_MAX:
            reasons.append("cpu_busy")
        if monitor_on:
            try:
                u, k = IT.gpu_util_in_waits(os.path.join(self.dir, f"round{rnd}_gpu_nvidia.csv"), waits, 8)
                rinfo["gpu_wait_percent"] = round(u, 1) if u is not None else None
                rinfo["gpu_wait_samples"] = k
                if u is not None and u > IT.GPU_BUSY_MAX:
                    reasons.append("gpu_busy")
                elif k < IT.MIN_GPU_SAMPLES:
                    rinfo["gpu_check"] = "insufficient"                 # 監控樣本太少，無法判定
            except Exception:
                rinfo["gpu_check_error"] = traceback.format_exc()[-300:]
        if FORCE_INVALID and self.name == "cuda" and rnd == 1:
            reasons.append("forced_test")
        rinfo.update(valid=not reasons, invalid_reasons=reasons, errors=errors[:10], requests_done=done)
        self.full_seen = self.full_seen or placement_ok
        if reasons:
            why = "、".join(IT.REASON_ZH.get(r, r) for r in reasons)
            structural = "placement" in reasons and not self.full_seen
            if self.makeups < self.cap and not structural:
                self.plan.append(monitor_on)
                self.makeups += 1
                say(f"  這一輪不算（{why}），最後會多跑一輪補上。")
            else:
                say(f"  這一輪不算（{why}）。" + ("補跑的次數已經用完。" if self.makeups >= self.cap else ""))
        self.info["rounds_info"].append(rinfo)
        self.info.update(rounds_done=rnd, rounds_planned=ROUNDS, makeup_cap=self.cap, makeup_rounds=self.makeups,
                         valid_rounds=sum(1 for x in self.info["rounds_info"] if x.get("valid")),
                         no_monitor_rounds=[i + 1 for i, m in enumerate(self.plan) if not m])
        self.save_info()

    def finish(self):
        if not self.fh.closed:
            self.fh.close()
        self.info["minutes"] = round((time.time() - self.t_begin) / 60, 1)
        self.info["end"] = datetime.datetime.now().isoformat()
        IT.nvidia_snapshot(self.dir, "end")
        self.save_info()
        dest = f"{OUT}_{self.name}.zip"
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            for fn in sorted(os.listdir(self.dir)):
                z.write(os.path.join(self.dir, fn), fn)
        say(f"{self.name}：{self.info.get('valid_rounds', 0)}/{self.rnd} 輪算數 → {os.path.relpath(dest, HERE)}")


def main():
    IT.keep_awake(True)
    arms, ok = [], False
    try:
        for exe in list(BACKENDS.values()) + [MODEL]:
            if not os.path.exists(exe):
                sys.exit(f"找不到 {exe}")
        has_batt, ac = IT.on_ac_power()
        if has_batt and not ac:
            sys.exit("筆電沒有插電，停止。")
        os.makedirs(OUT, exist_ok=True)
        say(f"實驗 A：結果存到 {os.path.relpath(OUT, HERE)}" + ("（試流程：假伺服器）" if DRY else "（試跑，不算數）" if QUICK else ""))
        say("記錄硬體資訊……")
        hw = IT.collect_hw()
        arms = [Arm(n, exe, hw) for n, exe in BACKENDS.items()]
        for a in arms:
            a.setup()
        say(f"開始：CUDA、Vulkan 輪流，各 {ROUNDS} 輪（每輪 {BLOCKS * len(IT.GAPS) * IT.REPS} 個請求）")
        while any(a.pending() for a in arms):
            for a in arms:
                if a.pending():
                    a.run_round()
                    if any(x.pending() for x in arms):
                        time.sleep(10)                  # 和測試包相同：輪與輪之間停 10 秒
        ok = not any(a.stopped for a in arms)
        say("實驗 A 全部跑完。" if ok else "實驗 A 沒有全部跑完（見上面的訊息）。")
    except SystemExit:
        raise
    except Exception:
        say("發生錯誤：\n" + traceback.format_exc())
        for a in arms:
            a.info["notes"].append("error: " + traceback.format_exc()[-2000:])
    finally:
        for a in arms:
            try:
                a.finish()
            except Exception:
                say(f"{a.name} 打包失敗：\n" + traceback.format_exc())
        IT.keep_awake(False)
        print(f"RESULT_DIR={OUT}", flush=True)          # 給 run_rtx4050_supplement.py 找結果用
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
