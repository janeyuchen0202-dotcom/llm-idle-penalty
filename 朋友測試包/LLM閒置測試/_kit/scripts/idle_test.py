"""LLM 閒置測試：給朋友在自己電腦上跑的版本（2026-10-14）

用法（由 BAT 呼叫）：python idle_test.py gpu | cpu | night      （加 --quick：試跑用，每個測試 1 輪、15 個請求）
  night＝睡前跑：GPU 10 輪 → CPU 10 輪 → 結果存到桌面 → 2 分鐘後關機（倒數時按任意鍵可取消）
  試跑時再加 --force-invalid：把第 1 輪當成不算，用來確認補跑會動（BAT 不會用到）

做的事和論文的閒置測試相同：一次只送一個請求，每個請求前先讓伺服器閒置 g 秒
（g = 0、2、3、5、8 秒，順序隨機），看閒置之後的請求會不會變慢。
- gpu：llama.cpp Vulkan 版，模型全部放進獨立顯示卡（沒有獨立顯示卡就用內顯，並標明）
- cpu：llama.cpp CPU 版，只用 CPU
- 每輪重新啟動伺服器；每輪 60 個請求（5 種閒置時間 × 3 次 × 4 個區塊）
- 第 5、10 輪不記錄 GPU／CPU 狀態，用來確認記錄本身不影響結果
- 每輪結束時照排除規則（朋友測試包/排除規則.md）檢查；不算的輪次自動補跑，最多補原定輪數的一半（10/15 加）
- 關掉螢幕、筆電闔上也能正常運作：每輪記錄螢幕開關；中途睡著或卡住的輪次自動補跑；睡前跑睡著過就不自動關機（10/16 加）
- 上次被中斷（例如半夜 Windows 更新重開機）沒打包的結果，下次執行時會先補包到桌面
- 結果打包成 zip 放到桌面；使用者名稱、電腦名稱、路徑都會遮掉

只用 Python 內建模組（可攜版 Python 就能跑）。
"""
import csv
import ctypes
import datetime
import json
import os
import random
import re
import socket
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # _kit
ROOT = os.path.dirname(KIT)                                          # 朋友看到的測試資料夾
WORK = os.path.join(KIT, "work")
KIT_VERSION = "2026-10-18"

GAPS = [0, 2, 3, 5, 8]          # 閒置秒數（和論文 10/11 的測試相同）
REPS, BLOCKS = 3, 4
ROUNDS_DAY, ROUNDS_NIGHT = 5, 10     # 現在跑：5 輪；睡前跑：GPU、CPU 各 10 輪
WARMUP = 3
WORKLOADS = {"main": ("medium.txt", 128),    # 838 token prompt、128 token 輸出（論文 RTX 4050 的主要設定）
             "short": ("short.txt", 16)}     # 118 token prompt、16 token 輸出（論文 HP 的設定）
SLOW_GPU_S = 8.0                # GPU 測試暖機時一個請求超過這個秒數 → 改用 short
SLOW_CPU_S = 15.0               # CPU 測試暖機時超過這個秒數 → 每個區塊的重複次數改成 2

# 排除規則（朋友測試包/排除規則.md，10/15 定案）：每輪結束時檢查，不算的輪次自動補跑
CPU_BUSY_MAX = 30.0             # 等待期間（閒置 2 秒以上的那幾段）CPU 平均使用率超過這個 % → 這輪不算
MIN_GPU_SAMPLES = 20            # 一輪的等待期間少於這麼多筆顯示卡紀錄 → 無法判定有沒有別的程式在用（記下來，分析時另外列）
GPU_BUSY_MAX = 10.0             # GPU 測試：等待期間顯示卡平均使用率超過這個 % → 有別的程式在用，這輪不算
GUARD_START, GUARD_END = 1.2, 0.2   # 算顯示卡使用率時，每段等待去掉開頭和最後幾秒（使用率是往前平均的，開頭會混到上一個請求）
FORCE_INVALID = False           # 試跑用（--force-invalid）：第 1 輪當成不算
QUICK = False                   # 試跑模式（--quick）：睡前跑只倒數、不會真的關機
PLACEMENT_DROP = 0.05            # 某一輪占用的顯示卡記憶體比開始前的確認少 5% 以上 → 這次放得比較少，這輪不算
STALL_MAX = 2.0                 # 某段等待或某個請求比應該的多出這麼多秒 → 電腦中途睡著或卡住，這輪不算
LONG_REQ_S = 120.0              # 一個請求超過這個秒數也當成卡住（正常最慢約 30 秒）
SLEPT = False                   # 這次執行有沒有睡著過；睡前跑睡著過就不自動關機（朋友可能已經醒來在用電腦）
REASON_ZH = {"placement": "模型沒有完全放進顯示卡", "request_failed": "有請求失敗或伺服器中斷",
             "battery": "筆電在用電池", "cpu_busy": "等待時 CPU 太忙", "gpu_busy": "等待時顯示卡被其他程式使用",
             "sleep": "電腦中途睡著或卡住", "forced_test": "試跑：故意當成不算",
             "wrong_device": "用到別張顯示卡（例如內顯）"}

# ---------------------------------------------------------------- 小工具
def say(msg=""):
    print(msg, flush=True)


def pause(msg="按 Enter 繼續……"):
    try:
        input(msg)
    except EOFError:
        pass


def keep_awake(on):
    """執行期間不要休眠、不要關螢幕；程式結束就自動恢復"""
    flags = 0x80000000 | (0x00000001 | 0x00000002 if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def disable_quick_edit():
    """關掉主控台的「快速編輯」：不然朋友點一下視窗，程式就會整個卡住"""
    try:
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-10)
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            k32.SetConsoleMode(h, (mode.value | 0x0080) & ~0x0040)
    except Exception:
        pass


def on_ac_power():
    """回傳 (是否有電池, 是否插著電)"""
    class SPS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                    ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    s = SPS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        return False, True
    has_battery = (s.BatteryFlag & 0xFF) not in (128, 255)
    return has_battery, (s.ACLineStatus & 0xFF) == 1


def desktop_dir():
    """真正的桌面位置（被 OneDrive 接管也找得到）"""
    buf = ctypes.create_unicode_buffer(260)
    if ctypes.windll.shell32.SHGetFolderPathW(None, 0x0010, None, 0, buf) == 0 and os.path.isdir(buf.value):
        return buf.value
    return None


def powershell(cmd, timeout=60):
    """執行一段 PowerShell，回傳 UTF-8 文字（失敗回傳空字串）"""
    full = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + cmd
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", full],
                           capture_output=True, timeout=timeout)
        return r.stdout.decode("utf-8", "replace").strip()
    except Exception:
        return ""


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------- 硬體資訊
# 只讀取、不改設定、不需要系統管理員；不收序號、程式清單這類隱私資訊。每一項失敗都不影響其他項
HW_PS = r"""
$o = [ordered]@{}
function T($f) { try { & $f } catch { "unavailable" } }
$o.cpu = T { @(Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed, L2CacheSize, L3CacheSize) }
$o.computer = T { Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model, TotalPhysicalMemory, PCSystemType }
$o.chassis = T { (Get-CimInstance Win32_SystemEnclosure).ChassisTypes }
$o.memory = T { @(Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity, Speed, ConfiguredClockSpeed, Manufacturer) }
$o.gpus = T { @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, DriverDate, AdapterRAM, VideoProcessor, CurrentHorizontalResolution, CurrentVerticalResolution, CurrentRefreshRate) }
$o.monitors_active = T { @(Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorBasicDisplayParams | Where-Object { $_.Active }).Count }
$o.os = T { Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, OSArchitecture, LastBootUpTime }
$o.os_display_version = T { (Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion").DisplayVersion }
$o.hw_gpu_scheduling = T { (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" -ErrorAction Stop).HwSchMode }
$o.power_mode_overlay = T { Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes" -ErrorAction Stop | Select-Object ActiveOverlayAcPowerScheme, ActiveOverlayDcPowerScheme }
$o.battery = T { @(Get-CimInstance Win32_Battery | Select-Object Name, EstimatedChargeRemaining, BatteryStatus) }
$o | ConvertTo-Json -Depth 4
"""

# powercfg 的設定（只查詢）：別名 → 說明
POWERCFG = {"PROCTHROTTLEMIN": ("SUB_PROCESSOR", "CPU 最低效能狀態 %"),
            "PROCTHROTTLEMAX": ("SUB_PROCESSOR", "CPU 最高效能狀態 %"),
            "PERFBOOSTMODE": ("SUB_PROCESSOR", "CPU 加速模式"),
            "ASPM": ("SUB_PCIEXPRESS", "PCIe 連結省電"),
            "LIDACTION": ("SUB_BUTTONS", "闔上螢幕的動作：0 不做事、1 睡眠、2 休眠、3 關機")}


def run_text(cmd, timeout=60):
    """執行 Windows 內建指令。中文輸出通常用系統的 OEM 字碼頁（台灣通常是 950），但有些電腦輸出 UTF-8
    （10/18：第 1 台的電源計畫名稱因此變亂碼）。所以先用 UTF-8 嚴格解碼，失敗才用 OEM 字碼頁"""
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        try:
            return r.stdout.decode("utf-8")
        except UnicodeDecodeError:
            pass
        cp = ctypes.windll.kernel32.GetOEMCP()
        try:
            return r.stdout.decode(f"cp{cp}")
        except Exception:
            return r.stdout.decode("utf-8", "replace")
    except Exception as e:
        return f"unavailable: {e}"


def tool_version(exe):
    """llama.cpp 的版本。它印在 stderr（10/18：原本只讀 stdout，所以第 1 台起都沒記到）"""
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, timeout=60, cwd=os.path.dirname(exe))
        txt = (r.stdout + r.stderr).decode("utf-8", "replace")
    except Exception as e:
        return f"unavailable: {e}"
    m = re.search(r"^version:.*$", txt, flags=re.M)
    return m.group(0).strip() if m else (txt.strip()[-200:] or None)


def gpu_process_memory(pid):
    """這個程式占用的顯示卡記憶體（GB，專用＋共用；內顯用的是共用記憶體）。讀不到回傳 None
    10/16 在 RTX 4050 量到：全部放進顯示卡 2.05、只放 10 層 0.81；Intel 內顯 2.29"""
    txt = powershell("$r = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | "
                     f"Where-Object {{ $_.Name -like 'pid_{pid}_*' }}; "
                     "if ($r) { '{0},{1}' -f ($r | Measure-Object DedicatedUsage -Sum).Sum, "
                     "($r | Measure-Object SharedUsage -Sum).Sum }", timeout=60)
    try:
        d, sh = (int(x) for x in txt.strip().splitlines()[-1].split(","))
        return round((d + sh) / 2 ** 30, 3)
    except (ValueError, IndexError):
        return None


def active_monitors():
    """作用中的螢幕數（Windows 的 WmiMonitorBasicDisplayParams）；關掉螢幕電源或闔上筆電時可能變少。讀不到回傳 None"""
    txt = powershell("@(Get-CimInstance -Namespace root\\wmi -ClassName WmiMonitorBasicDisplayParams | "
                     "Where-Object { $_.Active }).Count", timeout=30)
    try:
        return int(txt.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def collect_power():
    out = {"active_scheme": run_text(["powercfg", "/getactivescheme"]).strip()}
    for alias, (sub, desc) in POWERCFG.items():
        # /qh：連隱藏的設定也列出來（10/16：ASUS 的電源計畫把「闔上螢幕」藏起來，用 /q 讀不到）
        txt = run_text(["powercfg", "/qh", "SCHEME_CURRENT", sub, alias])
        vals = re.findall(r"0x([0-9a-fA-F]{8})\s*$", txt, flags=re.M)     # 最後兩個是 AC、DC 目前的值
        out[alias] = {"desc": desc, "ac": int(vals[-2], 16) if len(vals) >= 2 else None,
                      "dc": int(vals[-1], 16) if len(vals) >= 2 else None, "raw": txt.strip()[-400:]}
    return out


def collect_hw():
    hw = {}
    txt = powershell(HW_PS, timeout=180)
    try:
        hw = json.loads(txt)
    except Exception:
        hw = {"raw": txt}
    hw["power"] = collect_power()
    smi = nvidia_smi()
    if smi:
        q = subprocess.run([smi, "--query-gpu=index,name,driver_version,memory.total,power.limit,"
                            "clocks.max.graphics,clocks.max.memory", "--format=csv"],
                           capture_output=True, timeout=30)
        hw["nvidia_smi"] = q.stdout.decode("utf-8", "replace").strip()
    return hw


def nvidia_snapshot(run_dir, name):
    """NVIDIA 的完整狀態（含降頻原因、功耗上限、P-state）存成文字檔；不是 NVIDIA 就跳過。
    每張卡獨有的識別碼（GPU UUID、序號）存檔前遮掉（10/16 審閱指出）"""
    smi = nvidia_smi()
    if not smi:
        return
    try:
        r = subprocess.run([smi, "-q"], capture_output=True, timeout=60)
        txt = r.stdout.decode("utf-8", "replace")
        txt = re.sub(r"(?m)^([^:\r\n]*(?:UUID|Serial Number)[^:\r\n]*:[ \t]*)[^\r\n]*",
                     lambda m: m.group(1) + "<removed>", txt)
        open(os.path.join(run_dir, f"nvidia_smi_q_{name}.txt"), "w", encoding="utf-8").write(txt)
    except Exception:
        pass


def nvidia_smi():
    for p in (os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"),
              r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        if os.path.exists(p):
            return p
    return None


def as_list(x):
    """PowerShell 的 JSON：只有一項時可能不是清單"""
    if isinstance(x, list):
        return x
    return [x] if isinstance(x, dict) else []


def physical_cores(hw):
    try:
        return max(1, sum(int(c.get("NumberOfCores") or 0) for c in as_list(hw.get("cpu"))))
    except Exception:
        return max(1, (os.cpu_count() or 2) // 2)


# ---------------------------------------------------------------- 選顯示卡
def is_integrated(name):
    n = name.lower()
    if "nvidia" in n:
        return False
    if "intel" in n:                                   # Intel 只有 Arc A／B 系列是獨立顯示卡
        return not re.search(r"arc\(tm\) [ab]\d", n)
    if "radeon" in n or "amd" in n:                    # AMD 獨立顯示卡的名稱有 RX 或 Pro
        return not re.search(r"\b(rx|pro)\b", n)
    return True


def list_devices(server_exe):
    """llama.cpp 看得到的 Vulkan 裝置 [(Vulkan0, 名稱, MiB), ...] 和原始輸出"""
    r = subprocess.run([server_exe, "--list-devices"], capture_output=True, timeout=120, cwd=os.path.dirname(server_exe))
    out = (r.stdout + r.stderr).decode("utf-8", "replace")
    devs = [(m.group(1), m.group(2), int(m.group(3)))
            for m in re.finditer(r"^\s*(Vulkan\d+):\s*(.+?)\s*\((\d+) MiB", out, flags=re.M)]
    return devs, out


def find_device(server_exe, name, prefer):
    """照顯示卡名稱找它現在的編號；同名的有好幾張，就優先用原本的編號。找不到回傳 None
    （10/18：第 3 台的筆電螢幕關掉之後，內顯和獨顯的順序對調，照開始時的編號就跑到內顯上）"""
    try:
        devs, _ = list_devices(server_exe)
    except Exception:
        return None
    same = [d[0] for d in devs if d[1] == name]
    return (prefer if prefer in same else same[0]) if same else None


def device_in_log(log_path, dev):
    """伺服器紀錄開頭列出的裝置清單裡，編號 dev 對到的顯示卡名稱（讀不到回傳 None）"""
    try:
        txt = open(log_path, encoding="utf-8", errors="replace").read()
    except Exception:
        return None
    names = dict(re.findall(r"- (Vulkan\d+) : (.+?) \(\d+ MiB", txt))
    return names.get(dev, "").strip() or None


def pick_gpu(server_exe):
    """回傳 (裝置名稱如 Vulkan0, 顯示卡名稱, 是否為內顯, 全部清單文字)"""
    devs, out = list_devices(server_exe)
    if not devs:
        return None, None, None, out
    discrete = [d for d in devs if not is_integrated(d[1])]
    if discrete:
        d = max(discrete, key=lambda x: x[2])
        return d[0], d[1], False, out
    return devs[0][0], devs[0][1], True, out


# ---------------------------------------------------------------- 伺服器與請求
class Server:
    def __init__(self, exe, args, log_path, cwd=None):
        self.exe, self.args, self.log_path, self.cwd = exe, args, log_path, cwd
        self.proc, self.port = None, free_port()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout=240):
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        cmd = [self.exe] + self.args + ["--host", "127.0.0.1", "--port", str(self.port)]
        self.log.write("COMMAND: " + " ".join(cmd) + "\n")
        self.log.flush()
        self.proc = subprocess.Popen(cmd, cwd=self.cwd or os.path.dirname(self.exe), stdout=self.log, stderr=subprocess.STDOUT,
                                     creationflags=0x08000000)          # 不另開視窗
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                if urllib.request.urlopen(self.base + "/health", timeout=3).status == 200:
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()
        if getattr(self, "log", None):
            self.log.close()

    def request(self, prompt, max_tokens):
        """和論文的閒置測試完全相同的請求；回傳 (送出, 第一個 token, 完成) 的 perf_counter、牆上時間、token 數"""
        payload = {"model": "local", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0.0, "stream": True, "ignore_eos": True}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
                                     method="POST", headers={"Content-Type": "application/json",
                                                             "Authorization": "Bearer sk-no-key-required"})
        wall, t_send, t_first, n = time.time(), time.perf_counter(), None, 0
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                s = line[5:].strip()
                if s == "[DONE]":
                    break
                try:
                    delta = (json.loads(s).get("choices") or [{}])[0].get("delta") or {}
                except json.JSONDecodeError:
                    continue
                if delta.get("content"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    n += 1
        return t_send, t_first, time.perf_counter(), wall, n


def wait_until(t):
    while True:
        left = t - time.perf_counter()
        if left <= 0:
            return
        time.sleep(min(left, 0.01))


def offload_info(log_path):
    try:
        txt = open(log_path, encoding="utf-8", errors="replace").read()
    except Exception:
        return {}
    m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", txt)
    lines = [l.strip() for l in txt.splitlines() if re.search(r"Vulkan\d|offload|device|CPU_Mapped|buffer size", l)]
    return {"offloaded": m.group(1) if m else None, "layers": m.group(2) if m else None, "log_lines": lines[:40]}


# ---------------------------------------------------------------- 排除規則的檢查（10/15）
class _FT(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]


def sys_times():
    """整台電腦的 CPU 時間 (閒置, 全部)，單位 100 ns。在每段等待的前後各讀一次，算等待期間 CPU 有多忙"""
    i, k, u = _FT(), _FT(), _FT()
    try:
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u)):
            return None
    except Exception:
        return None
    idle, kern, user = ((x.hi << 32) | x.lo for x in (i, k, u))
    return idle, kern + user                   # 核心時間本來就包含閒置時間


def parse_wall(s):
    """紀錄檔裡的時間 → 和 time.time() 一樣的秒數
    nvidia-smi：2026/10/15 21:30:01.123（本地時間）；PowerShell 的 o 格式：2026-10-15T21:30:01.1234567+08:00"""
    m = re.match(r"(\d{4})[-/](\d\d)[-/](\d\d)[T ](\d\d):(\d\d):(\d\d)(\.\d+)?\s*(Z|[+-]\d\d:\d\d)?$", s.strip())
    if not m:
        return None
    try:
        y, mo, d, h, mi, se = (int(x) for x in m.groups()[:6])
        tz = m.group(8)
        if tz:
            off = 0 if tz == "Z" else (1 if tz[0] == "+" else -1) * (int(tz[1:3]) * 3600 + int(tz[4:6]) * 60)
            dt = datetime.datetime(y, mo, d, h, mi, se, tzinfo=datetime.timezone(datetime.timedelta(seconds=off)))
        else:
            dt = datetime.datetime(y, mo, d, h, mi, se)
        return dt.timestamp() + float(m.group(7) or 0)
    except (ValueError, OverflowError):
        return None


def gpu_util_in_waits(path, waits, ncols=None):
    """等待期間顯示卡的平均使用率（%）和樣本數；waits 是每段等待的 (開始, 結束) 牆上時間。
    這時候測試自己沒在用顯示卡，所以量到的使用率都來自別的程式。
    兩種紀錄檔的第 6 欄剛好都是使用率：nvidia-smi 的 utilization.gpu、monitor.ps1 的 gpu3d_sum"""
    wins = [(a + GUARD_START, b - GUARD_END) for a, b in waits if b - GUARD_END > a + GUARD_START]
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return None, 0
    vals = []
    for line in lines:
        p = [x.strip() for x in line.split(",")]
        if len(p) < 6 or (ncols and len(p) != ncols):          # 欄位數不對就跳過
            continue
        t = parse_wall(p[0])
        try:
            v = float(p[5])
        except ValueError:
            continue
        if t is not None and any(a <= t <= b for a, b in wins):
            vals.append(v)
    return (sum(vals) / len(vals) if vals else None), len(vals)


# ---------------------------------------------------------------- 狀態記錄
class Monitors:
    def __init__(self, mode, run_dir, rnd, nvidia):
        self.procs = []
        if nvidia:
            f = os.path.join(run_dir, f"round{rnd}_gpu_nvidia.csv")
            self.procs.append(subprocess.Popen(
                [nvidia, "--query-gpu=timestamp,index,pstate,clocks.gr,clocks.mem,utilization.gpu,"
                 "temperature.gpu,power.draw", "--format=csv,noheader,nounits", "-lms", "100", "-f", os.path.basename(f)],
                cwd=run_dir, creationflags=0x08000000))          # 只傳檔名：外部程式不一定能處理中文路徑
        ps1 = os.path.join(KIT, "scripts", "monitor.ps1")
        out = os.path.join(run_dir, f"round{rnd}_monitor.csv")
        self.procs.append(subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1, "-Mode", mode, "-Out", out],
            creationflags=0x08000000))

    def stop(self):
        for p in self.procs:
            try:
                p.kill()
                p.wait(timeout=10)
            except Exception:
                pass


def busy_check(nvidia):
    cpu = powershell("(Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation "
                     "-Filter \"Name='_Total'\").PercentProcessorUtility")
    gpu = ""
    if nvidia:
        r = subprocess.run([nvidia, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, timeout=30)
        gpu = r.stdout.decode("utf-8", "replace").strip().splitlines()[0] if r.stdout else ""
    try:
        cpu_v = float(cpu)
    except ValueError:
        cpu_v = None
    try:
        gpu_v = float(gpu)
    except ValueError:
        gpu_v = None
    return cpu_v, gpu_v


# ---------------------------------------------------------------- 打包
def sanitize_text(txt):
    """遮掉測試資料夾的路徑、使用者名稱、電腦名稱。
    先換最長的路徑（它通常包含使用者名稱，先換使用者名稱的話路徑就對不上了）；
    JSON 裡的反斜線是兩個，也要換；不分大小寫（10/16 試跑發現漏遮）"""
    for v, tag in ((ROOT, "<TESTFOLDER>"), (os.environ.get("USERPROFILE", ""), "<USERPROFILE>"),
                   (os.environ.get("USERNAME", ""), "<USERNAME>"), (os.environ.get("COMPUTERNAME", ""), "<COMPUTERNAME>")):
        if v and len(v) >= 2:
            for form in (v.replace("\\", "\\\\"), v, v.replace("\\", "/")):
                txt = re.sub(re.escape(form), lambda m: tag, txt, flags=re.I)
    return txt


def short_name(name, kind):
    if not name:
        return kind
    if kind == "CPU":
        m = re.search(r"(i\d-\d{4,5}\w*|Ryzen \d \d{4}\w*|Ultra \d \d{3}\w*)", name)
        s = m.group(1) if m else name
    else:
        s = re.sub(r"\((R|TM)\)", "", name)
        s = re.sub(r"\b(NVIDIA|GeForce|AMD|Intel|Graphics)\b", "", s, flags=re.I)
        s = s.replace("Laptop GPU", "Laptop")
    s = re.sub(r"[^A-Za-z0-9]+", "-", s.strip()).strip("-")
    return s[:40] or kind


def package(run_dir, label, unfinished):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    name = f"閒置測試結果_{label}_{stamp}{'_未完成' if unfinished else ''}.zip"
    dest = os.path.join(desktop_dir() or ROOT, name)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(run_dir)):
            if fn == "packaged.txt":
                continue
            p = os.path.join(run_dir, fn)
            if fn.endswith((".log", ".json", ".txt")):
                z.writestr(fn, sanitize_text(open(p, encoding="utf-8", errors="replace").read()))
            else:
                z.write(p, fn)
    open(os.path.join(run_dir, "packaged.txt"), "w", encoding="utf-8").write(os.path.basename(dest))
    return dest


def package_leftovers():
    """上次如果被中斷（例如半夜 Windows 更新重開機），先把還沒打包的結果補包到桌面"""
    out = []
    if not os.path.isdir(WORK):
        return out
    for d in sorted(os.listdir(WORK)):
        run_dir = os.path.join(WORK, d)
        if not os.path.isdir(run_dir) or os.path.exists(os.path.join(run_dir, "packaged.txt")):
            continue
        label = d.split("_")[0].upper()
        try:
            label = json.load(open(os.path.join(run_dir, "run_info.json"), encoding="utf-8")).get("label", label)
        except Exception:
            pass
        try:
            out.append(package(run_dir, label, True))
        except Exception:
            pass
    return out


def shutdown_countdown(seconds=120):
    """睡前跑：結果都存好之後關機；倒數期間按任意鍵可以取消（試跑模式只倒數，不會真的關機）"""
    import msvcrt
    if QUICK:
        seconds = 15
        say(f"（試跑模式：不會真的關機）倒數 {seconds} 秒，按任意鍵可以取消。")
    else:
        subprocess.run(["shutdown", "/s", "/t", str(seconds), "/c",
                        "LLM 閒置測試完成，結果已經存到桌面。電腦將在 2 分鐘後關機。"], capture_output=True)
        say(f"電腦將在 {seconds // 60} 分鐘後關機。醒著的話，按任意鍵可以取消。")
    t0 = time.time()
    while time.time() - t0 < seconds:
        if msvcrt.kbhit():
            msvcrt.getwch()
            if not QUICK:
                subprocess.run(["shutdown", "/a"], capture_output=True)
            say("已取消關機。")
            return
        time.sleep(0.2)
    if QUICK:
        say("（試跑模式：倒數結束。正式跑的話，這時候電腦會關機。）")


class _Skip(Exception):
    """這台不能跑這個測試：只打包硬體資訊"""


def run_test(mode, rounds, interactive=True, open_folder=True):
    """跑一個測試（gpu 或 cpu）。回傳 (結果 zip 路徑, 是否全部跑完, 是否因為這台不能跑而跳過)"""
    global SLEPT
    need = [os.path.join(KIT, "model", "qwen2.5-coder-3b-instruct-q4_k_m.gguf"),
            os.path.join(KIT, "llama-vulkan" if mode == "gpu" else "llama-cpu", "llama-server.exe")]
    if not all(os.path.exists(p) for p in need):
        say("找不到測試需要的檔案。請先把下載的 zip 整個解壓縮，再到解壓縮後的資料夾裡執行。")
        if interactive:
            pause()
        return None, False, False

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(WORK, f"{mode}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    info = {"kit_version": KIT_VERSION, "mode": mode, "start": datetime.datetime.now().isoformat(),
            "gaps": GAPS, "reps": REPS, "blocks": BLOCKS, "rounds": rounds, "warmup": WARMUP, "rounds_done": 0,
            "no_monitor_rounds": [r for r in range(1, rounds + 1) if r % 5 == 0], "notes": []}
    server, mon, unfinished, label, dest = None, None, True, mode.upper(), None
    fh, skipped = None, False
    keep_awake(True)
    try:
        # ---- 電源
        has_batt, ac = on_ac_power()
        info["laptop_battery"] = has_batt
        if has_batt and not ac:
            if interactive:
                while not ac:
                    say("偵測到筆電沒有插電。請插上電源（每 3 秒檢查一次；要放棄請關掉這個視窗）。")
                    time.sleep(3)
                    has_batt, ac = on_ac_power()
            else:
                info["notes"].append("laptop on battery at start")

        # ---- 硬體資訊
        say("記錄硬體資訊……")
        hw = collect_hw()
        json.dump(hw, open(os.path.join(run_dir, "hardware.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        nvidia = None
        lid = ((hw.get("power") or {}).get("LIDACTION") or {}).get("ac") if has_batt else None
        info["lid_action_ac"] = lid
        if lid == 0:
            say("這台筆電闔上螢幕也會繼續跑。")
        elif lid in (1, 2, 3):
            say("注意：這台筆電闔上螢幕就會" + {1: "睡眠", 2: "休眠", 3: "關機"}[lid] +
                "，測試會停住。請不要闔上，把亮度調到最低就好。")

        # ---- 選裝置、伺服器參數
        if mode == "gpu":
            exe = need[1]
            dev, dev_name, integrated, listing = pick_gpu(exe)
            open(os.path.join(run_dir, "devices.txt"), "w", encoding="utf-8").write(listing)
            if not dev:
                say("這台電腦找不到可以用的 GPU（Vulkan），GPU 測試沒辦法跑。")
                say("請把這個視窗截圖，連同桌面上的 GPU 結果檔，一起傳給請你幫忙測試的人。llama.cpp 的訊息：")
                for line in [x for x in listing.splitlines() if x.strip() and not re.search(r"[A-Za-z]:\\", x)][-8:]:
                    say("  " + line[:150])
                info["notes"].append("no Vulkan device")
                raise _Skip()
            info.update(device=dev, device_name=dev_name, integrated=integrated)
            label = f"GPU_{short_name(dev_name, 'GPU')}" + ("_內顯" if integrated else "")
            say(("沒有找到獨立顯示卡，改用內顯：" if integrated else "使用的顯示卡：") + dev_name)
            if "nvidia" in dev_name.lower():
                nvidia = nvidia_smi()
            threads = 4
            args = ["-m", os.path.basename(need[0]), "-c", "4096", "--parallel", "1", "-t", str(threads), "--cache-ram", "0",
                    "--device", dev, "-ngl", "all"]
        else:
            exe = need[1]
            threads = physical_cores(hw)
            cpu_name = (as_list(hw.get("cpu")) or [{}])[0].get("Name", "") or ""
            label = f"CPU_{short_name(cpu_name, 'CPU')}"
            say(f"使用的 CPU：{cpu_name}（{threads} 個執行緒）")
            args = ["-m", os.path.basename(need[0]), "-c", "4096", "--parallel", "1", "-t", str(threads), "--cache-ram", "0"]
        info.update(threads=threads, label=label)
        probe_full, probe_mem = False, None
        if mode == "gpu":
            # 這版 llama.cpp 平常不寫出放了幾層，要開詳細紀錄（-lv 4）才會寫。
            # 正式量測不開（和論文相同），只在開始前用詳細紀錄啟動一次，確認放了幾層、占多少顯示卡記憶體
            say("確認模型能不能全部放進顯示卡……")
            probe_log = os.path.join(run_dir, "placement_probe.log")
            probe = Server(exe, args + ["-lv", "4"], probe_log, cwd=os.path.dirname(need[0]))
            probe_ok = probe.start()
            probe_mem = gpu_process_memory(probe.proc.pid) if probe_ok else None
            probe.stop()
            pr = offload_info(probe_log)
            pr.update(started=probe_ok, gpu_mem_gb=probe_mem)
            info["placement_probe"] = pr
            probe_full = bool(pr.get("offloaded")) and pr["offloaded"] == pr.get("layers")
            if probe_ok and pr.get("offloaded") and not probe_full:
                say(f"注意：這張顯示卡的記憶體放不下整個模型（只放得下 {pr['offloaded']}/{pr['layers']} 層），"
                    "GPU 測試照樣跑，結果只當參考。")
        info["llama_version"] = tool_version(exe)
        info["python_version"] = sys.version
        nvidia_snapshot(run_dir, "start")

        # ---- 開跑前檢查電腦是不是在忙
        cpu_u, gpu_u = busy_check(nvidia)
        info["busy_before"] = {"cpu_percent": cpu_u, "gpu_percent": gpu_u}
        if (cpu_u is not None and cpu_u > 30) or (gpu_u is not None and gpu_u > 15):
            if interactive:
                say(f"電腦好像還在忙（CPU {cpu_u}%、GPU {gpu_u}%）。請關掉其他程式。")
                pause("關好之後按 Enter 繼續……")
            else:
                info["notes"].append(f"busy at start: CPU {cpu_u}%, GPU {gpu_u}%")

        workload = "main" if mode == "gpu" else "short"
        reps = REPS
        fields = ["round", "req_id", "block", "gap", "idle_before", "t_send", "wall_send", "ttft", "service",
                  "out_tokens", "workload", "monitor", "cpu_wait", "error", "stall"]
        fh = open(os.path.join(run_dir, "requests.csv"), "w", newline="", encoding="utf-8")   # 每筆即時寫入
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        info["rounds_info"] = []
        t_begin = time.time()

        # 每一輪記不記錄狀態：第 5、10 輪不記錄，當對照。
        # 某一輪不算的話，在最後加一輪補上（記錄與否和被補的那輪一樣），最多補原定輪數的一半：5 輪 → 3 輪，10 輪 → 5 輪
        plan = [r % 5 != 0 for r in range(1, rounds + 1)]
        cap = (rounds + 1) // 2
        makeups, full_seen, stopped, rnd = 0, probe_full, False, 0
        while rnd < len(plan):
            rnd += 1
            monitor_on = plan[rnd - 1]
            tag = f"第 {rnd} 輪" + ("（補跑）" if rnd > rounds else "")
            say()
            say(f"{'GPU' if mode == 'gpu' else 'CPU'} 測試{tag}：啟動伺服器……")
            log_path = os.path.join(run_dir, f"round{rnd}_server.log")
            if mode == "gpu":
                # 每一輪照名稱重新找顯示卡的編號。找不到就照舊，這輪結束時會從伺服器紀錄檢查出來
                now = find_device(exe, dev_name, dev)
                if now and now != dev:
                    info["notes"].append(f"round {rnd}: {dev_name} moved from {dev} to {now}")
                    say(f"  顯示卡的順序變了，改用 {now}（{dev_name}）。")
                    dev = now
                    args[args.index("--device") + 1] = dev
            # 模型只傳檔名、工作資料夾設在模型那一層：llama.cpp 在 Windows 上打不開含中文的路徑（10/16 試跑發現）
            server = Server(exe, args, log_path, cwd=os.path.dirname(need[0]))
            if not server.start():
                info["notes"].append(f"round {rnd}: server failed to start")
                say("伺服器沒有啟動成功，停止這個測試（已經跑完的部分還是會打包）。")
                stopped = True
                break
            rinfo = {"round": rnd, "makeup": rnd > rounds, "monitor": int(monitor_on),
                     "gpu_mem_gb": gpu_process_memory(server.proc.pid) if mode == "gpu" else None,
                     "monitors_active_start": active_monitors()}      # 螢幕開或關：每輪開始、結束各記一次
            if mode == "gpu":
                rinfo["device"] = dev
            placement_ok, placement = True, None
            if mode == "gpu":
                if info["placement_probe"].get("offloaded") and not probe_full:
                    placement_ok, placement = False, "partial"          # 這張卡本來就放不下
                elif probe_full and probe_mem and rinfo["gpu_mem_gb"] is not None:
                    placement_ok = rinfo["gpu_mem_gb"] >= (1 - PLACEMENT_DROP) * probe_mem
                    placement = "full" if placement_ok else "partial"   # 少 5% 以上：這次放得比較少
                else:
                    placement = "unknown"       # 確認不了：記下來，分析時用服務時間檢查（排除規則第 1 點）
            rinfo["placement"] = placement
            if not placement_ok:
                info["notes"].append(f"round {rnd}: model only partly on GPU")
            errors, on_batt, waits, stalls = [], False, [], []
            cpu_idle = cpu_all = 0

            prompt_file, max_tokens = WORKLOADS[workload]
            prompt = open(os.path.join(KIT, "prompts", prompt_file), encoding="utf-8").read()
            try:
                warm = []
                for i in range(WARMUP):                                 # 暖機，不記錄
                    ts, tf, td, wall, n = server.request(f"[warmup {rnd}-{i}] " + prompt, max_tokens)
                    warm.append(td - ts)
                rinfo["warmup_service"] = [round(x, 3) for x in warm]
                if rnd == 1:
                    med = statistics.median(warm)
                    if mode == "gpu" and workload == "main" and med > SLOW_GPU_S:
                        workload = "short"
                        info["notes"].append(f"slow device ({med:.1f} s per request): switched to the short workload")
                        say("這台跑得比較慢，改用比較短的請求（和論文裡較慢的裝置一樣）。")
                        prompt_file, max_tokens = WORKLOADS[workload]
                        prompt = open(os.path.join(KIT, "prompts", prompt_file), encoding="utf-8").read()
                        for i in range(WARMUP):
                            server.request(f"[warmup {rnd}-s{i}] " + prompt, max_tokens)
                    if mode == "cpu" and med > SLOW_CPU_S:
                        reps = 2
                        info["notes"].append(f"slow CPU ({med:.1f} s per request): reps reduced to 2")
            except Exception as e:
                errors.append(f"warmup: {type(e).__name__}: {e}"[:200])
            info["workload"], info["reps_used"] = workload, reps

            mon = Monitors(mode, run_dir, rnd, nvidia) if monitor_on and not errors else None
            rng = random.Random(2026 + rnd)
            n_req, done = BLOCKS * len(GAPS) * reps, 0
            last_end = time.perf_counter()
            last_wall = time.time()                                     # 上一個請求結束的牆上時間（偵測睡著用）
            t_round = time.time()
            for b in range(1, BLOCKS + 1):
                if errors:                                              # 出了問題：這輪停下來，最後補跑
                    break
                has_batt, ac = on_ac_power()
                on_batt = on_batt or (has_batt and not ac)
                order = GAPS * reps
                rng.shuffle(order)
                for g in order:
                    st0 = sys_times() if g >= 2 else None               # 等待前後各讀一次 CPU 時間
                    wait_until(last_end + g)
                    st1 = sys_times() if st0 else None
                    cpu_w = ""
                    if st0 and st1 and st1[1] > st0[1]:
                        cpu_idle += st1[0] - st0[0]
                        cpu_all += st1[1] - st0[1]
                        cpu_w = round(100 * (1 - (st1[0] - st0[0]) / (st1[1] - st0[1])), 1)
                    rid = (rnd - 1) * 1000 + done
                    row = dict(round=rnd, req_id=rid, block=b, gap=g, workload=workload, monitor=int(monitor_on),
                               cpu_wait=cpu_w, error="")
                    try:
                        ts, tf, td, wall, n = server.request(f"[request {rid:05d}] " + prompt, max_tokens)
                    except Exception as e:
                        row["error"] = f"{type(e).__name__}: {e}"[:200]
                        errors.append(f"request {rid}: {row['error']}")
                        writer.writerow(row)
                        fh.flush()
                        break
                    wall_end = time.time()
                    # 睡著或卡住：等待比設定的久、牆上時間和程式計時對不上，或一個請求久得不正常
                    stall = max((ts - last_end) - g, (wall - last_wall) - g, (wall_end - wall) - (td - ts))
                    if stall > STALL_MAX or td - ts > LONG_REQ_S:
                        stalls.append(rid)
                    row.update(idle_before=round(ts - last_end, 4), t_send=round(ts, 4), wall_send=round(wall, 4),
                               ttft=round(tf - ts, 4) if tf else "", service=round(td - ts, 4), out_tokens=n,
                               stall=round(stall, 3))
                    writer.writerow(row)
                    fh.flush()
                    if n == 0:
                        errors.append(f"request {rid}: no output tokens")
                    if g >= 2:
                        waits.append((wall - (ts - last_end), wall))    # 這段等待的開始和結束（牆上時間）
                    last_end, last_wall = td, wall_end
                    done += 1
                    if done % 10 == 0 or done == n_req:
                        per = (time.time() - t_round) / done
                        left = per * (n_req - done) + (len(plan) - rnd) * (per * n_req + 30)
                        say(f"  {tag}：{done}/{n_req}，這個測試大約還要 {left / 60:.0f} 分鐘")
            if mon:
                mon.stop()
                mon = None
            has_batt, ac = on_ac_power()
            on_batt = on_batt or (has_batt and not ac)
            nvidia_snapshot(run_dir, f"round{rnd}_end")
            server.stop()
            server = None
            rinfo["monitors_active_end"] = active_monitors()

            # ---- 這一輪算不算（排除規則第 1 點）；分析時會用同樣的紀錄再算一次
            reasons = []
            if not placement_ok:
                reasons.append("placement")
            if mode == "gpu":
                rinfo["device_used"] = device_in_log(log_path, dev)          # 伺服器紀錄裡這個編號對到的顯示卡
                if rinfo["device_used"] and rinfo["device_used"] != dev_name:
                    reasons.append("wrong_device")
            if errors or done < n_req:
                reasons.append("request_failed")
            if on_batt:
                reasons.append("battery")
            if stalls:
                reasons.append("sleep")
                rinfo["stalled_requests"] = stalls[:20]
                SLEPT = True
            rinfo["cpu_wait_percent"] = round(100 * (1 - cpu_idle / cpu_all), 1) if cpu_all else None
            if rinfo["cpu_wait_percent"] is not None and rinfo["cpu_wait_percent"] > CPU_BUSY_MAX:
                reasons.append("cpu_busy")
            if mode == "gpu" and monitor_on:
                try:
                    src = f"round{rnd}_gpu_nvidia.csv" if nvidia else f"round{rnd}_monitor.csv"
                    u, k = gpu_util_in_waits(os.path.join(run_dir, src), waits, 8 if nvidia else 6)
                    rinfo["gpu_wait_percent"] = round(u, 1) if u is not None else None
                    rinfo["gpu_wait_samples"] = k
                    if u is not None and u > GPU_BUSY_MAX:
                        reasons.append("gpu_busy")
                    elif k < MIN_GPU_SAMPLES:
                        rinfo["gpu_check"] = "insufficient"             # 監控樣本太少，無法判定
                except Exception:
                    rinfo["gpu_check_error"] = traceback.format_exc()[-300:]
            if FORCE_INVALID and rnd == 1:
                reasons.append("forced_test")
            rinfo.update(valid=not reasons, invalid_reasons=reasons, errors=errors[:10], requests_done=done)
            full_seen = full_seen or (mode == "gpu" and placement_ok)
            if reasons:
                why = "、".join(REASON_ZH.get(r, r) for r in reasons)
                # 模型放不進顯示卡：之前有一輪放得進去才補跑；從來沒放進去過，代表顯示卡記憶體不夠，補跑也沒用
                structural = "placement" in reasons and not full_seen
                if makeups < cap and not structural:
                    plan.append(monitor_on)
                    makeups += 1
                    say(f"  這一輪不算（{why}），最後會多跑一輪補上。")
                else:
                    say(f"  這一輪不算（{why}）。" + ("補跑的次數已經用完。" if makeups >= cap else ""))
            info["rounds_info"].append(rinfo)
            info.update(rounds_done=rnd, rounds_planned=rounds, makeup_cap=cap, makeup_rounds=makeups,
                        valid_rounds=sum(1 for x in info["rounds_info"] if x.get("valid")),
                        no_monitor_rounds=[i + 1 for i, m in enumerate(plan) if not m])
            if rnd < len(plan):
                time.sleep(10)
        fh.close()
        unfinished = stopped
        info["minutes"] = round((time.time() - t_begin) / 60, 1)
    except _Skip:
        skipped = True
    except KeyboardInterrupt:
        info["notes"].append("interrupted by user")
    except Exception:
        info["notes"].append("error: " + traceback.format_exc())
        say("發生錯誤，已經跑完的部分還是會打包。")
    finally:
        if fh and not fh.closed:
            fh.close()
        if mon:
            mon.stop()
        if server:
            server.stop()
        keep_awake(False)
        nvidia_snapshot(run_dir, "end")
        info["end"] = datetime.datetime.now().isoformat()
        try:
            json.dump(info, open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            dest = package(run_dir, label, unfinished)
            say()
            say("=" * 60)
            say("這台電腦不能跑這個測試，硬體資訊已經存到桌面：" if skipped else
                "完成！結果已經存到桌面：" if not unfinished else "測試沒有全部跑完，已經跑完的部分存到桌面：")
            say("  " + os.path.basename(dest))
            say("請把這個檔案傳給請你幫忙測試的人（LINE 或 email 都可以）。")
            say("=" * 60)
            if open_folder:
                subprocess.Popen(["explorer", "/select,", dest])
        except Exception:
            say("打包結果時出了問題，請把這個視窗截圖傳給請你幫忙測試的人。")
            say(traceback.format_exc())
    return dest, not unfinished, skipped


# ---------------------------------------------------------------- 主程式
def main():
    global BLOCKS, ROUNDS_DAY, ROUNDS_NIGHT, FORCE_INVALID, QUICK
    arg = (sys.argv[1] if len(sys.argv) > 1 else "gpu").lower()
    if arg not in ("gpu", "cpu", "night"):
        sys.exit("用法：idle_test.py gpu|cpu|night [--quick]")
    if "--quick" in sys.argv:            # 只給作者試跑用：每個測試 1 輪、15 個請求（BAT 不會用到）
        BLOCKS, ROUNDS_DAY, ROUNDS_NIGHT = 1, 1, 1
        FORCE_INVALID = "--force-invalid" in sys.argv     # 試跑補跑：第 1 輪當成不算
        QUICK = True
    disable_quick_edit()
    say("=" * 60)
    if arg == "night":
        say(f"  LLM 閒置測試：睡前跑（GPU {ROUNDS_NIGHT} 輪 → CPU {ROUNDS_NIGHT} 輪，約 2–4 小時）")
    else:
        say(f"  LLM {'GPU' if arg == 'gpu' else 'CPU'} 閒置測試（{ROUNDS_DAY} 輪，約 40–60 分鐘）")
    say("=" * 60)
    say("跑的時候請：")
    say("  1. 關掉遊戲、影片、瀏覽器、動態桌布（例如 Wallpaper Engine）等其他程式（有沒存的檔案請先存好）")
    say("  2. 筆電要插著電源，最好不要闔上螢幕（等一下會告訴你這台闔上會不會睡著）；")
    say("     螢幕會一直亮著，可以把亮度調低。桌機可以直接關掉螢幕。")
    say("  3. 不要點這個視窗，也不要用電腦")
    if arg == "night":
        say("  4. 全部跑完、結果存到桌面後，電腦會在 2 分鐘後【自動關機】。")
        say("     關機時其他程式會被強制關閉，所以請先存檔、關掉其他程式。")
    say("收集的只有時間紀錄、GPU／CPU 狀態和硬體型號，使用者名稱和電腦名稱會遮掉。")
    say("如果有幾輪被干擾（例如背景有程式在跑），程式會自動多跑幾輪補上，時間會久一點。")
    say()
    pause("準備好了按 Enter 開始……")

    left = package_leftovers()
    if left:
        say("先把上次沒跑完的結果存到桌面：" + "、".join(os.path.basename(p) for p in left))

    if arg == "night":
        g, g_done, g_skip = run_test("gpu", ROUNDS_NIGHT, interactive=True, open_folder=False)
        c, c_done, c_skip = run_test("cpu", ROUNDS_NIGHT, interactive=False, open_folder=False)
        # 兩個測試都完整跑完（這台沒有 GPU 而跳過也算）、而且沒睡著過，才自動關機；
        # 否則留著視窗，朋友早上可以看到發生什麼事（10/16 審閱指出：原本只要有結果檔就會關機）
        all_done = (g_done or g_skip) and (c_done or c_skip) and not (g_skip and c_skip)
        if (g or c) and SLEPT:
            say("電腦中途睡著過（例如闔上了筆電），為了安全，這次不自動關機。結果已經存到桌面。")
            pause()
        elif (g or c) and not all_done:
            say("有測試沒有全部跑完，為了安全，這次不自動關機。已經跑完的部分存到桌面了，")
            say("請把桌面上的結果 zip 和這個視窗的截圖傳給請你幫忙測試的人。")
            pause()
        elif g or c:
            shutdown_countdown()
        else:
            say("兩個測試都沒有產生結果，所以不關機。請把這個視窗截圖傳給請你幫忙測試的人。")
            pause()
    else:
        run_test(arg, ROUNDS_DAY)


if __name__ == "__main__":
    main()
