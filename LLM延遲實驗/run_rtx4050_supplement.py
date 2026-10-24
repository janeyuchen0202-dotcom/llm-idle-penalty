"""RTX 4050 補充實驗 A、B 一次跑完（2026-10-21；設計：RTX4050補充實驗_設計草稿_20261021.md；判定標準：事前預期.md 的事前登記）
開跑前檢查 → 記下使用者的預測 → CPU 設定改回實驗用 → A（約 2 小時）→ B（約 3 小時）→ 分析 → 還原

開跑前要自己做的（程式改不了，只能提醒）：
- Armoury Crate 切到 Performance（Fn＋F5），風扇跟著模式（不要手動設定）
- 接外接螢幕（和 10/19 相同）、插電、關掉 Wallpaper Engine／影片／遊戲
- 用遠端桌面啟動的話，開始後 10 分鐘內斷線
程式會做的：
1. 檢查：系統管理員、插電、電源計畫是 Performance（27fa6203…）、沒有 Wallpaper Engine、沒有別的 llama-server、
   模型的 SHA-256、兩個 llama.cpp 都是 b9549、螢幕
2. 記下使用者的預測（原文、時間）到 事前預期.md
3. CPU 設定（插電）改回實驗用：最低 100%、最高 100%、大核上限 100%、加速模式 2（積極）；原本的值先存起來，最後一定改回去；
   改完確認有負載時 % Processor Performance 超過 100%（Turbo 有開）
4. 等顯示卡閒置降頻（遠端桌面要斷線）→ 實驗 A（run_backend_probe.py）→ 分析（analyze_backend_probe.py）
5. 實驗 B（run_keepwarm5.py，自己會再檢查一次）→ 分析（analyze_keepwarm5.py）
6. 還原：時脈放開、CPU 設定改回開跑前的值
畫面上的所有輸出另存到 out/supplement_<時間>_console.txt

用法（系統管理員 PowerShell，在 LLM延遲實驗 資料夾）：python run_rtx4050_supplement.py
      只跑一個：--only A ／ --only B
      試流程：--dry（假伺服器、請求數變少、不改 CPU 設定、預測寫到 out/ 的測試檔；可以加 --pred-a "…" --pred-b "…"）
      視窗被關掉、沒有還原的話：python run_rtx4050_supplement.py --restore（照最近一次存的值改回 CPU 設定、放開時脈）
"""
import ctypes
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
DRY, RESTORE = "--dry" in sys.argv, "--restore" in sys.argv
ONLY = sys.argv[sys.argv.index("--only") + 1].upper() if "--only" in sys.argv else "AB"
KW5 = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "kw5"     # 10/22 加：B 重跑時的檔名（例如 kw5m），不蓋掉之前的
STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M")
PFX = "supplementq" if DRY else "supplement"
UP = os.environ["USERPROFILE"]
PERF_GUID = "27fa6203-3987-4dcc-918d-748559d549ec"          # Armoury Crate 的 Performance 模式（10/19 起實驗都用這個）
MODEL = os.path.join(UP, "LLM", "models", "qwen2.5-coder-3b-instruct-q4_k_m.gguf")
MODEL_SHA = "724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7"   # 和論文、測試包相同（前 16 碼 10/14 記錄）
EXES = {"CUDA": os.path.join(UP, "LLM-cuda-b9549", "bin", "llama-server.exe"), "Vulkan": os.path.join(UP, "LLM", "llama-server.exe")}
VERSION = "version: 9549 (04eb4c446)"
CPU_KEYS = ["PROCTHROTTLEMIN", "PROCTHROTTLEMAX", "PROCTHROTTLEMAX1", "PERFBOOSTMODE"]
CPU_EXPERIMENT = {"PROCTHROTTLEMIN": 100, "PROCTHROTTLEMAX": 100, "PROCTHROTTLEMAX1": 100, "PERFBOOSTMODE": 2}
PREREG_MARK = "## RTX 4050 補充實驗 A、B 的事前登記"
WALLPAPER = {"wallpaper32.exe", "wallpaper64.exe", "webwallpaper32.exe"}
# 10/21 試流程時查到：這些也在用顯示卡（閒置時使用率 24%），看到就提醒
GPU_APPS = {"steam.exe": "Steam", "discord.exe": "Discord", "chrome.exe": "Chrome", "msedge.exe": "Edge", "obs64.exe": "OBS"}


class Tee:
    """畫面上的輸出同時寫進檔案"""
    def __init__(self, path):
        self.out, self.f = sys.stdout, open(path, "a", encoding="utf-8")

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.out.flush()
        self.f.flush()


def say(m=""):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def keep_awake(on):
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x1 | 0x2 if on else 0))


def disable_quick_edit():
    """關掉主控台的「快速編輯」：不然點一下視窗，畫面輸出就會卡住，實驗程式也會跟著卡住（和測試包相同）"""
    try:
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-10)
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            k32.SetConsoleMode(h, (mode.value | 0x0080) & ~0x0040)
    except Exception:
        pass


def run_text(cmd, timeout=60):
    """Windows 內建指令的輸出：先用 UTF-8 解碼，失敗才用系統的 OEM 字碼頁（和測試包相同）"""
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except Exception as e:
        return f"unavailable: {e}"
    try:
        return r.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return r.stdout.decode(f"cp{ctypes.windll.kernel32.GetOEMCP()}", "replace")


def powershell(cmd, timeout=60):
    return run_text(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
                     "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + cmd], timeout).strip()


def smi(args, timeout=30):
    try:
        r = subprocess.run(["nvidia-smi"] + args, capture_output=True, timeout=timeout)
        return r.returncode, r.stdout.decode("utf-8", "replace")
    except Exception as e:
        return -1, str(e)


def on_ac_power():
    class SPS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte), ("BatteryLifePercent", ctypes.c_byte),
                    ("SystemStatusFlag", ctypes.c_byte), ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    s = SPS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        return True
    return (s.ACLineStatus & 0xFF) == 1


def processes():
    txt = run_text(["tasklist", "/FO", "CSV", "/NH"])
    return {m.group(1).lower() for m in re.finditer(r'^"([^"]+)"', txt, flags=re.M)}


# ---------------------------------------------------------------- CPU 設定
def read_cpu():
    vals = {}
    for k in CPU_KEYS:
        txt = run_text(["powercfg", "/qh", "SCHEME_CURRENT", "SUB_PROCESSOR", k])
        v = re.findall(r"0x([0-9a-fA-F]{8})\s*$", txt, flags=re.M)     # 最後兩個是插電、電池目前的值
        vals[k] = {"ac": int(v[-2], 16), "dc": int(v[-1], 16)} if len(v) >= 2 else None
    return vals


def set_cpu_ac(target):
    for k, v in target.items():
        subprocess.run(["powercfg", "/setacvalueindex", "SCHEME_CURRENT", "SUB_PROCESSOR", k, str(v)], capture_output=True)
    subprocess.run(["powercfg", "/setactive", "SCHEME_CURRENT"], capture_output=True)
    now = read_cpu()
    return now, [k for k, v in target.items() if (now.get(k) or {}).get("ac") != v]


def fmt_cpu(vals):
    return "、".join(f"{k} {v['ac']}" if v else f"{k} 讀不到" for k, v in vals.items())


def processor_performance(n_load=2, samples=5):
    """兩個忙碌的程序跑著的時候，\\Processor Information(_Total)\\% Processor Performance 的最大值（超過 100 = Turbo 有開）"""
    procs = [subprocess.Popen([sys.executable, "-c", "while True: pass"], creationflags=0x08000000) for _ in range(n_load)]
    try:
        time.sleep(1.0)
        txt = powershell(f"$v = @(); 1..{samples} | ForEach-Object {{ $v += (Get-CimInstance "
                         "Win32_PerfFormattedData_Counters_ProcessorInformation -Filter \"Name='_Total'\").PercentProcessorPerformance; "
                         "Start-Sleep -Milliseconds 800 }; $v -join ','", timeout=90)
    finally:
        for p in procs:
            p.kill()
    vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", txt.splitlines()[-1])] if txt else []
    return max(vals) if vals else None


def restore_cpu(saved):
    target = {k: v["ac"] for k, v in saved.items() if v}
    now, bad = set_cpu_ac(target)
    say(f"CPU 設定改回開跑前的值：{fmt_cpu(now)}" + (f"（沒改成功：{bad}，請把這段訊息給 Claude）" if bad else ""))


def unlock_clocks():
    a, _ = smi(["-rgc"])
    b, _ = smi(["-rmc"])
    say("時脈放開（nvidia-smi -rgc、-rmc）" + ("" if a == b == 0 else "：有一個指令失敗，請把這段訊息給 Claude"))


# ---------------------------------------------------------------- 開跑前
def checks():
    say("== 自動檢查")
    problems = []
    admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
    say(f"  系統管理員：{'是' if admin else '不是'}")
    if not admin and not DRY:
        problems.append("請用「以系統管理員身分執行」打開 PowerShell 再執行（B 要鎖時脈、改 CPU 設定也要）")
    ac = on_ac_power()
    say(f"  插電：{'是' if ac else '不是'}")
    if not ac:
        problems.append("請插上電源")
    m = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", run_text(["powercfg", "/getactivescheme"]), re.I)
    guid = m.group(1).lower() if m else None
    say(f"  電源計畫：{guid}" + ("（Performance）" if guid == PERF_GUID else ""))
    if guid != PERF_GUID:
        problems.append("電源計畫不是 Performance：請用 Armoury Crate（或 Fn＋F5）切到 Performance")
    ps = processes()
    if ps & WALLPAPER:
        problems.append("Wallpaper Engine 還開著，請關掉（右下角圖示 → 退出）")
    if "llama-server.exe" in ps:
        problems.append("已經有 llama-server 在跑，請先關掉")
    if "remoting_desktop.exe" in ps:
        say("  注意：遠端桌面現在連著。開跑後 10 分鐘內要斷線（程式會等顯示卡降頻才開始）")
    apps = [v for k, v in GPU_APPS.items() if k in ps]
    if apps:
        say(f"  注意：{'、'.join(apps)} 還開著（會用到顯示卡，可能讓幾輪不算而要補跑），建議關掉")
    cpu = powershell("(Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter \"Name='_Total'\").PercentProcessorUtility")
    say(f"  CPU 使用率：{cpu}%")
    mon = powershell("@(Get-CimInstance -Namespace root\\wmi -ClassName WmiMonitorBasicDisplayParams | Where-Object { $_.Active }).Count")
    code, disp = smi(["--query-gpu=display_active,pstate,clocks.gr,clocks.mem", "--format=csv,noheader"])
    say(f"  螢幕：{mon} 個作用中｜NVIDIA 顯示卡接著螢幕（Display Active）：{disp.strip()}（10/19 是 Enabled）")
    for name, exe in EXES.items():
        r = subprocess.run([exe, "--version"], capture_output=True, timeout=60, cwd=os.path.dirname(exe)) if os.path.exists(exe) else None
        v = re.search(r"^version:.*$", (r.stdout + r.stderr).decode("utf-8", "replace"), re.M) if r else None
        say(f"  llama.cpp {name}：{v.group(0).strip() if v else '找不到'}")
        if not v or v.group(0).strip() != VERSION:
            problems.append(f"llama.cpp {name} 不是 b9549：{exe}")
    say("  模型 SHA-256 計算中……")
    h = hashlib.sha256()
    with open(MODEL, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    say(f"  模型 SHA-256：{'和論文相同' if h.hexdigest() == MODEL_SHA else '不同！' + h.hexdigest()}")
    if h.hexdigest() != MODEL_SHA:
        problems.append("模型檔和論文用的不同")
    if not DRY and PREREG_MARK not in open(os.path.join(HERE, "事前預期.md"), encoding="utf-8").read():
        problems.append("事前預期.md 裡找不到事前登記")
    return problems, disp.strip()


def ask(prompt):
    while True:
        try:
            s = input(prompt).strip()
        except EOFError:
            s = ""
        if s:
            return s
        print("（不能空白。想好了再打，打完按 Enter）")


def record_predictions():
    arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
    pred = {}
    if "A" in ONLY:
        pred["A"] = arg("--pred-a") or ask("\n請寫下你對實驗 A 的預測（CUDA 版會不會也有閒置代價？大約多幾秒？）：\n> ")
    if "B" in ONLY:
        pred["B"] = arg("--pred-b") or ask("\n請寫下你對實驗 B 的預測（例如中間三點會不會在兩端連線上或上方？）：\n> ")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    target = os.path.join(HERE, "out", f"{PFX}_{STAMP}_predictions_test.md") if DRY else os.path.join(HERE, "事前預期.md")
    text = f"\n\n### 使用者的預測（{now}，run_rtx4050_supplement.py 開跑前輸入，原文照錄）\n\n" + \
        "".join(f"- 實驗 {k}：{v}\n" for k, v in pred.items())
    open(target, "a", encoding="utf-8").write(text)
    say(f"預測已記下（{now}）→ {os.path.relpath(target, HERE)}")
    print(text.strip())


def wait_gpu_idle():
    """和 run_keepwarm5.py 的 prechecks 相同：時脈放開後，閒置要降到 1000 MHz 以下（遠端桌面連著時降不下來）"""
    say("確認顯示卡閒置時會降頻（遠端桌面要斷線；最多等 10 分鐘）……")
    t0, ok_n = time.time(), 0
    while time.time() - t0 < 600:
        code, out = smi(["--query-gpu=clocks.gr,utilization.gpu", "--format=csv,noheader,nounits"])
        try:
            gr, ut = (float(x) for x in out.strip().split(",")[:2])
        except ValueError:
            gr, ut = 9999, 100
        ok_n = ok_n + 1 if (gr < 1000 and ut <= 5) else 0
        if ok_n >= 3:
            say(f"  OK：閒置時顯示卡 {gr:.0f} MHz")
            return True
        time.sleep(3)
    return False


def run_step(args, label):
    say(f"===== {label}：python {' '.join(args)}")
    p = subprocess.Popen([sys.executable, "-X", "utf8"] + args, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    lines = []
    for raw in p.stdout:
        line = raw.decode("utf-8", "replace")
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    p.wait()
    say(f"===== {label} 結束（代碼 {p.returncode}）")
    return p.returncode, lines


# ---------------------------------------------------------------- 主程式
def restore_only():
    files = sorted(glob.glob(os.path.join(HERE, "out", "supplement_*_cpu_before.json")))
    if not files:
        sys.exit("找不到存下來的 CPU 設定（out/supplement_*_cpu_before.json）。")
    saved = json.load(open(files[-1], encoding="utf-8"))
    say(f"照 {os.path.basename(files[-1])} 還原")
    unlock_clocks()
    restore_cpu(saved)


def main():
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    sys.stdout = Tee(os.path.join(HERE, "out", f"{PFX}_{STAMP}_console.txt"))
    if RESTORE:
        return restore_only()
    disable_quick_edit()
    t_begin = time.time()
    print("=" * 70)
    print(f"  RTX 4050 補充實驗：{'、'.join(x for x in 'AB' if x in ONLY)}" + ("（試流程）" if DRY else "") +
          f"　預計約 {'5' if ONLY == 'AB' else '2' if ONLY == 'A' else '3'} 小時，跑完會自動還原")
    print("=" * 70)
    print("開跑前請確認（程式檢查不到）：")
    print("  1. Armoury Crate 是 Performance 模式，風扇跟著模式（沒有手動設定）")
    print("  2. 外接螢幕接著（和 10/19 相同）")
    print("  3. Wallpaper Engine、Steam、Discord、Chrome、影片、遊戲等會用到顯示卡的程式都關了")
    print("  4. 用遠端桌面啟動的話：等一下輸入完預測後，10 分鐘內斷線")
    print("  5. 跑的時候不要點這個視窗、不要關掉它；筆電不要闔上")
    if not DRY:
        ask("都確認好了請打 y 再按 Enter（還沒的話直接關掉這個視窗）：")
    problems, disp = checks()
    if problems and DRY:
        say("（試流程：下面的問題照樣繼續）\n  - " + "\n  - ".join(problems))
    elif problems:
        say("有問題，沒有開始：\n  - " + "\n  - ".join(problems))
        return 1
    if not DRY and not disp.lower().startswith("enabled"):
        ask("NVIDIA 顯示卡沒有接著螢幕（10/19 是接著的）。確定要照現在的狀態跑，請打 y 再按 Enter（不要的話關掉視窗、接好螢幕再重來）：")
    record_predictions()

    saved = None
    keep_awake(True)
    try:
        before = read_cpu()
        say(f"CPU 設定（插電）現在：{fmt_cpu(before)}")
        if DRY:
            perf = processor_performance()
            say(f"（試流程：不改 CPU 設定）有負載時 % Processor Performance 最高 {perf}%")
        else:
            json.dump(before, open(os.path.join(HERE, "out", f"{PFX}_{STAMP}_cpu_before.json"), "w", encoding="utf-8"), indent=1)
            saved = before
            now, bad = set_cpu_ac(CPU_EXPERIMENT)
            say(f"CPU 設定改成實驗用：{fmt_cpu(now)}")
            if bad:
                say(f"這幾項沒改成功：{bad}，沒有開始。")
                return 1
            perf = processor_performance()
            say(f"有負載時 % Processor Performance 最高 {perf}%（10/21 改之前是 155%）")
            if perf is None or perf <= 100:
                say("Turbo 好像沒有開（沒有超過 100%），沒有開始。請確認 Armoury Crate 是 Performance，再把這段訊息給 Claude。")
                return 1
            unlock_clocks()
        results = {}
        if "A" in ONLY:
            if not DRY and not wait_gpu_idle():
                say("顯示卡閒置 10 分鐘都降不到 1000 MHz 以下（遠端桌面還連著？），沒有開始。")
                return 1
            code, lines = run_step(["run_backend_probe.py"] + (["--dry", "--quick"] if DRY else []), "實驗 A")
            rd = next((l.split("=", 1)[1].strip() for l in lines if l.startswith("RESULT_DIR=")), None)
            results["A"] = code
            if rd and os.path.exists(rd + "_cuda.zip") and os.path.exists(rd + "_vulkan.zip"):
                run_step(["analyze_backend_probe.py", rd], "實驗 A 的分析")
        if "B" in ONLY:
            fake = None
            if DRY:
                fake = subprocess.Popen([sys.executable, os.path.join(HERE, "fake_server.py"), "--dist", "uniform", "--mean", "1.0",
                                         "--spread", "0.1", "--port", "9000", "--log", os.path.join(HERE, "out", "kwq5_fake_server_log.csv")],
                                        cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(2)
            try:
                code, _ = run_step(["run_keepwarm5.py", "--tag", KW5] + (["--dry", "--quick", "--url", "http://127.0.0.1:9000"] if DRY else []), "實驗 B")
                results["B"] = code
            finally:
                if fake:
                    fake.kill()
            if all(os.path.exists(os.path.join(HERE, "out", f"{'kwq5' if DRY else KW5}_tau{n}.csv")) for n in ("0", "3", "6", "12", "inf")):
                run_step(["analyze_keepwarm5.py", "kwq5" if DRY else KW5], "實驗 B 的分析")
        say("全部結束：" + "、".join(f"實驗 {k} {'完整' if v == 0 else '沒有全部跑完'}" for k, v in results.items()) +
            f"，共 {(time.time() - t_begin) / 3600:.1f} 小時")
        return 0
    except KeyboardInterrupt:
        say("被中斷（Ctrl＋C）。")
        return 1
    except Exception:
        say("發生錯誤：\n" + traceback.format_exc())
        return 1
    finally:
        if not DRY:
            unlock_clocks()
        if saved:
            restore_cpu(saved)
        keep_awake(False)
        say("可以關掉這個視窗了。結果和畫面紀錄都在 LLM延遲實驗/out/。")


if __name__ == "__main__":
    sys.exit(main())
