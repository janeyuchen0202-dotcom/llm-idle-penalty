"""保溫取捨的實測（2026-10-19，新第 VI 節）：確認時脈狀態 → 開 nvidia-smi 每 100 毫秒記錄 → 照 plan 跑一組排隊實驗

- locked：等到時脈真的鎖住（閒置時顯示卡 ≥ 2400 MHz、記憶體 ≥ 7500 MHz，連續 3 次）才開始；最多等 6 小時
- default：等到時脈真的還原（閒置時顯示卡會降到 1000 MHz 以下）才開始；最多等 6 小時
- run_plan.py 的輸出（含 load_gen.py 印的開始時間，用來和耗電紀錄對時）存到 out/<tag>_runlog.txt
- 事前預測寫在 事前預期.md（2026-10-19 09:12），用 predict_power.py 算的

用法：python run_power.py locked|default
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def clocks():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.gr,clocks.mem,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        gr, mem, util = (float(x) for x in out.strip().split(",")[:3])
        return gr, mem, util
    except Exception:
        return None


def other_llama():
    """有別的 llama-server 在跑（例如其他測試）就先不要開始"""
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe"], capture_output=True,
                             timeout=20).stdout                  # 中文系統的輸出不是 UTF-8，直接比對位元組
        return b"llama-server.exe" in out
    except Exception:
        return False


def wait_for(cond, limit_s=6 * 3600):
    """只在顯示卡閒著（使用率 ≤ 5%）時判斷：locked 是閒著時時脈還維持在高檔；default 是閒著時時脈降得下來。
    （10/19 修正：原本沒看使用率，看影片或跑別的測試時時脈也會變高，可能被誤認成已經鎖好）"""
    t0, ok_n = time.time(), 0
    while time.time() - t0 < limit_s:
        c = clocks()
        if c and not other_llama():
            gr, mem, util = c
            idle = util <= 5
            good = idle and ((gr >= 2400 and mem >= 7500) if cond == "locked" else (gr < 1000))
            ok_n = ok_n + 1 if good else 0
            if ok_n >= 3:
                print(f"時脈狀態確認（{cond}）：顯示卡 {gr:.0f} MHz、記憶體 {mem:.0f} MHz、使用率 {util:.0f}%", flush=True)
                return True
        time.sleep(5)
    return False


def main():
    cond = sys.argv[1]
    assert cond in ("locked", "default")
    plan_file = f"plan_power_{cond}.json"
    tag = json.load(open(os.path.join(HERE, plan_file), encoding="utf-8"))["runs"][0]["tag"]
    print(f"{time.strftime('%H:%M:%S')} 等待時脈狀態：{'鎖住' if cond == 'locked' else '還原'}……", flush=True)
    if not wait_for(cond):
        sys.exit("6 小時內沒有等到正確的時脈狀態，沒有開跑。")
    gpu_log = os.path.join(HERE, "out", f"{tag}_gpu.csv")
    if os.path.exists(gpu_log):
        sys.exit(f"{gpu_log} 已經存在，為了不蓋掉資料，停止。")
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,pstate,clocks.gr,clocks.mem,utilization.gpu,power.draw",
                            "--format=csv,noheader,nounits", "-lms", "100", "-f", gpu_log])
    log = open(os.path.join(HERE, "out", f"{tag}_runlog.txt"), "w", encoding="utf-8")
    log.write(f"開始（本機時間）：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.flush()
    try:
        p = subprocess.Popen([sys.executable, "-X", "utf8", os.path.join(HERE, "run_plan.py"), plan_file], cwd=HERE,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        for raw in p.stdout:
            line = raw.decode("utf-8", "replace")
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        p.wait()
        log.write(f"run_plan 結束代碼 {p.returncode}；結束（本機時間）：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    finally:
        time.sleep(3)
        smi.terminate()
        try:
            smi.wait(timeout=10)
        except Exception:
            smi.kill()
        log.close()
    print(f"{time.strftime('%H:%M:%S')} 完成：{tag}", flush=True)


if __name__ == "__main__":
    main()
