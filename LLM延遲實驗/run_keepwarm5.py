"""RTX 4050 補充實驗 B（2026-10-21，設計：RTX4050補充實驗_設計草稿_20261021.md）：保溫五個點在同一段時間重跑
判定標準：事前預期.md（2026-10-21 的事前登記）。要用系統管理員權限執行（鎖時脈要用）。

五個點：τ = 0（不保溫＝預設）、3、6、12 秒、∞（一直鎖），順序用種子 5011 打亂（12、0、6、∞、3）；
每點 180 個請求，到達時間和 10/19 同一串（種子 11、λ = 0.10885，和 load_gen.py 相同的抽法），丟掉前 60 個。
保溫的做法和 run_keepwarm.py 相同：每段閒置一開始鎖住時脈（-lgc 2550,2550、-lmc 8001,8001），閒置滿 τ 秒還沒有新請求就放開；
τ = 0 完全不鎖，τ = ∞ 從頭鎖到尾。結束時（不管正常或出錯）一定放開。伺服器用 start_llama.bat（和論文、10/19 相同）。

第 1 步：校正，三種，區塊交錯（D1 L1 K1 D2 L2 K2 D3 L3 K3 D4 L4）；換狀態後先送一個不記錄的請求
  D 預設（一直不鎖）：閒置 0、2、3、5、8 秒 × 3 次 × 4 區塊，順序和 10/11 的 probe_base_1011 相同
  L 一直鎖：同上，時脈全程鎖住，順序和 probe_locked_1011 相同
  K 鎖住再放開：和 run_keepwarm.py 的校正相同（閒置 0、1、2.5、3.5、4、5、6、8、11 秒 × 4 次 × 3 區塊，閒置滿 3 秒放開）
第 2 步：五個點開跑前，自動寫下預測（事前預期.md、plan_kw5_tau*.json）
  τ = 0 用 D、τ = ∞ 用 L，算法和 predict_power.py 相同；τ = 3、6、12 用 K，算法和 run_keepwarm.py 相同
第 3 步：五個點 → 分析：analyze_keepwarm5.py

用法：python run_keepwarm5.py                     正式（系統管理員 PowerShell；run_rtx4050_supplement.py 會呼叫）
      python run_keepwarm5.py --dry --quick --url http://127.0.0.1:9000
                                                 試流程：不鎖時脈、不開真的伺服器、連假伺服器，請求數變少
"""
import csv
import ctypes
import datetime
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRY, QUICK = "--dry" in sys.argv, "--quick" in sys.argv
URL = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else "http://127.0.0.1:8080"
SEED, LAM, N, K = 11, 0.10885, 180, 60              # 和 10/19 五個點完全相同的到達時間
TAUS = [0.0, 3.0, 6.0, 12.0, math.inf]
ORDER_SEED = SEED + 5000                            # 五個點的順序：12、0、6、∞、3
TAU_C = 3.0                                         # K 校正時固定 3 秒放開
DL_GAPS, DL_REPS, DL_BLOCKS = [0, 2, 3, 5, 8], 3, 4                           # D、L：和 10/11 的兩組閒置測試相同
CAL_GAPS, CAL_REPS, CAL_BLOCKS = [0, 1, 2.5, 3.5, 4, 5, 6, 8, 11], 4, 3       # K：和 run_keepwarm.py 相同
B_PRED = 1.837                                      # 規則 7 用的 b（10/11 預設的閒置測試）
EDGES = [1.0, 2.0, 5.0]
BIN, NBIN = 0.5, 16
R = 2000
if QUICK:
    N, K, DL_BLOCKS, CAL_BLOCKS = 16, 5, 1, 1
PROMPT = open(os.path.join(HERE, "prompts", "medium.txt"), encoding="utf-8").read()
MAX_TOKENS = 128
STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M")
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "kw5"     # 10/22 加：重跑時換檔名（例如 kw5m），不蓋掉之前的
PFX = "kwq5" if QUICK or DRY else TAG               # 試流程的檔名和正式的分開


def say(m=""):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def tname(tau):
    """檔名用"""
    return "inf" if math.isinf(tau) else f"{tau:g}"


def tlabel(tau):
    """給人看的"""
    return "∞" if math.isinf(tau) else f"{tau:g}"


def keep_awake(on=True):
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x1 | 0x2 if on else 0))


def smi(args, timeout=30):
    r = subprocess.run(["nvidia-smi"] + args, capture_output=True, timeout=timeout)
    return r.returncode, r.stdout.decode("utf-8", "replace")


class Clock:
    """鎖／放開時脈（和 run_keepwarm.py 相同）。dry 模式不真的呼叫，只記錄"""
    def __init__(self):
        self.locked, self.log, self.mu = False, [], threading.Lock()

    def _do(self, args):
        if DRY:
            time.sleep(0.05)
            return True
        return smi(args)[0] == 0

    def lock(self, why=""):
        with self.mu:
            if self.locked:
                return
            t0 = time.time()
            ok = self._do(["-lgc", "2550,2550"]) and self._do(["-lmc", "8001,8001"])
            self.locked = ok
            self.log.append(("lock", round(t0, 3), round(time.time(), 3), ok, why))

    def unlock(self, why="", force=False):
        with self.mu:
            if not self.locked and not force:
                return
            t0 = time.time()
            ok = self._do(["-rgc"]) and self._do(["-rmc"])
            self.locked = not ok
            self.log.append(("unlock", round(t0, 3), round(time.time(), 3), ok, why))


CLOCK = Clock()


def request(prompt, max_tokens=MAX_TOKENS):
    """和論文的閒置測試、load_gen.py、run_keepwarm.py 相同的請求"""
    payload = {"model": "local", "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": True, "ignore_eos": True}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer sk-no-key-required"})
    wall, t_send, t_first, n = time.time(), time.perf_counter(), None, 0
    with urllib.request.urlopen(req, timeout=600) as resp:
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


def gpu_logger(path):
    return subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,pstate,clocks.gr,clocks.mem,utilization.gpu,power.draw",
                             "--format=csv,noheader,nounits", "-lms", "100", "-f", path])


def schedule():
    """和 load_gen.py（predict_plan.py 的 schedule）相同的抽法，所以到達時間和 10/19 的五個點一模一樣"""
    r, t, arr = random.Random(SEED), 0.0, []
    for _ in range(N):
        t += r.expovariate(LAM)
        r.choices([0], weights=[1.0], k=1)
        r.choices([MAX_TOKENS], weights=[1.0], k=1)
        arr.append(t)
    return arr


# ---------------------------------------------------------------- 開跑前檢查（和 run_keepwarm.py 相同）
def prechecks():
    if DRY:
        say("（試流程：跳過權限和時脈檢查）")
        return
    code, out = smi(["-lgc", "2550,2550"])
    smi(["-rgc"])
    if code != 0:
        sys.exit("鎖不住時脈：請用「以系統管理員身分執行」的 PowerShell 執行這個程式。\n" + out[-300:])
    smi(["-rmc"])
    say("有系統管理員權限。確認時脈放開時，閒置會降下來（遠端桌面連著時會降不下來）……")
    t0, ok_n = time.time(), 0
    while time.time() - t0 < 600:                       # 10 分鐘：可以用遠端桌面啟動，再在這段時間內斷線
        code, out = smi(["--query-gpu=clocks.gr,utilization.gpu", "--format=csv,noheader,nounits"])
        try:
            gr, ut = (float(x) for x in out.strip().split(",")[:2])
        except ValueError:
            gr, ut = 9999, 100
        ok_n = ok_n + 1 if (gr < 1000 and ut <= 5) else 0
        if ok_n >= 3:
            say(f"  OK：閒置時顯示卡 {gr:.0f} MHz")
            return
        time.sleep(3)
    sys.exit("放開時脈後，閒置 10 分鐘都降不到 1000 MHz 以下。請先關掉 Chrome 遠端桌面的連線、Wallpaper Engine 等程式再執行。")


def start_server():
    if DRY:
        return None
    try:
        if urllib.request.urlopen(URL + "/health", timeout=3).status == 200:
            sys.exit("本機 8080 已經有伺服器在跑，先關掉再執行。")
    except Exception:
        pass
    p = subprocess.Popen(["cmd", "/c", os.path.join(HERE, "start_llama.bat")], cwd=HERE,
                         stdout=open(os.path.join(HERE, f"llama_{PFX}_{STAMP}.log"), "w"), stderr=subprocess.STDOUT)
    for _ in range(100):
        time.sleep(3)
        try:
            if urllib.request.urlopen(URL + "/health", timeout=3).status == 200:
                return p
        except Exception:
            pass
    sys.exit("伺服器沒有啟動成功。")


def speed_check():
    request("[speedcheck warmup] " + PROMPT)
    s = []
    for i in range(3):
        ts, tf, td, wall, n = request(f"[speedcheck {i}] " + PROMPT)
        s.append(td - ts)
    m = sum(s) / 3
    say(f"規則 7 速度檢查：3 個請求平均 {m:.3f} 秒，b = {B_PRED}，差 {100 * (m / B_PRED - 1):+.1f}%")
    if not DRY and abs(m / B_PRED - 1) > 0.10:
        sys.exit("服務時間和 b 差超過 10%，依規則 7 停止。")


# ---------------------------------------------------------------- 第 1 步：校正
def cal_blocks():
    """區塊順序：D1 L1 K1 D2 L2 K2 D3 L3 K3 D4 L4"""
    out = []
    for i in range(max(DL_BLOCKS, CAL_BLOCKS)):
        out += [(p, i + 1) for p in ("D", "L") if i < DL_BLOCKS] + ([("K", i + 1)] if i < CAL_BLOCKS else [])
    return out


def calibrate():
    blocks = cal_blocks()
    say(f"第 1 步：校正（D 預設、L 一直鎖、K 鎖住 {TAU_C:g} 秒再放開；{len(blocks)} 個區塊交錯：{' '.join(p + str(b) for p, b in blocks)}）")
    rows, spans = [], []
    rng_dl = {"D": random.Random(2026), "L": random.Random(2026)}       # 和 probe_idle.py 相同：10/11 兩組的閒置順序
    rng_k = random.Random(SEED + 1000)                                   # 和 run_keepwarm.py 的校正相同
    gpu = gpu_logger(os.path.join(HERE, "out", f"{PFX}_cal_gpu.csv"))
    try:
        for part, b in blocks:
            # 換狀態：L 鎖住；D、K 放開（K 和 run_keepwarm.py 一樣從放開開始）。再送一個不記錄的請求，讓第一段閒置從這個狀態開始
            if part == "L":
                CLOCK.lock(f"cal L{b} start")
            else:
                CLOCK.unlock(f"cal {part}{b} start", force=True)
            time.sleep(1.0)
            request(f"[cal transition {part}{b}] " + PROMPT)
            last_end, w0 = time.perf_counter(), time.time()
            if part == "K":
                order = CAL_GAPS * CAL_REPS
                rng_k.shuffle(order)
            else:
                order = DL_GAPS * DL_REPS
                rng_dl[part].shuffle(order)
            for g in order:
                if part == "K" and g > 0:                   # 有閒置：一開始就鎖住，滿 TAU_C 還沒送就放開（和 run_keepwarm.py 相同）
                    CLOCK.lock("cal idle start")
                    if g > TAU_C:
                        wait_until(last_end + TAU_C)
                        CLOCK.unlock("cal tau_c")
                wait_until(last_end + g)
                if part == "K":
                    state = "queued" if g == 0 else ("warm" if g <= TAU_C else "cold")
                else:
                    state = "queued" if g == 0 else "idle"
                lk = CLOCK.locked
                ts, tf, td, wall, n = request(f"[cal {len(rows):04d}] " + PROMPT)
                rows.append({"req_id": len(rows), "part": part, "block": b, "gap": g, "state": state, "locked_at_send": int(lk),
                             "idle_before": round(ts - last_end, 4),
                             "u_after_unlock": round(ts - last_end - TAU_C, 4) if state == "cold" else "",
                             "t_send": round(ts, 4), "wall_send": round(wall, 4), "ttft": round(tf - ts, 4) if tf else "",
                             "service": round(td - ts, 4), "out_tokens": n})
                last_end = td
            spans.append((part, b, w0, time.time()))
            say(f"  校正 {part}{b} 完成（到目前 {len(rows)} 個請求）")
    finally:
        CLOCK.unlock("cal end", force=True)
        time.sleep(2)
        gpu.terminate()
        gpu.wait(timeout=10)
    with open(os.path.join(HERE, "out", f"{PFX}_cal.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(spans, open(os.path.join(HERE, "out", f"{PFX}_cal_spans.json"), "w", encoding="utf-8"), indent=0)
    say(f"  校正完成：{len(rows)} 個請求")
    return rows, spans


# ---------------------------------------------------------------- 第 2 步：預測
def parse_t(s):
    return datetime.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def read_gpu(path):
    ts, pw, ps = [], [], []
    for p in csv.reader(open(path, encoding="utf-8")):
        try:
            t_, s_, w_ = parse_t(p[0]), p[1].strip(), float(p[5])
        except (ValueError, IndexError):
            continue
        ts.append(t_)
        ps.append(s_)
        pw.append(w_)
    return np.array(ts), np.array(pw), np.array(ps)


def endpoint_params(rows, ts, pw, ps, spans):
    """τ = 0、∞ 的參數：predict_power.py 的 load()。請求中功耗、閒置 8 秒那幾段的功耗曲線（每 0.5 秒）、
    P8 的功耗中位數（只用這一種校正的區塊；沒有 P8 就用 8 秒時的值）"""
    rows = sorted(rows, key=lambda r: r["wall_send"])
    busy = np.zeros(len(ts), bool)
    for r in rows:
        busy |= (ts >= r["wall_send"] + 0.1) & (ts <= r["wall_send"] + r["service"] - 0.05)
    prof = {}
    for j in range(1, len(rows)):
        r = rows[j]
        if r["gap"] != 8:
            continue
        a = r["wall_send"] - r["idle_before"]
        for k in range(16):
            sel = (ts >= a + 0.5 * k) & (ts < a + 0.5 * (k + 1))
            if sel.any():
                prof.setdefault(k, []).append(pw[sel].mean())
    profile = np.array([np.mean(prof[k]) if k in prof else np.nan for k in range(16)])
    if np.isnan(profile).any():                      # 某一段剛好沒有讀數：用前後兩段內插（predict_power.py 沒有這一步，那時每段都有）
        ok = ~np.isnan(profile)
        profile = np.interp(np.arange(16), np.flatnonzero(ok), profile[ok])
    inside = np.zeros(len(ts), bool)
    for _, _, w0, w1 in spans:
        inside |= (ts >= w0) & (ts <= w1)
    p8 = pw[inside & (ps == "P8")]
    tail = float(np.median(p8)) if len(p8) else float(profile[-1])
    return rows, float(pw[busy].mean()), profile, tail


def predict_endpoint(rows, p_busy, profile, tail, rng, arr):
    """predict_power.py 的模擬：一個接一個的 b 加上閒置代價（照閒置時間分四段），閒置能量照功耗曲線"""
    k0 = K
    s0 = np.array([r["service"] for r in rows if r["gap"] == 0])
    by_bin = [[] for _ in range(4)]
    for r in rows:
        if r["gap"] > 0:
            by_bin[int(np.digitize(r["idle_before"], EDGES))].append(r["service"])
    by_bin = [np.array(x) if x else np.array([s0.mean()]) for x in by_bin]
    pool = s0[rng.integers(0, len(s0), (R, len(s0)))]
    base = pool[np.arange(R)[:, None], rng.integers(0, len(s0), (R, N))]
    pd_ = s0[rng.integers(0, len(s0), (R, len(s0)))].mean(1)
    dv = np.stack([by_bin[i][rng.integers(0, len(by_bin[i]), (R, len(by_bin[i])))].mean(1) for i in range(4)], 1) - pd_[:, None]
    cum = np.concatenate([[0.0], np.cumsum(profile * 0.5)])          # 閒置 I 秒的能量：前 8 秒照曲線，之後照 tail

    def idle_energy(I):
        I = np.maximum(I, 0.0)
        full = np.minimum(I, 8.0)
        k = np.floor(full / 0.5).astype(int)
        e = cum[k] + (full - 0.5 * k) * profile[np.minimum(k, 15)]
        return e + np.maximum(I - 8.0, 0.0) * tail

    c_prev, acc_w, acc_t, energy = np.full(R, -np.inf), np.zeros(R), np.zeros(R), np.zeros(R)
    t_start = None
    for i in range(N):
        start = np.maximum(arr[i], c_prev)
        w = start - arr[i]
        idle = np.where(np.isfinite(c_prev), np.maximum(arr[i] - c_prev, 0.0), 0.0)
        pen = np.where(w == 0, dv[np.arange(R), np.digitize(arr[i] - c_prev, EDGES)], 0.0)
        svc = base[:, i] + pen
        if i >= k0:
            if t_start is None:
                t_start = c_prev.copy()
            acc_w += w
            acc_t += w + svc
            energy += idle_energy(idle) + p_busy * svc
        c_prev = start + svc
    wait, resp, pw = acc_w / (N - k0), acc_t / (N - k0), energy / (c_prev - t_start)
    q = lambda x: [round(float(v), 3) for v in np.percentile(x, [2.5, 97.5])]
    return {"wait_pred": round(float(wait.mean()), 3), "wait_95": q(wait), "resp_pred": round(float(resp.mean()), 3),
            "resp_95": q(resp), "power_pred_W": round(float(pw.mean()), 2), "power_95": q(pw)}


def lockunlock_params(rows, ts, pw):
    """τ = 3、6、12 的參數：run_keepwarm.py 的 cal_params()（只用 K 的請求）"""
    S0 = np.array([r["service"] for r in rows if r["state"] == "queued"])
    warm = np.array([r["service"] for r in rows if r["state"] == "warm"])
    cold_by = [[] for _ in range(4)]
    for r in rows:
        if r["state"] == "cold":
            cold_by[int(np.digitize(r["u_after_unlock"], EDGES))].append(r["service"])
    cold_by = [np.array(x) if x else np.array([S0.mean()]) for x in cold_by]
    busy = {"queued": [], "warm": [], "cold": []}
    for r in rows:
        sel = (ts >= r["wall_send"]) & (ts <= r["wall_send"] + r["service"])
        if sel.any():
            busy[r["state"]].append(pw[sel].mean())
    p_busy = {k: float(np.mean(v)) if v else float("nan") for k, v in busy.items()}
    lock_prof, unl_prof = {}, {}
    for r in sorted(rows, key=lambda r: r["wall_send"]):
        if r["state"] != "cold":
            continue
        a = r["wall_send"] - r["idle_before"]                       # 閒置開始（牆上時間）
        for k in range(int(TAU_C / BIN)):
            s = (ts >= a + BIN * k) & (ts < a + BIN * (k + 1))
            if s.any():
                lock_prof.setdefault(k, []).append(pw[s].mean())
        if r["gap"] == max(CAL_GAPS):
            for k in range(NBIN):
                s = (ts >= a + TAU_C + BIN * k) & (ts < a + TAU_C + BIN * (k + 1))
                if s.any():
                    unl_prof.setdefault(k, []).append(pw[s].mean())
    lock_p = np.array([np.mean(lock_prof[k]) for k in sorted(lock_prof)])
    unl_p = np.array([np.mean(unl_prof[k]) if k in unl_prof else np.nan for k in range(NBIN)])
    unl_p = np.where(np.isnan(unl_p), np.nanmin(unl_p), unl_p)
    return {"S0": S0, "warm": warm, "cold_by": cold_by, "p_busy": p_busy, "lock_prof": lock_p,
            "lock_flat": float(lock_p[2:].mean()) if len(lock_p) > 2 else float(lock_p.mean()),
            "unl_prof": unl_p, "unl_tail": float(unl_p[-1])}


def predict_lockunlock(P, taus, arr):
    """run_keepwarm.py 的 predict()：保溫中到達付 δ_w，放開後到達照放開了多久（u）付 δ_after；閒置能量照兩條曲線"""
    rng = np.random.default_rng(SEED + 200)
    S0 = P["S0"]
    lock_cum = np.concatenate([[0.0], np.cumsum(P["lock_prof"] * BIN)])
    unl_cum = np.concatenate([[0.0], np.cumsum(P["unl_prof"] * BIN)])
    nl = len(P["lock_prof"])

    def e_lock(I):                                  # 鎖住的閒置 I 秒的能量：前 TAU_C 秒照曲線，之後照平的
        I = np.maximum(I, 0.0)
        a = np.minimum(I, BIN * nl)
        k = np.floor(a / BIN).astype(int)
        return lock_cum[k] + (a - BIN * k) * P["lock_prof"][np.minimum(k, nl - 1)] + np.maximum(I - BIN * nl, 0.0) * P["lock_flat"]

    def e_unl(u):                                   # 放開後 u 秒的能量
        u = np.maximum(u, 0.0)
        a = np.minimum(u, BIN * NBIN)
        k = np.floor(a / BIN).astype(int)
        return unl_cum[k] + (a - BIN * k) * P["unl_prof"][np.minimum(k, NBIN - 1)] + np.maximum(u - BIN * NBIN, 0.0) * P["unl_tail"]

    out = {}
    for tau in taus:
        base = S0[rng.integers(0, len(S0), (R, N))]
        pd_ = S0[rng.integers(0, len(S0), (R, len(S0)))].mean(1)
        dv = np.stack([P["cold_by"][i][rng.integers(0, len(P["cold_by"][i]), (R, len(P["cold_by"][i])))].mean(1)
                       for i in range(4)], 1) - pd_[:, None]
        dw = (P["warm"][rng.integers(0, len(P["warm"]), (R, len(P["warm"])))].mean(1) - pd_) if len(P["warm"]) else np.zeros(R)
        c_prev = np.full(R, -np.inf)
        acc_w, acc_t, energy, t_start = np.zeros(R), np.zeros(R), np.zeros(R), None
        for i in range(N):
            start = np.maximum(arr[i], c_prev)
            w = start - arr[i]
            idle = np.where(np.isfinite(c_prev), np.maximum(arr[i] - c_prev, 0.0), np.inf)
            queued = w > 0
            warm = (~queued) & (idle <= tau)
            cold = (~queued) & (idle > tau)
            u = np.where(cold, idle - tau, 0.0)
            pen = np.where(warm, dw, 0.0) + np.where(cold, dv[np.arange(R), np.digitize(u, EDGES)], 0.0)
            svc = base[:, i] + pen
            pb = np.where(queued, P["p_busy"]["queued"], np.where(warm, P["p_busy"]["warm"], P["p_busy"]["cold"]))
            if i >= K:
                if t_start is None:
                    t_start = c_prev.copy()
                fin = np.isfinite(idle)
                e_idle = np.where(fin & (idle <= tau), e_lock(np.where(fin, idle, 0.0)), 0.0) + \
                    np.where(fin & (idle > tau), e_lock(np.full(R, tau)) + e_unl(np.where(fin, idle - tau, 0.0)), 0.0)
                acc_w += w
                acc_t += w + svc
                energy += e_idle + pb * svc
            c_prev = start + svc
        wait, resp, pw = acc_w / (N - K), acc_t / (N - K), energy / (c_prev - t_start)
        q = lambda x: [round(float(v), 3) for v in np.percentile(x, [2.5, 97.5])]
        out[tau] = {"wait_pred": round(float(wait.mean()), 3), "wait_95": q(wait), "resp_pred": round(float(resp.mean()), 3),
                    "resp_95": q(resp), "power_pred_W": round(float(pw.mean()), 2), "power_95": q(pw)}
    return out


def predict(rows, spans):
    ts, pw, ps = read_gpu(os.path.join(HERE, "out", f"{PFX}_cal_gpu.csv"))
    arr = np.array(schedule())
    rng = np.random.default_rng(SEED + 100)            # 和 predict_power.py 相同：同一個亂數，先預設再鎖時脈
    out, summary = {}, {}
    for part, tau in (("D", 0.0), ("L", math.inf)):
        sub = [r for r in rows if r["part"] == part]
        sub, p_busy, profile, tail = endpoint_params(sub, ts, pw, ps, [s for s in spans if s[0] == part])
        out[tau] = predict_endpoint(sub, p_busy, profile, tail, rng, arr)
        s0 = np.array([r["service"] for r in sub if r["gap"] == 0])
        summary[part] = {"b": round(float(s0.mean()), 4), "n_b": int(len(s0)),
                         "delta_by_idle": {lab: round(float(np.mean([r["service"] for r in sub if r["gap"] > 0 and
                                                                   int(np.digitize(r["idle_before"], EDGES)) == i]) - s0.mean()), 3)
                                           for i, lab in enumerate(["<1s", "1-2s", "2-5s", ">=5s"])
                                           if any(r["gap"] > 0 and int(np.digitize(r["idle_before"], EDGES)) == i for r in sub)},
                         "p_busy_W": round(p_busy, 1), "idle_profile_W": [round(float(x), 1) for x in profile], "tail_W": round(tail, 1)}
    P = lockunlock_params([r for r in rows if r["part"] == "K"], ts, pw)
    out.update(predict_lockunlock(P, [3.0, 6.0, 12.0], arr))
    summary["K"] = {"b": round(float(P["S0"].mean()), 4), "n_b": int(len(P["S0"])),
                    "warm_delta": round(float(P["warm"].mean() - P["S0"].mean()), 3) if len(P["warm"]) else None,
                    "delta_after_u": {lab: round(float(P["cold_by"][i].mean() - P["S0"].mean()), 3)
                                      for i, lab in enumerate(["<1s", "1-2s", "2-5s", ">=5s"])},
                    "p_busy_W": {k: round(v, 1) for k, v in P["p_busy"].items()}, "lock_idle_W": round(P["lock_flat"], 1),
                    "unlock_profile_W": [round(float(x), 1) for x in P["unl_prof"]]}
    return summary, out


def write_prediction(summary, out):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    order = TAUS[:]
    random.Random(ORDER_SEED).shuffle(order)
    d, l = out[0.0], out[math.inf]
    dp = l["power_pred_W"] - d["power_pred_W"]
    slope = (l["resp_pred"] - d["resp_pred"]) / dp if abs(dp) > 1e-6 else None      # 試流程不鎖時脈，兩端功耗可能一樣
    lines = [f"\n\n## 保溫五個點（同一段時間{'' if PFX == 'kw5' else '，檔名 ' + PFX}）的事前預測（run_keepwarm5.py 自動寫入，{now}，校正剛跑完、五個點還沒開始）\n",
             f"- 設計和判定標準：本檔「RTX 4050 補充實驗 A、B 的事前登記」；校正：{PFX}_cal.csv（{'試流程，不算數' if DRY or QUICK else '正式'}）",
             f"- 五個點的順序：{'、'.join(tlabel(t) for t in order)}",
             f"- D 預設：b = {summary['D']['b']} 秒（{summary['D']['n_b']} 個）；閒置代價 {summary['D']['delta_by_idle']}；"
             f"請求中 {summary['D']['p_busy_W']} W；閒置曲線 {summary['D']['idle_profile_W']}；P8 {summary['D']['tail_W']} W",
             f"- L 一直鎖：b = {summary['L']['b']} 秒（{summary['L']['n_b']} 個）；閒置代價 {summary['L']['delta_by_idle']}；"
             f"請求中 {summary['L']['p_busy_W']} W；閒置曲線 {summary['L']['idle_profile_W']}；8 秒以後 {summary['L']['tail_W']} W",
             f"- K 鎖住再放開：b = {summary['K']['b']} 秒（{summary['K']['n_b']} 個）；保溫中到達的 δ = {summary['K']['warm_delta']}；"
             f"放開後的 δ_after = {summary['K']['delta_after_u']}；請求中 {summary['K']['p_busy_W']}；鎖住閒置 {summary['K']['lock_idle_W']} W；"
             f"放開後閒置曲線 {summary['K']['unlock_profile_W']}",
             "", "| τ | 平均等待 | 平均回應時間 | 顯示卡平均功耗（命中範圍 ±10%） | 模型預測的形狀：回應時間 − 兩端預測的連線 |",
             "|---|---|---|---|---|"]
    for tau in TAUS:
        o = out[tau]
        gap = "" if tau in (0.0, math.inf) or slope is None else \
            f"{o['resp_pred'] - (d['resp_pred'] + slope * (o['power_pred_W'] - d['power_pred_W'])):+.3f} 秒"
        lines.append(f"| {tlabel(tau)}{'' if math.isinf(tau) else ' 秒'} | {o['wait_pred']}（{o['wait_95'][0]}–{o['wait_95'][1]}） | "
                     f"{o['resp_pred']}（{o['resp_95'][0]}–{o['resp_95'][1]}） | "
                     f"{o['power_pred_W']} W（{o['power_pred_W'] * 0.9:.1f}–{o['power_pred_W'] * 1.1:.1f}） | {gap} |")
    target = os.path.join(HERE, "事前預期.md" if not (DRY or QUICK) else f"out/{PFX}_prediction_test.md")
    open(target, "a", encoding="utf-8").write("\n".join(lines) + "\n")
    for tau, o in out.items():
        plan = {"seed": SEED, "max_tokens": MAX_TOKENS, "prompt_file": "medium.txt", "tau": tname(tau), "condition": f"keepwarm5_tau{tname(tau)}",
                "made_at": now, "runs": [{"tag": f"{PFX}_tau{tname(tau)}", "rate": LAM, "n": N, "pred": o["wait_pred"], "lo": o["wait_95"][0],
                                          "hi": o["wait_95"][1], "resp_pred": o["resp_pred"], "resp_lo": o["resp_95"][0],
                                          "resp_hi": o["resp_95"][1], "power_pred_W": o["power_pred_W"]}]}
        json.dump(plan, open(os.path.join(HERE, f"plan_{PFX}_tau{tname(tau)}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump({"made_at": now, "summary": summary, "pred": {tname(t): o for t, o in out.items()}},
              open(os.path.join(HERE, "out", f"{PFX}_prediction.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    say(f"  預測已寫入 {os.path.basename(target)}（{now}）")


# ---------------------------------------------------------------- 第 3 步：五個點
def run_point(tau):
    tag = f"{PFX}_tau{tname(tau)}"
    arr = schedule()
    rows, mu, threads = [], threading.Lock(), []
    state = {"out": 0, "idle_since": None}
    stop = threading.Event()
    keep = tau > 0                                     # τ = 0：完全不鎖（預設）

    def controller():                                  # 閒置滿 τ 秒還沒有新請求就放開（τ = ∞ 不用）
        while not stop.is_set():
            with mu:
                idle = state["out"] == 0 and state["idle_since"] is not None and time.perf_counter() - state["idle_since"] >= tau
            if idle and CLOCK.locked:
                CLOCK.unlock(f"{tag} idle {tname(tau)}s")
            time.sleep(0.02)

    def one(i, t_s, t0):
        row = {"req_id": i, "t_scheduled": round(t_s, 4), "status": "ok", "error": ""}
        with mu:
            row["state_at_send"] = "queued" if state["out"] > 0 else ("warm" if CLOCK.locked else "cold")
            state["out"] += 1
            state["idle_since"] = None
        try:
            ts, tf, td, wall, n = request(f"[request {i:05d}] " + PROMPT)
            row.update(t_send=round(ts - t0, 4), t_first_token=round(tf - t0, 4) if tf else "", t_done=round(td - t0, 4),
                       ttft=round(tf - ts, 4) if tf else "", total=round(td - ts, 4), out_tokens=n, wall_send=round(wall, 4))
        except Exception as e:
            row.update(status="error", error=f"{type(e).__name__}: {e}"[:200])
        with mu:
            state["out"] -= 1
            idle_now = state["out"] == 0
            if idle_now:
                state["idle_since"] = time.perf_counter()
        if idle_now and keep:
            CLOCK.lock(f"{tag} idle start")                # 一段閒置一開始就鎖住
        with mu:
            rows.append(row)

    gpu = gpu_logger(os.path.join(HERE, "out", f"{tag}_gpu.csv"))
    if keep:
        CLOCK.lock(f"{tag} start")                         # 開始時伺服器閒著：當成一段閒置的開始
    else:
        CLOCK.unlock(f"{tag} start", force=True)
    with mu:
        state["idle_since"] = time.perf_counter()
    ctl = None
    if keep and not math.isinf(tau):
        ctl = threading.Thread(target=controller, daemon=True)
        ctl.start()
    wall_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.perf_counter()
    say(f"  {tag}：開始（UTC {wall_start}），{N} 個請求，約 {arr[-1] / 60:.0f} 分鐘")
    for i, t_s in enumerate(arr):
        wait_until(t0 + t_s)
        th = threading.Thread(target=one, args=(i, t_s, t0), daemon=True)
        th.start()
        threads.append(th)
        if (i + 1) % 30 == 0:
            say(f"    已送出 {i + 1}/{N}")
    for th in threads:
        th.join(timeout=900)
    stop.set()
    if ctl:
        ctl.join(timeout=5)
    CLOCK.unlock(f"{tag} end")
    time.sleep(2)
    gpu.terminate()
    gpu.wait(timeout=10)
    rows.sort(key=lambda r: r["req_id"])
    fields = ["req_id", "t_scheduled", "t_send", "t_first_token", "t_done", "ttft", "total", "out_tokens", "status", "error",
              "wall_send", "state_at_send"]
    with open(os.path.join(HERE, "out", f"{tag}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    late = [float(r["t_send"]) - float(r["t_scheduled"]) for r in rows if r.get("t_send") not in (None, "")]
    ok = sum(1 for r in rows if r["status"] == "ok")
    open(os.path.join(HERE, "out", f"{tag}_runlog.txt"), "w", encoding="utf-8").write(
        f"實驗開始時間（UTC）：{wall_start}\n成功 {ok}/{N}，送出延誤最大 {max(late or [0]):.3f} 秒\n")
    say(f"  {tag}：完成，成功 {ok}/{N}，送出延誤最大 {max(late or [0]):.3f} 秒")
    return ok == N


def main():
    keep_awake(True)
    server, done = None, 0
    try:
        prechecks()
        server = start_server()
        speed_check()
        rows, spans = calibrate()
        summary, out = predict(rows, spans)
        write_prediction(summary, out)
        order = TAUS[:]
        random.Random(ORDER_SEED).shuffle(order)
        say(f"第 3 步：五個點，順序 {'、'.join(tlabel(t) for t in order)}")
        for tau in order:
            time.sleep(10)
            done += run_point(tau)
        say("全部跑完。")
    except SystemExit:
        raise
    except Exception:
        say("發生錯誤：\n" + traceback.format_exc())
    finally:
        CLOCK.unlock("main end", force=True)
        json.dump(CLOCK.log, open(os.path.join(HERE, "out", f"{PFX}_clock_log_{STAMP}.json"), "w", encoding="utf-8"), indent=0)
        if server is not None:
            subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"], capture_output=True)
            say("已關閉 llama-server。")
        keep_awake(False)
    return 0 if done == len(TAUS) else 1


if __name__ == "__main__":
    sys.exit(main())
