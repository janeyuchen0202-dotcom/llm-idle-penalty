"""全文稽核（2026-10-24）：改寫版論文的每一個數字，從原始資料重算，放回原句，確認那句話一字不差地出現在論文裡

- 論文：重寫/論文_目前版本_20261024.md（排進論文的英文；和 數字清單.py 用同一套讀法）
- 每一條檢查是 (節, 句子, 來源)：句子裡的數字是重算出來的值，照論文的寫法格式化。
  句子在論文裡找得到 → ✅；找不到 → ⚠️（印出重算的句子，和論文對照）
- 檢查的種類：重算（從原始量測紀錄算）、紀錄（從原始紀錄或程式碼讀設定值）、陳述（沒有紀錄可以核對，例如變壓器瓦數）
- 最後用 數字清單.py 的規則找出論文裡每一個數字，檢查是不是都被某一條 ✅ 的句子涵蓋；沒有涵蓋的列出來
- 重算沿用 recompute.py 的函式（照做完的順序重建、丟掉前 1/3…）；不重跑會寫檔的分析程式（analyze_strengthen.py 除外，
  它的 WARM_DIV 版本不寫檔，預設版本會重寫 strengthen_summary.csv，內容相同）
- 輸出：稽核/全文稽核結果.md
用法：在 ICAIIC投稿 資料夾執行  python 稽核/全文稽核.py      （約 3–5 分鐘）
"""
import contextlib
import csv
import glob
import io
import json
import os
import re
import statistics
import sys
import zipfile
from datetime import datetime

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recompute as rc            # noqa: E402
import 數字清單 as nl             # noqa: E402

PAPER = os.path.dirname(HERE)
BASE = os.path.dirname(PAPER)
EXP, OUT = rc.EXP, rc.OUT
FRIENDS = os.path.join(BASE, "朋友測試包", "朋友結果")
KIT = os.path.join(BASE, "朋友測試包", "LLM閒置測試", "_kit", "scripts")
ANALYSIS = os.path.join(BASE, "朋友測試包", "分析程式")
RESULT = os.path.join(HERE, "全文稽核結果.md")
CLAIMS = []          # (節, 句子, 來源, 種類)
V = {}               # 各節重算的結果，摘要、引言、結論重複用（同一個數字全文都來自同一次重算）


def C(sec, sentence, src, kind="重算"):
    CLAIMS.append((sec, sentence, src, kind))


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def neg(s):
    return s.replace("-", "−")


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}%"


def code_has(path, pattern):
    """程式碼裡有這一行（設定值）"""
    return re.search(pattern, open(path, encoding="utf-8").read()) is not None


# ------------------------------------------------------------------ III-A 測試平台
def sec_testbed():
    hw = json.load(open(os.path.join(OUT, "backend_20261022-0041", "cuda", "hardware.json"), encoding="utf-8"))
    ram = sum(m["Capacity"] for m in hw["memory"]) / 2 ** 30
    smi = hw["nvidia_smi"].splitlines()[1].split(", ")
    vram, drv = float(smi[3].split()[0]) / 1024, smi[2]
    cpu = hw["cpu"]["Name"]
    assert "i9-13900H" in cpu and "RTX 4050 Laptop" in smi[1], (cpu, smi)
    C("III-A", f"Intel Core i9-13900H CPU, {ram:.0f} GB of memory and an NVIDIA GeForce RTX 4050 Laptop GPU ({vram:.0f} GB, driver {drv})",
      "out/backend_20261022-0041/cuda/hardware.json（WMI、nvidia-smi）", "紀錄")
    osname = hw["os"]["Caption"]
    C("III-A", f"running Windows {'10' if 'Windows 10' in osname else '?'} on AC power from a 240 W adapter",
      "作業系統：hardware.json；240 W：使用者 10/23 的陳述（沒有紀錄）", "陳述")
    ac = hw["power"]["PROCTHROTTLEMIN"]["ac"]
    C("III-A", f"with the CPU's minimum performance state at {ac}% and default GPU power settings",
      "hardware.json 的 powercfg（PROCTHROTTLEMIN，AC）", "紀錄")
    log = open(os.path.join(EXP, "llama_strengthen.log"), encoding="utf-8", errors="replace").read()
    q = re.search(r"qwen2\.5-coder-3b-instruct-q(\d)_k_m\.gguf", log, re.I)
    C("III-A", f"quantized to {q.group(1)} bits (Q4_K_M)", "llama_strengthen.log 的模型檔名", "紀錄")
    # 3B 放得進每一張志願者的顯示卡；佔多少記憶體、最小的卡
    layers, mem_gb, vram_min = [], [], []
    for z in glob.glob(os.path.join(FRIENDS, "*", "閒置測試結果_GPU_*.zip")) + glob.glob(os.path.join(FRIENDS, "*", "*", "*.zip")):
        zz = zipfile.ZipFile(z)
        ri = [n for n in zz.namelist() if n.endswith("run_info.json")]
        if not ri:
            continue
        info = json.loads(zz.read(ri[0]))
        if info.get("mode") == "cpu" or "placement_probe" not in info:
            continue
        pp = info["placement_probe"]
        layers.append(pp["offloaded"] == pp["layers"])
        if info.get("gpu_mem_gb"):
            mem_gb.append(info["gpu_mem_gb"])
        smi_q = [n for n in zz.namelist() if n.endswith("nvidia_smi_q_start.txt")]
        if smi_q:
            m = re.search(r"FB Memory Usage\s+Total\s+:\s+(\d+) MiB", zz.read(smi_q[0]).decode("utf-8", "replace"))
            if m:
                vram_min.append(int(m.group(1)))
    for z in (glob.glob(os.path.join(OUT, "backend_*.zip"))):
        info = json.loads(zipfile.ZipFile(z).read("run_info.json"))
        if info.get("gpu_mem_gb"):
            mem_gb.append(info["gpu_mem_gb"])
    assert all(layers), "有志願者的顯示卡沒有全放進去"
    C("III-A", f"We chose a 3B model so that it fits entirely on every volunteer GPU ({min(mem_gb):.1f}–{max(mem_gb):.1f} GB; "
               f"the smallest had {min(vram_min) / 1024:.0f} GB)",
      f"志願者 GPU 測試包的 run_info.json（{len(layers)} 份全部 37/37 層）、gpu_mem_gb（{len(mem_gb)} 份）、nvidia-smi -q 的記憶體", "紀錄")
    tc = open(os.path.join(EXP, "prompts", "token_counts.txt"), encoding="utf-8").read()
    med = re.search(r"medium\.txt\s+目標 \d+ tokens\s+實際 (\d+) tokens", tc).group(1)
    toka = sorted({int(t) for n in rc.A_RUNS for t in rc.D[n]["tok"]})
    tokc = sorted({int(t) for n in rc.C_RUNS for t in rc.D[n]["tok"]})
    share = np.mean([np.mean(rc.D[n]["tok"] == 32) for n in rc.C_RUNS])
    assert len(toka) == 1 and len(tokc) == 2 and abs(share - 0.5) < 0.06, (toka, tokc, share)
    cv = {n: rc.win(rc.run(n))[2] for n in ("mix_A_rho50", "var_C_rho50")}
    C("III-A", f"Each request has an {med}-token prompt with a unique prefix and produces {toka[0]} output tokens (workload A) or "
               f"{tokc[0]} or {tokc[1]} tokens with equal probability (workload C); their service times have coefficients of variation "
               f"of {f2(cv['mix_A_rho50'].std() / cv['mix_A_rho50'].mean())} and {f2(cv['var_C_rho50'].std() / cv['var_C_rho50'].mean())}",
      f"prompts/token_counts.txt；17 組的 out_tokens（C 組 32 token 的比例 {share:.2f}）；out/mix_A_rho50.csv、var_C_rho50.csv")
    ts = gpu_times(os.path.join(OUT, "probe_idle_3b_gpu.csv"))
    lms = code_has(os.path.join(EXP, "probe_idle.py"), r'"-lms", "100"')
    C("III-A", f"nvidia-smi logs the GPU clocks, P-state and power about every {np.median(np.diff(ts)):.1f} s",
      f"probe_idle.py 設定 -lms 100；out/probe_idle_3b_gpu.csv 實際間隔中位數 {1000 * np.median(np.diff(ts)):.0f} ms",
      "紀錄" if lms else "⚠️")
    t16 = sorted({int(r["out_tokens"]) for tag in ("m1_rho30", "m1_rho50", "m1_rho70", "win_rho30", "win_rho50", "win_rho70")
                  for r in rc.rows_of(tag)} | {int(r["out_tokens"]) for p in ("probe_m1", "probe_m4", "probe_wingpu", "probe_wincpu")
                                               for r in rc.probe(p)})
    C("III-A", f"used the same model file, {t16[0]} output tokens and requests sent over Wi-Fi", "out/m1_*、win_*、probe_m1、m4、win*.csv",
      "紀錄" if len(t16) == 1 else "⚠️")


def gpu_times(path):
    ts = []
    for line in open(path, encoding="utf-8", errors="replace"):
        p = [x.strip() for x in line.split(",")]
        try:
            ts.append(datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp())
        except (ValueError, IndexError):
            continue
    return np.array(sorted(ts))


# ------------------------------------------------------------------ III-B、III-C
QUEUE_SKIP = ("_gpu", "_cpu", "summary", "bins", "runlog", "my_server_log", "netidle", "idle_penalty")


def queue_files():
    """所有排隊類的請求紀錄（每個請求有送出和做完的時間）：RTX 4050、M1、HP、假伺服器、保溫，和志願者的排隊測試"""
    for fpath in sorted(glob.glob(os.path.join(OUT, "*.csv"))):
        b = os.path.basename(fpath)
        if any(s in b for s in QUEUE_SKIP):
            continue
        rows = list(csv.DictReader(open(fpath, encoding="utf-8")))
        if rows and "t_done" in rows[0]:
            yield fpath, b, rows
    for z in sorted(glob.glob(os.path.join(FRIENDS, "*", "排隊測試_*", "*.zip"))):
        zz = zipfile.ZipFile(z)
        for nm in zz.namelist():
            if nm.startswith("queue_"):
                yield z, f"{os.path.basename(os.path.dirname(os.path.dirname(z)))[:2]} 台 {nm}", \
                    list(csv.DictReader(io.StringIO(zz.read(nm).decode("utf-8"))))


def sec_generator():
    pw = json.load(open(os.path.join(EXP, "plan_win.json"), encoding="utf-8"))          # 10/28：III-A 補的 M1／HP 排隊組
    tc = open(os.path.join(EXP, "prompts", "token_counts.txt"), encoding="utf-8").read()
    n_short = re.search(r"short\.txt\s+目標 \d+ tokens\s+實際 (\d+) tokens", tc).group(1)
    C("III-A", f"with a {n_short}-token prompt on the HP laptop" if pw["prompt_file"] == "short.txt" else "（HP 不是用 short.txt）",
      "plan_win.json（prompt_file）、prompts/token_counts.txt", "紀錄")
    tags = lambda p: [r["tag"] for r in json.load(open(os.path.join(EXP, p), encoding="utf-8"))["runs"]]
    ok = tags("plan_m1.json") == ["m1_rho30", "m1_rho50", "m1_rho70"] and tags("plan_win.json") == ["win_rho30", "win_rho50", "win_rho70"]
    C("III-A", "the M1 and the MX250 also ran queueing runs at ρ = 0.3, 0.5 and 0.7" if ok
      else "（M1／HP 的排隊組不對）", "plan_m1.json、plan_win.json（runs）", "紀錄")
    late01 = code_has(os.path.join(EXP, "analyze_strengthen.py"), r"max_lateness_s'\]\) < 0\.01")
    late1 = code_has(os.path.join(KIT, "queue_test.py"), r'max_lateness_s"\] <= 1\.0 and res\["late_gt_1s"\] == 0')
    C("III-B", "we detect this as a send late by more than 0.01 s on the RTX 4050, or 1 s in the volunteer kit",
      "analyze_strengthen.py（送出延誤 < 0.01 秒）、queue_test.py（late_gt_1s）", "紀錄" if late01 and late1 else "⚠️")
    lates = []                     # 10/28：留下的 LLM 排隊組送出延誤最多多少（不含假伺服器的驗證組 my_／val_，和睡著作廢的 var_C_rho75）
    for p in glob.glob(os.path.join(OUT, "*.csv")):
        b = os.path.basename(p)
        rows = [] if b.startswith(("my_", "val_")) or b == "var_C_rho75.csv" else list(csv.DictReader(open(p, encoding="utf-8")))
        if rows and "t_scheduled" in rows[0]:
            lates += [float(r["t_send"]) - float(r["t_scheduled"]) for r in rows if r.get("t_send")]
    for z in glob.glob(os.path.join(FRIENDS, "*", "排隊測試_*", "*.zip")):
        zz = zipfile.ZipFile(z)
        for nm in [x for x in zz.namelist() if x.startswith("queue_")]:
            lates += [float(r["t_send"]) - float(r["t_scheduled"])
                      for r in csv.DictReader(io.StringIO(zz.read(nm).decode("utf-8"))) if r.get("t_send")]
    C("III-B", f"no send in the runs we kept was over {int(np.ceil(1000 * max(lates)))} ms late",
      "out/*.csv 的 t_send − t_scheduled（不含 my_／val_、var_C_rho75）、志願者排隊測試 zip")
    n_req, swaps, gaps = 0, 0, []
    for _, name, rows in queue_files():
        ok = sorted((r for r in rows if r.get("status", "ok") == "ok"), key=lambda r: float(r["t_send"]))
        n_req += len(rows)
        a = np.array([float(r["t_send"]) for r in ok])
        c = np.array([float(r["t_done"]) for r in ok])
        sw = np.where(c[1:] < c[:-1])[0]
        swaps += len(sw)
        gaps += list(a[sw + 1] - a[sw])
    C("III-C", f"requests sent within {1000 * max(gaps):.0f} ms of each other were sometimes served in reverse "
               f"({swaps} of the {n_req:,} requests in all our queueing runs)",
      "所有排隊類的請求紀錄（out/*.csv 有 t_done 的檔案，含假伺服器的驗證組；志願者排隊測試 zip）；照送出時間排序後，做完時間比前一個早的次數")


# ------------------------------------------------------------------ III-D 閒置測試、志願者
def sec_probes():
    g = sorted({float(r["gap"]) for r in rc.probe("probe_idle_3b") if r["cond"] == "A"})
    C("III-D", "g = " + ", ".join(f"{x:g}" for x in g[:-1]) + f" and {g[-1]:g} s on the RTX 4050", "out/probe_idle_3b.csv（A 條件）", "紀錄")
    z = glob.glob(os.path.join(FRIENDS, "09_*", "閒置測試結果_GPU_*.zip"))[0]
    zz = zipfile.ZipFile(z)
    req = list(csv.DictReader(io.StringIO(zz.read("requests.csv").decode("utf-8"))))
    per_round = {n for n in __import__("collections").Counter(r["round"] for r in req).values()}
    gv = sorted({float(r["gap"]) for r in req})
    C("III-D", "with g = " + ", ".join(f"{x:g}" for x in gv[:-1]) + f" and {gv[-1]:g} s and {per_round.pop()} requests per round",
      "志願者測試包的 requests.csv（第 9 台；每輪的請求數都一樣）", "紀錄" if len(per_round) == 0 else "⚠️")
    warm = re.search(r"^WARMUP = (\d+)", open(os.path.join(KIT, "idle_test.py"), encoding="utf-8").read(), re.M).group(1)
    rounds = set()
    for zp in glob.glob(os.path.join(FRIENDS, "*", "閒置測試結果_*.zip")):
        zf = zipfile.ZipFile(zp)
        ri = [n for n in zf.namelist() if n.endswith("run_info.json")]
        if ri:
            rounds.add(json.loads(zf.read(ri[0])).get("rounds"))
    rounds.discard(None)
    word = {"3": "three"}.get(warm, warm)
    # 10/28：每輪先重開伺服器、再送暖機請求（Windows 版和 Mac 版都檢查）
    src_w = open(os.path.join(KIT, "idle_test.py"), encoding="utf-8").read()
    src_m = open(os.path.join(BASE, "朋友測試包", "Mac版", "mac_idle_test.sh"), encoding="utf-8").read()
    order_ok = all(0 <= x.find("server.start()") < x.find("for i in range(WARMUP)") for x in (src_w, src_m))
    C("III-D", f"Each round restarted the server and sent {word} warm-up requests, and the volunteers chose "
               f"{' or '.join(str(x) for x in sorted(rounds))} rounds before seeing any result" if order_ok
      else "（重開伺服器和暖機的順序不對）",
      "idle_test.py、Mac版/mac_idle_test.sh（server.start() 在暖機迴圈之前；WARMUP）、各測試包 run_info.json 的 rounds", "紀錄")
    # 10/28：CPU 測試用短的請求（和 HP 筆電同一個 short.txt）；看程式的設定，也看每份志願者 CPU 原始資料的 out_tokens
    wl = re.search(r'"short": \("short\.txt", (\d+)\)', src_w).group(1)
    cpu_short = re.search(r'workload = "main" if mode == "gpu" else "short"', src_w) is not None
    same_prompt = (open(os.path.join(KIT, "..", "prompts", "short.txt"), "rb").read()
                   == open(os.path.join(EXP, "prompts", "short.txt"), "rb").read())
    tc = open(os.path.join(EXP, "prompts", "token_counts.txt"), encoding="utf-8").read()
    n_short = re.search(r"short\.txt\s+目標 \d+ tokens\s+實際 (\d+) tokens", tc).group(1)
    toks, n_cpu = set(), 0
    for zp in glob.glob(os.path.join(FRIENDS, "*", "閒置測試結果_CPU_*.zip")):
        zz = zipfile.ZipFile(zp)
        rq = [n for n in zz.namelist() if n.endswith("requests.csv")]
        toks |= {r["out_tokens"] for r in csv.DictReader(io.StringIO(zz.read(rq[0]).decode("utf-8-sig")))}
        n_cpu += 1
    C("III-D", f"the same requests (on CPUs, a {n_short}-token prompt and {wl} output tokens)"
      if cpu_short and same_prompt and n_cpu > 0 and toks == {wl} else f"（CPU 測試的請求不對：{toks}）",
      f"idle_test.py（CPU 測試用 short：WORKLOADS 和 workload 那一行）、_kit/prompts/short.txt 和 LLM延遲實驗/prompts/short.txt 相同、"
      f"token_counts.txt、{n_cpu} 份志願者 CPU 測試 zip 的 out_tokens 全部是 {wl}", "紀錄")
    af = open(os.path.join(ANALYSIS, "analyze_friend.py"), encoding="utf-8").read()
    cpu = re.search(r"cpu > (\d+)", af).group(1)
    gpu = re.search(r"gpu_u > (\d+)", af).group(1)
    C("III-D", f"(failed requests, battery power, CPU above {cpu}% or GPU above {gpu}% while waiting, and stalls)",
      "分析程式/analyze_friend.py 的排除規則", "紀錄")
    cal_n, gaps, loads, machines = set(), set(), set(), set()
    for zp in glob.glob(os.path.join(FRIENDS, "*", "排隊測試_*", "*.zip")):
        info = json.loads(zipfile.ZipFile(zp).read("run_info.json"))
        cal_n.add(info["calibration"]["n"])
        gaps |= set(info["cal_gaps"])
        loads |= {x[1] for x in info["loads"]}
        machines.add(os.path.basename(os.path.dirname(os.path.dirname(zp)))[:2])
    nrun = len(glob.glob(os.path.join(FRIENDS, "*", "排隊測試_*", "*.zip")))
    words = {3: "Three", 2: "Two", 4: "Four"}
    twice = nrun - len(machines)
    C("III-D", f"{words[len(machines)]} desktops also ran a queue kit, {'one of them twice' if twice == 1 else f'{twice} twice'}: "
               f"a calibration of {' or '.join(str(x) for x in sorted(cal_n))} requests with idle times of {min(gaps):g}–{max(gaps):g} s, "
               f"then Poisson loads at ρ = " + ", ".join(f"{x:g}" for x in sorted(loads)[:-1]) + f" and {sorted(loads)[-1]:g} with fixed seeds",
      f"志願者排隊測試 zip 的 run_info.json（{nrun} 份、{len(machines)} 台）", "紀錄")


# ------------------------------------------------------------------ III-E、III-F、IV
def sec_statistics(waiting):
    C("III-E", f"discarding a quarter or a half instead changed the mean absolute errors little ({waiting['sens_mod']}% and "
               f"{waiting['sens_iid']}% for the setup-time and standard models)", "重跑 analyze_strengthen.py（WARM_DIV=4、2）")
    bt = re.search(r"^BATCHES = (\d+)", open(os.path.join(EXP, "analyze_strengthen.py"), encoding="utf-8").read(), re.M).group(1)
    C("III-E", f"so 95% confidence intervals of the mean wait come from {bt} batch means", "analyze_strengthen.py（BATCHES、T975_9）", "紀錄")
    agree, n, diff = friend_round_agreement()
    C("III-E", f"which gave the same conclusion in {agree} of {n} device tests",
      f"重跑 summarize_all.py（暫存資料夾）：每組「以輪算」和「以請求算」的 95% CI 是否都排除 0；不同的：{'、'.join(diff) or '沒有'}")
    fk = FAKE
    C("III-F", f"its mean waits of {f3(fk['w'][0])} s at ρ = 0.5 and {f3(fk['w'][1])} s at ρ = 0.8 fell inside the 95% ranges of "
               f"simulations with the same arrival times, whose means were {f3(fk['sim'][0])} and {f3(fk['sim'][1])} s "
               f"(the P-K values are {f3(fk['pk'][0])} and {f3(fk['pk'][1])} s); its idle effect was only "
               f"{f2(min(fk['pen']))}–{f2(max(fk['pen']))} s" if fk["inside"] else "（假伺服器有一組不在範圍內）",
      "out/my_rho50.csv、my_rho80.csv；稽核/check_outsample.py")
    rv = os.path.join(EXP, "run_validation.py")
    mean = float(re.search(r'"--mean", type=float, default=([\d.]+)', open(rv, encoding="utf-8").read()).group(1))
    spread = float(re.search(r'"--spread", type=float, default=([\d.]+)', open(rv, encoding="utf-8").read()).group(1))
    C("III-F", f"A fake server with service times uniform on [{mean - spread:.2f}, {mean + spread:.2f}] s validated the tool. "
               "With the first 20 requests discarded",
      "run_validation.py（--dist uniform、--mean、--spread 的預設值）；丟前 20 個：recompute.py sec_validation", "紀錄"
      if code_has(rv, r'"--dist", "uniform"') else "⚠️")
    R = re.search(r"^R = (\d+)", open(os.path.join(EXP, "analyze_strengthen.py"), encoding="utf-8").read(), re.M).group(1)
    C("IV-C", f"we simulate each run {int(R):,} times with its own arrival times", "analyze_strengthen.py（R）", "紀錄")
    e = re.search(r"^EDGES = \[([\d., ]+)\]", open(os.path.join(EXP, "analyze_strengthen.py"), encoding="utf-8").read(), re.M).group(1)
    e = [float(x) for x in e.split(",")]
    C("IV-B", f"four idle-time bins (below {e[0]:g} s, {e[0]:g}–{e[1]:g} s, {e[1]:g}–{e[2]:g} s, and {e[2]:g} s or more)",
      "analyze_strengthen.py（EDGES）", "紀錄")


def fake_server():
    import check_outsample as co
    with contextlib.redirect_stdout(io.StringIO()):
        fk = co.fake_server()
    out = {"w": [], "pk": [], "pen": [], "sim": [x[2] for x in fk], "inside": all(x[5] for x in fk)}
    for name, lam in (("my_rho50", 1.0), ("my_rho80", 1.6)):
        d = rc.run(name)
        out["w"].append(d["W"][20:].mean())
        out["pk"].append(lam * 0.2708 / (2 * (1 - lam * 0.5)))
        idle, q = d["W"] == 0, d["W"] > 0
        idle[0] = False
        out["pen"].append(d["S"][idle].mean() - d["S"][q].mean())
    return out


def friend_round_agreement():
    """每組志願者閒置測試（all_results.csv 的列）：以請求算、以輪算的代價 95% CI 有沒有排除 0（下限 > 0），
    兩者結論相同的組數（和 朋友結果/彙整/彙整.md 的定義相同：10/2x 唯一不同的是第 4 台 CPU）"""
    rows = list(csv.DictReader(open(SUMMARY, encoding="utf-8-sig")))
    use = [r for r in rows if r["輪CI_lo"] not in ("", "nan")]
    diff = [f"{r['機器']} {r['測試']}" for r in use if (float(r["CI_lo"]) > 0) != (float(r["輪CI_lo"]) > 0)]
    return len(use) - len(diff), len(use), diff


# ------------------------------------------------------------------ V-A 閒置代價
def clock_at_send(probe_name, cond, gaps):
    rows = rc.probe(probe_name)
    g = []
    for line in open(os.path.join(OUT, f"{probe_name}_gpu.csv"), encoding="utf-8", errors="replace"):
        p = [x.strip() for x in line.split(",")]
        try:
            g.append((datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp(), p[1], float(p[2])))
        except (ValueError, IndexError):
            continue
    g.sort()
    gt, gc = np.array([x[0] for x in g]), np.array([x[2] for x in g])
    vals = [gc[max(0, np.searchsorted(gt, float(r["wall_send"])) - 1)] for r in rows if r["cond"] == cond and float(r["gap"]) in gaps]
    return np.mean(vals), g


def sec_idle():
    probes = ["probe_idle_3b", "probe_idle_3b_nolog", "probe_idle_3b_heavy"]
    pg = {p: rc.per_gap(p) for p in probes}
    small = max(abs(v) for p in probes for gg, v in pg[p].items() if 0 < gg <= 1.5)
    g2 = [pg[p][2.0] for p in probes if 2.0 in pg[p]]
    per_run = [np.mean([v for gg, v in pg[p].items() if gg >= 3]) for p in probes]
    avg = np.mean(per_run)
    b0 = np.mean([rc.probe_extra(p)[0] for p in probes])
    t = stats.t.ppf(0.975, len(per_run) - 1) * np.std(per_run, ddof=1) / np.sqrt(len(per_run))
    C("V-A", ("Up to an idle time of 1.5 s there was none. " if small < 0.05 else "（g ≤ 1.5 秒有代價）") +
      f"It appeared at 2 s ({f2(min(g2))}–{f2(max(g2))} s) and averaged {f2(avg)} s for 3 s or more, about {pct(avg / b0)} of the "
      f"back-to-back service time (95% CI {f2(avg - t)}–{f2(avg + t)} s over three probe runs)",
      "out/probe_idle_3b*.csv（三次閒置測試；CI 用三次的平均、t 分布）")
    tt = [rc.per_gap(p, key="ttft") for p in probes]
    big = [v for p in probes for gg, v in pg[p].items() if gg >= 3]
    share = np.mean([tt[i][gg] for i in range(3) for gg in pg[probes[i]] if gg >= 3]) / np.mean(big)
    C("V-A", "Almost all of it fell before the first output token, that is, in the prefill" if share > 0.85 else "（第一個 token 前的比例不到 85%）",
      f"同上（第一個 token 前佔 {share:.0%}）")
    c15, g = clock_at_send("probe_idle_3b", "A", (0, 0.5, 1, 1.5))
    c3, _ = clock_at_send("probe_idle_3b", "A", (3,))
    # 閒置開始後幾秒開始降頻：g = 8 秒的閒置裡，每 0.1 秒的時脈中位數第一次低於 2,400 MHz
    gt, gp, gc = np.array([x[0] for x in g]), np.array([x[1] for x in g]), np.array([x[2] for x in g])
    rows = [r for r in rc.probe("probe_idle_3b") if r["cond"] == "A" and float(r["gap"]) == 8]
    bins = np.arange(0, 8, 0.1)
    med = []
    for lo in bins:
        v = []
        for r in rows:
            t_end = float(r["wall_send"]) - float(r["idle_before"])
            sel = (gt >= t_end + lo) & (gt < t_end + lo + 0.1)
            v += list(gc[sel])
        med.append(np.median(v) if v else np.nan)
    drop = bins[np.argmax(np.array(med) < 2400)]
    drops = []
    for r in rc.probe("probe_idle_3b"):
        if r["cond"] != "A" or float(r["gap"]) < 3:
            continue
        t_end = float(r["wall_send"]) - float(r["idle_before"])
        j = np.searchsorted(gt, t_end)
        k = next((i for i in range(j, len(gt)) if gt[i] < float(r["wall_send"]) and gp[i] == "P8"), None)
        if k is not None:
            drops.append(gt[k] - t_end)
    p8 = np.median(gc[gp == "P8"])
    C("V-A", f"about {drop:.0f} s after the last request ended, the GPU began to lower its clock from about {round(c15, -1):,.0f} MHz; "
             f"it ran at about {round(c3, -2):,.0f} MHz at 3 s and entered its idle state P8 ({p8:.0f} MHz) after a median of "
             f"{np.median(drops):.1f} s",
      f"out/probe_idle_3b_gpu.csv（g = 8 秒的閒置裡時脈中位數第一次低於 2,400 MHz：{drop:.1f} 秒）")
    kc = [clock_at_send("probe_idle_3b", "K", (x,))[0] for x in (3, 5, 8)]
    ka = np.mean([v for gg, v in rc.per_gap("probe_idle_3b", "K").items() if gg >= 3])
    aa = np.mean([v for gg, v in pg["probe_idle_3b"].items() if gg >= 3])
    C("V-A", f"the clock still fell to about {round(min(kc) / 50) * 50:,.0f}–{round(max(kc) / 50) * 50:,.0f} MHz, and the penalty for 3 s or more "
             f"fell only from {f2(aa)} to {f2(ka)} s",
      f"out/probe_idle_3b.csv、_gpu.csv（K 條件；閒置 3、5、8 秒送出時的時脈平均 {'、'.join(f'{x:,.0f}' for x in kc)} MHz）")
    hc = clock_at_send("probe_idle_3b_heavy", "H", (3, 5, 8))[0]
    hbig = np.mean([v for gg, v in rc.per_gap("probe_idle_3b_heavy", "H").items() if gg >= 3])
    C("V-A", f"held the clock at about {round(hc, -1):,.0f} MHz, and the penalty vanished ({neg(f'{hbig:.3f}')} s)",
      f"out/probe_idle_3b_heavy.csv、_gpu.csv（H 條件；閒置 ≥3 秒送出時的時脈平均 {hc:,.1f} MHz）")
    b0_, d0, _, _ = rc.probe_extra("probe_base_1011")
    b1, d1, lo1, hi1 = rc.probe_extra("probe_locked_1011")
    clk, mem = [], []
    for line in open(os.path.join(OUT, "probe_locked_1011_gpu.csv"), encoding="utf-8", errors="replace"):
        p = [x.strip() for x in line.split(",")]
        try:
            clk.append(float(p[2]))
            mem.append(float(p[3]))
        except (ValueError, IndexError):
            pass
    C("V-A", f"Locking the GPU core and memory clocks at {max(clk):,.0f} and {max(mem):,.0f} MHz had the same effect: on the same day, "
             f"idle periods of 3 s or more added {f2(d0)} s without the lock and {f3(d1)} s with it (95% CI {neg(f3(lo1))} to {f3(hi1)} s)",
      "out/probe_base_1011.csv、probe_locked_1011.csv（Welch 95% CI）、_gpu.csv")
    sys.path.insert(0, ANALYSIS)
    import analyze_friend as AF
    pen = {}
    for be in ("cuda", "vulkan"):
        with contextlib.redirect_stdout(io.StringIO()):
            _, r = AF.analyze(os.path.join(OUT, f"backend_20261022-0041_{be}.zip"))
        pen[be] = float(r["d"])
    C("V-A", f"the CUDA build of the same release showed {f3(pen['cuda'])} s against {f3(pen['vulkan'])} s for Vulkan, "
             f"a ratio of {pen['cuda'] / pen['vulkan']:.2f}", "out/backend_20261022-0041_cuda.zip、_vulkan.zip（analyze_friend.analyze）")
    p7 = rc.per_gap("probe_idle_7b16")[5.0]
    d7 = rc.probe_extra("probe_b7v2")[1]
    C("V-A", f"split between GPU and CPU ({f2(p7)} s after 5 s of idleness) and entirely on the GPU ({f2(d7)} s for 3 s or more)",
      "out/probe_idle_7b16.csv、probe_b7v2.csv")
    gaps, res, pools = [], [], {}
    for name in rc.A_RUNS + rc.C_RUNS:
        dd = rc.D[name]
        for key in sorted(set(dd["tok"])):
            q = (dd["W"] > 0) & (dd["tok"] == key)
            pools.setdefault(key, []).extend(dd["S"][q])
            idle = (dd["W"] == 0) & (dd["tok"] == key)
            idle[0] = False
            gaps.extend(dd["gap"][idle])
            res.extend(dd["S"][idle] - dd["S"][q].mean())
    gaps, res = np.array(gaps), np.array(res)
    b128 = np.mean(pools[128])
    C("V-A", f"Across the {len(rc.A_RUNS + rc.C_RUNS)} runs, requests that found the server idle took longer than back-to-back requests "
             f"({f2(b128)} s) by {f2(res[gaps < 1].mean())} s after idle periods shorter than 1 s, "
             f"{f2(res[(gaps >= 1) & (gaps < 2)].mean())} s after 1–2 s, {f2(res[(gaps >= 2) & (gaps < 5)].mean())} s after 2–5 s and "
             f"{f2(res[gaps >= 5].mean())} s after 5 s or more, "
             f"close to the {f2(avg)} s of the probes", "17 組 out/*.csv")
    ca, cc = rc.run("mix_cal_A"), rc.run("var_cal_C")
    iso_c = (cc["S"][cc["tok"] == 32].mean() + cc["S"][cc["tok"] == 224].mean()) / 2
    btb_c = (np.mean(pools[32]) + np.mean(pools[224])) / 2
    C("V-A", f"overstated the back-to-back service time by {pct(ca['S'].mean() / b128 - 1)} for workload A ({f2(ca['S'].mean())} versus "
             f"{f2(b128)} s) and by {pct(iso_c / btb_c - 1)} for workload C", "out/mix_cal_A.csv、var_cal_C.csv")
    gmax = max(float(r["gap"]) for r in rc.probe("probe_idle_3b"))
    C("Fig. 1", f"Gray bars: the penalty estimated from the {len(rc.A_RUNS + rc.C_RUNS)} queueing runs. (b) Median GPU clock during an "
                f"{gmax:g}-s idle period, logged about every {np.median(np.diff(gpu_times(os.path.join(OUT, 'probe_idle_3b_gpu.csv')))):.1f} s",
      "17 組；probe_idle_3b 最長的閒置；nvidia-smi 取樣間隔", "紀錄")
    C("Fig. 1", "(3B model; mean and 95% bootstrap interval over requests)",
      "analyze_probe.py 的 boot_diff（對個別請求重抽樣，2.5、97.5 百分位；Fig. 1 用它的 _summary.csv）",
      "紀錄" if code_has(os.path.join(EXP, "analyze_probe.py"), r"np\.percentile\(d, \[2\.5, 97\.5\]\)")
      and code_has(os.path.join(EXP, "analyze_probe.py"), r"rng\.choice\(x, \(b, len\(x\)\)\)") else "⚠️")
    ping = re.search(r"^PING_EVERY = ([\d.]+)", open(os.path.join(EXP, "probe_idle.py"), encoding="utf-8").read(), re.M).group(1)
    C("V-A", f"One-token requests every {ping} s were not enough", "probe_idle.py（PING_EVERY）", "紀錄")
    C("V-A", f"Requests that read the full prompt every {ping} s", "probe_idle.py（重保溫用同樣的間隔）", "紀錄")
    iso_a, iso_cc = ca["S"].mean() / b128 - 1, iso_c / btb_c - 1
    V.update(pen=f2(avg), share=pct(avg / b0), drop=f"{drop:.0f}", iso=f"{100 * min(iso_a, iso_cc):.0f}–{100 * max(iso_a, iso_cc):.0f}%")


# ------------------------------------------------------------------ V-B
def waiting():
    """重跑 analyze_strengthen.py：17 組樣本內的平均絕對誤差、改丟 1/4 或 1/2 的敏感度（和 recompute.sec_waiting 相同）"""
    with contextlib.redirect_stdout(io.StringIO()):
        rc.sec_waiting()
    got = {r["what"]: r["got"] for r in rc.RESULTS}
    mae = got["平均絕對誤差：一般 M/G/1／閒置效應模型"].split("／")
    sens = got["改丟前 1/4 或 1/2：考慮準備時間的模型／標準模型的平均絕對誤差"].split("／")
    return {"mae_iid": mae[0].rstrip("%"), "mae_mod": mae[1].rstrip("%"), "sens_mod": sens[0].rstrip("%"), "sens_iid": sens[1].rstrip("%")}


def sec_waiting(w):
    C("V-B", f"Fitted to the {len(rc.A_RUNS + rc.C_RUNS)} runs, the setup-time model matched their mean waiting times with a mean "
             f"absolute error of {w['mae_mod']}%, against {w['mae_iid']}% for the standard M/G/1 model", "重跑 analyze_strengthen.py")


# ------------------------------------------------------------------ V-C 事前預測、表 I、Fig. 2
WORDS = {1: "one", 2: "two", 3: "three", 5: "five", 6: "six", 7: "seven", 9: "nine", 17: "Seventeen"}


def sec_prereg():
    import 事前預測彙整 as pr
    with contextlib.redirect_stdout(io.StringIO()):
        rows = pr.collect()
    n = len(rows)
    e = np.array([abs(r["err"]) for r in rows])
    C("V-C", f"For {n} runs, we wrote down a prediction in advance", "稽核/事前預測彙整.py（各組當時的預測檔）", "紀錄")
    fit = [r["run"] for r in csv.DictReader(open(os.path.join(OUT, "strengthen_summary.csv"), encoding="utf-8"))]   # 10/28：擬合的 17 組
    b3 = [r for r in rows if r["device"] == "RTX 4050" and r["group"].startswith("3B")]
    b7 = [r for r in rows if r["device"] == "RTX 4050" and r["group"].startswith("7B")]
    ov = sum(r["tag"] in fit for r in b3)
    n_pl = len({r["group"] for r in b7})
    C("V-C", f"On the RTX 4050, these were {WORDS[len(b3)]} 3B runs ({WORDS[ov]} also among the {len(fit)} fitted ones) and "
             f"{WORDS[len(b7)]} 7B runs" if not any(r["tag"] in fit for r in b7) and n_pl == 2
      else "（7B 也在擬合的組裡？）", "稽核/事前預測彙整.py（各組的機器、組別）、out/strengthen_summary.csv（擬合的 17 組）")
    order = []                     # 10/24 晚：志願者的預測寫在測試跑完之後、比對實測（打開排隊資料）之前
    for d, pat, _ in pr.QUEUE:
        pf = glob.glob(os.path.join(FRIENDS, d, pat))[0]
        made = json.load(open(pf, encoding="utf-8"))["made_at"]
        res = json.load(open(os.path.join(os.path.dirname(pf), os.path.basename(pf).replace("預測_", "結果_")), encoding="utf-8"))
        end = json.loads(zipfile.ZipFile(glob.glob(os.path.join(FRIENDS, d, "*.zip"))[0]).read("run_info.json"))["end"]
        order.append(end < made < res["compared_at"] and res["pred_made_at"] == made)
    n_vol = sum(r["device"].startswith(("RTX 20", "RTX 30")) for r in rows)
    C("V-C", f"For the {n_vol} volunteer runs, we wrote it after the kit had run but before opening the data" if all(order)
      else "（時間順序不對）", "預測檔的 made_at、結果檔的 compared_at、排隊測試 zip 的 run_info.json（end）；流程見 朋友測試包/排隊測試/設計說明.md",
      "紀錄")
    C("V-C", f"Across the {n} runs, the median error was {100 * np.median(e):.1f}% (mean {100 * e.mean():.1f}%), and "
             f"{int((e <= 0.10).sum())} predictions were within 10%", "稽核/事前預測彙整.py（實測照做完的順序重建）")
    hw = np.array([(r["hi"] - r["lo"]) / 2 / r["pred"] for r in rows])
    desk = np.array([r["device"].startswith(("RTX 20", "RTX 30")) for r in rows])
    hit = np.array([r["hit"] for r in rows])
    C("V-C", f"The ranges were narrow, with a median half-width of {100 * np.median(hw):.0f}% ({100 * np.median(hw[desk]):.1f}% on the "
             f"volunteer desktops). {WORDS.get(hit.sum(), hit.sum())} of the {n} measurements fell inside them, and "
             f"{int((e[~hit] < 0.10).sum())} of the {int((~hit).sum())} misses were off by less than 10%", "同上")
    stable = np.array([r["device"] in ("RTX 4050", "Apple M1") for r in rows])
    big = e > 0.10
    heat_ok = all(rows[i]["device"].startswith(("RTX 20", "RTX 30", "HP")) and re.search(r"rho(5|7)", rows[i]["tag"])
                  for i in np.where(big)[0])
    C("V-C", f"All {stable.sum()} predictions on the RTX 4050 and the M1 were within 10%, and all {WORDS[int(big.sum())]} errors larger "
             f"than 10% came from the HP laptop and the desktops at ρ ≈ 0.5 or more" if (e[stable] <= 0.10).all() and heat_ok else "（不成立）",
      "同上")
    closer = sum(r["closer"] for r in rows)
    be = [abs(r["base_err"]) for r in rows if np.isfinite(r["base_err"])]
    C("V-C", f"In {closer} of the {n} runs, the measurement was closer to the model than to the P-K formula listed with each prediction, "
             f"which used the mean service time for the 3B runs and the isolated-request time for the others and was off by "
             f"{100 * np.mean(be):.0f}% on average", "同上（對照公式算不出來的一組不算平均）")
    p = stats.binomtest(closer, n, 0.5).pvalue
    es, ea = np.array([r["err"] for r in rows if r["device"] in ("RTX 4050", "Apple M1")]), np.array([r["err"] for r in rows])
    ba = lambda x: (x.mean(), x.mean() - 1.96 * x.std(ddof=1), x.mean() + 1.96 * x.std(ddof=1))
    (b1, l1, h1), (b2, l2, h2) = ba(es), ba(ea)
    sg = lambda x, nd=0: neg(f"{100 * x:+.{nd}f}%")
    C("V-C", f"a sign test on these {closer} of {n} gives p < 10⁻⁸, and the Bland–Altman bias of the relative error is {sg(b1, 1)} on the "
             f"RTX 4050 and the M1 (95% limits of agreement {neg(f'{100 * l1:.0f}%')} to {sg(h1)}) and {sg(b2, 1)} overall "
             f"({neg(f'{100 * l2:.0f}%')} to {sg(h2)})" if p < 1e-8 else "（符號檢定 p ≥ 10⁻⁸）",
      f"同上（符號檢定 p = {p:.1g}；Bland–Altman：誤差 % 的平均 ± 1.96 標準差）")
    groups = [("RTX 4050", lambda d: d == "RTX 4050"), ("Apple M1", lambda d: d == "Apple M1"), ("HP laptop", lambda d: d.startswith("HP")),
              ("Volunteer desktops", lambda d: d.startswith(("RTX 20", "RTX 30"))), ("All", lambda d: True)]
    for name, fn in groups:
        s = [r for r in rows if fn(r["device"])]
        ee = np.array([abs(r["err"]) for r in s]) * 100
        fm = lambda x: f"{x:.1f}%" if x < 10 else f"{x:.0f}%"
        C("表 I", f"{name} ; {len(s)} ; {fm(ee.mean())} / {fm(np.median(ee))} ; {int((ee <= 10).sum())} ; {sum(r['hit'] for r in s)} ; "
                  f"{sum(r['closer'] for r in s)}", "同上")
    C("表 I", "Within^10%", "表頭：誤差 ≤ 10% 的組數", "紀錄")
    V.update(n_pre=n, median=f"{100 * np.median(e):.1f}%", closer=closer, hits=int(hit.sum()))
    ins = list(csv.DictReader(open(os.path.join(OUT, "strengthen_summary.csv"), encoding="utf-8")))
    n_inf = sum(not np.isfinite(r["base"]) for r in rows)
    fig = open(os.path.join(PAPER, "make_prereg_fig2.py"), encoding="utf-8").read()
    band = "lo_ax * 0.9" in fig and "hi_ax * 1.1" in fig
    C("Fig. 2", f"the {n} predictions written down in advance, with their simulation ranges", "稽核/事前預測彙整.csv", "紀錄")
    C("Fig. 2", f"Hollow circles: the {len(ins)} runs the model was fitted to. Crosses: the P-K formula listed with each prediction "
                f"({WORDS[n_inf]}, which diverges at its nominal load, is not shown). Gray band: ±10%",
      "out/strengthen_summary.csv、事前預測彙整.csv（對照公式是無限大的組）、make_prereg_fig2.py（帶子 0.9–1.1 倍）", "紀錄" if band else "⚠️")
    return rows


# ------------------------------------------------------------------ V-E 會變慢的機器
def queue_check(zp, pred):
    """重跑 分析程式/check_queue_runs.py 的檢查（讀原始 zip）：每組排隊中的服務時間 ÷ 校正 b、四段的溫度和時脈、自己的服務時間重新模擬"""
    sys.path.insert(0, ANALYSIS)
    import check_queue_runs as cq
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cq.main(zp, pred)
    out, tag = {}, None
    for line in buf.getvalue().splitlines():
        m = re.match(r"== (rho\d+)", line)
        if m:
            tag = m.group(1)
            out[tag] = {}
        elif tag and "校正 b 的" in line:
            out[tag]["ratio"] = float(re.search(r"校正 b 的 ([\d.]+) 倍", line).group(1))
        elif tag and line.strip().startswith("照時間分四段：時脈"):
            out[tag]["clock"] = [float(x.replace(",", "")) for x in re.search(r"時脈 ([\d、,]+) MHz", line).group(1).split("、")]
            out[tag]["temp"] = [float(x) for x in re.search(r"溫度 ([\d、]+)°C", line).group(1).split("、")]
        elif tag and "重新模擬" in line:
            m = re.search(r"重新模擬：([\d.]+).*實測 ([\d.]+)｜原本的預測 ([\d.]+)", line)
            out[tag]["resim"], out[tag]["meas"], out[tag]["pred"] = map(float, m.groups())
    return out


def resim_own_services(tag, rng, n_sim=2000):
    """HP 那幾組：用這組實際的送出時間，服務時間從這組自己量到的全部服務時間抽樣（一般 M/G/1，HP 沒有閒置代價），平均等待（丟前 1/3）"""
    d = rc.run(tag)
    a, S, k = d["a"], d["S"], d["n"] // 3
    draws = rng.choice(S, size=(n_sim, len(a)))
    done, acc = np.full(n_sim, -np.inf), np.zeros(n_sim)
    for i in range(len(a)):
        st = np.maximum(a[i], done)
        if i >= k:
            acc += st - a[i]
        done = st + draws[:, i]
    return float(acc.mean() / (len(a) - k))


def sec_slowdown():
    qs = {}
    for zp in sorted(glob.glob(os.path.join(FRIENDS, "*", "排隊測試_*", "*.zip"))):
        d = os.path.dirname(zp)
        pred = (glob.glob(os.path.join(d, "預測_方法1_*.json")) or glob.glob(os.path.join(d, "預測_2*.json")))[0]
        qs[os.path.relpath(d, FRIENDS)] = queue_check(zp, pred)
    r70 = {k: v["rho70"] for k, v in qs.items()}
    drift = [v["ratio"] - 1 for v in r70.values()]
    pw = json.load(open(os.path.join(EXP, "plan_win.json"), encoding="utf-8"))
    hp70 = rc.run("win_rho70")
    _, W70, S70 = rc.win(hp70)
    hp_drift = S70[W70 > 0].mean() / pw["b"] - 1
    C("V-E", f"At ρ ≈ 0.7, queued requests took {100 * min(drift):.0f}–{100 * max(drift):.0f}% longer than the back-to-back service time "
             f"used in the prediction on the desktops and {100 * hp_drift:.0f}% longer on the HP laptop",
      "重跑 check_queue_runs.py（志願者 4 次排隊測試的 ρ = 0.7）；out/win_rho70.csv ÷ plan_win.json 的 b")
    m9 = next(v for k, v in r70.items() if k.startswith("09_"))
    C("V-E", f"During its ρ ≈ 0.7 run, the RTX 3060 Ti warmed from {m9['temp'][0]:.0f} to {m9['temp'][-1]:.0f} °C, and its clock fell "
             f"from {m9['clock'][0]:,.0f} to {m9['clock'][-1]:,.0f} MHz", "同上（第 9 台 ρ = 0.7：忙的時候第一段和最後一段的中位數）")
    rec = [(v["resim"] - v["pred"]) / (v["meas"] - v["pred"]) for v in r70.values()]
    rng = np.random.default_rng(8)
    hp_rec = []
    for r in pw["runs"]:
        if r["tag"] in ("win_rho50", "win_rho70"):
            meas = rc.wq(rc.run(r["tag"]))
            hp_rec.append((resim_own_services(r["tag"], rng) - r["pred"]) / (meas - r["pred"]))
    C("V-E", f"resimulating each run with its own queued service times recovers {100 * min(rec):.0f}–{100 * max(rec):.0f}% of the error "
             f"on the desktops at ρ ≈ 0.7, and {100 * min(hp_rec):.0f}–{100 * max(hp_rec):.0f}% on the HP laptop",
      "重跑 check_queue_runs.py（桌機）；HP：照實際到達時間、從那一組自己的服務時間抽樣模擬 2,000 次（ρ ≈ 0.5、0.7）")
    # 方法 2（10/20 登記）：登記過的 9 組
    e1, e2, h1, h2 = [], [], 0, 0
    for k in qs:
        if k.startswith("04_"):
            continue                                       # 第 4 台的方法 2 是看過結果才想到的，不算登記過的
        d = os.path.join(FRIENDS, k)
        zp = glob.glob(os.path.join(d, "*.zip"))[0]
        p1 = {r["tag"]: r for r in json.load(open((glob.glob(os.path.join(d, "預測_方法1_*.json")) or
                                                   glob.glob(os.path.join(d, "預測_2*.json")))[0], encoding="utf-8"))["runs"]}
        p2 = {r["tag"]: r for r in json.load(open(glob.glob(os.path.join(d, "預測_方法2_*.json"))[0], encoding="utf-8"))["runs"]}
        import 事前預測彙整 as pr
        for tag in p1:
            m = pr.queue_wait(zp, tag)
            e1.append(abs(p1[tag]["pred"] - m) / m)
            e2.append(abs(p2[tag]["pred"] - m) / m)
            h1 += p1[tag]["lo"] <= m <= p1[tag]["hi"]
            h2 += p2[tag]["lo"] <= m <= p2[tag]["hi"]
    V.update(m2=(f"{100 * np.mean(e1):.1f}%", f"{100 * np.mean(e2):.1f}%"), m2n=WORDS[len(e1)])
    C("V-E", f"over the {WORDS[len(e1)]} later runs, it cut the mean error from {100 * np.mean(e1):.1f}% to "
             f"{100 * np.mean(e2):.1f}%, with {h2} of the {len(e1)} measurements inside their ranges against {h1}",
      "志願者排隊測試（第 6 台兩次、第 9 台）的方法 1、方法 2 預測檔和原始 zip")
    # 10/28：登記時間（設計說明.md 各節標題寫的時間）和每次排隊測試的開始時間（run_info.json 的 start）
    design = open(os.path.join(BASE, "朋友測試包", "排隊測試", "設計說明.md"), encoding="utf-8").read()
    tm = lambda pat: "T".join(re.search(pat, design).groups())
    t2 = tm(r"第二種預測方法（(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}) 寫")
    t3 = tm(r"方法 3 的事前登記（(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}) 寫")
    t5 = tm(r"## (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}) 加：方法改成 1–5")
    sess = []
    for zp in glob.glob(os.path.join(FRIENDS, "*", "排隊測試_*", "*.zip")):
        zz = zipfile.ZipFile(zp)
        n_runs = len({n for n in zz.namelist() if n.startswith("queue_")})
        sess.append((json.loads(zz.read("run_info.json"))["start"], n_runs, os.path.dirname(zp)))
    sess.sort()
    cmp1 = json.load(open(glob.glob(os.path.join(sess[0][2], "結果_2*.json"))[0], encoding="utf-8"))["compared_at"]
    later = sess[1:]
    ok2 = (os.path.basename(os.path.dirname(sess[0][2])).startswith("04_") and cmp1 < t2 < later[0][0]
           and sum(n for _, n, _ in later) == len(e1))
    C("V-E", "A variant registered after the first volunteer desktop's runs" if ok2 else "（方法 2 的登記時間不對）",
      f"設計說明.md（方法 2 登記 {t2}）；第一次（第 4 台）打開結果 {cmp1}；之後各次開始 {later[0][0][:16]} 起，共 {len(e1)} 組", "紀錄")
    last = [x for x in sess if x[0] > t5]
    ok5 = len(last) == 2 and all(x[0] < t3 for x in sess if x not in last)
    C("V-E", f"Three further refinements, registered before the last {WORDS[sum(n for _, n, _ in last)]} volunteer runs"
      if ok5 else "（方法 3–5 的登記時間不對）",
      f"設計說明.md（方法 3 登記 {t3}、方法 5 登記 {t5}，最晚的）；最後兩次測試開始 {last[0][0][:16]}、{last[-1][0][:16]}；"
      f"之前各次都在 {t3} 之前", "紀錄")


# ------------------------------------------------------------------ V-D 其他硬體、表 II
def sec_hardware():
    rows = list(csv.DictReader(open(SUMMARY, encoding="utf-8-sig")))
    # 每張顯示卡（第 3 台兩種電源設定合成一張）：代價、佔 b 的比例、明確開始；RTX 4050、M4 和最早那批從 recompute.py 的閒置測試算
    dev = {}
    for r in rows:
        key = (r["機器"], r["測試"])
        dev.setdefault(key, []).append(r)
    gpus, cpus = [], []
    for (m, t), rs in dev.items():
        pen = [float(r["代價"]) for r in rs]
        share = [float(r["比例"]) for r in rs]
        on = [float(r["明確開始"]) for r in rs if r["明確開始"] not in ("沒有", "")]
        # 「有代價」和「明確開始」用 summarize_all.py 的同一個標準：某個閒置秒數的 CI 下限 > 0 而且平均 ≥ 20 毫秒（第 175 行）
        item = {"name": rs[0]["裝置"], "pen": pen, "share": share, "onset": on, "has": len(on) > 0, "machine": m}
        (gpus if t == "GPU" else cpus).append(item)
    # RTX 4050：三次閒置測試合併（和 recompute.py 表 I 相同的反變異數加權）
    est = []
    for p in ("probe_idle_3b", "probe_idle_3b_nolog", "probe_idle_3b_heavy"):
        r = [x for x in rc.probe(p) if x["cond"] == "A"]
        s0 = np.array([float(x["service"]) for x in r if float(x["gap"]) == 0])
        s3 = np.array([float(x["service"]) for x in r if float(x["gap"]) >= 3])
        est.append((s0.mean(), s3.mean() - s0.mean(), np.sqrt(s3.var(ddof=1) / len(s3) + s0.var(ddof=1) / len(s0))))
    w = np.array([1 / se ** 2 for *_, se in est])
    d4050 = sum(wi * d for wi, (_, d, _) in zip(w, est)) / w.sum()
    b4050 = np.mean([b for b, _, _ in est])
    gpus.append({"name": "RTX 4050 Laptop", "pen": [d4050], "share": [d4050 / b4050], "onset": [2.0], "has": True, "machine": "author"})
    bm4, dm4, lm4, hm4 = rc.probe_extra("probe_m4")
    on4 = probe_onset("probe_m4")
    gpus.append({"name": "Apple M4", "pen": [dm4], "share": [dm4 / bm4], "onset": on4, "has": len(on4) > 0, "machine": "M4"})
    nv = [g for g in gpus if "RTX" in g["name"]]
    has = [g for g in nv if g["has"]]
    allsh = [s for g in has for s in g["share"]]
    V.update(nv=len(nv), nv_has=len(has), nv_vol=sum(g["machine"] != "author" for g in nv), sh_lo=pct(min(allsh)), sh_hi=pct(max(allsh)))
    C("V-D", f"Of the {len(nv)} NVIDIA RTX GPUs, {['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'][len(has)]} "
             f"showed a penalty after idle periods of 3 s or more, from {pct(min(allsh))} to {pct(max(allsh))} of the back-to-back service time",
      "重跑 summarize_all.py（志願者 9 張，CI 下限 > 0.02 秒算有代價）＋ RTX 4050（三次閒置測試合併）")
    gen = lambda g: re.search(r"RTX (\d)0", g["name"]).group(1) if "RTX" in g["name"] else "Apple"
    old = [g for g in has if gen(g) in ("2", "3", "4")]
    new = [g for g in has if gen(g) == "5"]
    osh = [s for g in old for s in g["share"]]
    nsh = [s for g in new for s in g["share"]]
    oon = {o for g in old for o in g["onset"]}
    non = sorted({o for g in new for o in g["onset"]})
    C("V-D", f"On the RTX 20, 30 and 40 series it was {100 * min(osh):.0f}–{100 * max(osh):.0f}% and appeared at "
             f"{'2 s' if oon == {2.0} else oon} of idleness; on {WORDS.get(len(new), len(new))} RTX 50-series GPUs it was "
             f"{100 * min(nsh):.0f}–{100 * max(nsh):.0f}% and became clear only at {non[0]:g}–{non[-1]:g} s", "同上（明確開始：summarize_all 的定義）")
    l4060 = [g for g in nv if "4060 Laptop" in g["name"]]
    ab = sorted(l4060, key=lambda g: -max(g["share"]))
    C("V-D", f"Two laptops with the same GPU model, an RTX 4060 Laptop, gave {100 * min(ab[0]['share']):.0f}–{100 * max(ab[0]['share']):.0f}% "
             f"and zero" if len(l4060) == 2 and not ab[1]["has"] else "（4060 Laptop 不是一台有、一台沒有）", "同上（第 3 台兩種電源設定、第 10 台）")
    csh = [s for c in cpus for s in c["share"]]
    C("V-D", f"On CPUs it was small everywhere, at most {100 * max(csh):.1f}% on {len(cpus)} machines ({sum(len(c['pen']) for c in cpus)} tests)",
      "重跑 summarize_all.py（CPU 測試）")
    m1 = next(g for g in gpus if g["machine"] == "M1")
    m1p = next(g for g in gpus if g["machine"] == "M1Pro")
    C("V-D", f"The fanless M1 Air was {-100 * m1['share'][0]:.0f}% faster after idle periods", "同上（M1 Air）")
    C("V-D", f"the M1 Pro, which has a fan, showed {100 * m1p['share'][0]:+.1f}%", "同上（M1 Pro）")
    with contextlib.redirect_stdout(io.StringIO()):
        rc.sec_hardware()
        rc.sec_round10()
    got = {r["what"]: r["got"] for r in rc.RESULTS}
    sl = [got[f"{n}：最後一個重保溫區塊比 A 區塊慢"] for n in ("M1", "M4")]
    C("V-D", f"Kept busy for several minutes, the fanless MacBooks slowed down by {sl[0]} (M1) and {sl[1]} (M4)",
      "recompute.py sec_hardware（probe_m1、probe_m4：最後一個重保溫區塊 ÷ A 區塊）")
    t16 = sorted({int(r["out_tokens"]) for p in ("probe_m4", "probe_wingpu", "probe_wincpu") for r in rc.probe(p)})
    C("V-D", f"The first checks of the M4, the MX250 and the CPU-only server sent requests over Wi-Fi with {t16[0]} output tokens",
      "out/probe_m4、probe_wingpu、probe_wincpu.csv", "紀錄" if len(t16) == 1 else "⚠️")
    net = float(got["Wi-Fi 本身閒置後多出的時間（最多）"].split()[0])
    short = re.match(r"([\d.]+) \(([\d.]+), ([\d.]+)\)", got["RTX 4050 用 118 token prompt、16 token 輸出：閒置 ≥3 秒多出的時間（95% CI）"])
    C("V-D", f"The network added up to about {round(net, 2):.2f} s after idle periods, so a small penalty could be hidden there, but the "
             f"RTX 4050 with the same short workload still showed {short.group(1)} s (95% CI {short.group(2)}–{short.group(3)} s)",
      "recompute.py（out/netidle_*.csv、probe_short3b_1011.csv）")
    # 表 II
    early = [rc.probe_extra("probe_wingpu"), rc.probe_extra("probe_wincpu")]

    def row(label, gs, count=None):
        pens = [p for g in gs for p in g["pen"]]
        shs = [s for g in gs for s in g["share"]]
        lo, hi = min(pens), max(pens)
        ptxt = (f"{'≈0' if abs(lo) < 0.01 else neg(f'{lo:+.2f}')} to {neg(f'{hi:+.2f}')}" if len(pens) > 1 and round(lo, 2) != round(hi, 2)
                else neg(f"{hi:+.2f}"))
        slo, shi = round(100 * min(shs)), round(100 * max(shs))
        stxt = (f"{slo}–{shi}%" if slo >= 0 else neg(f"{slo}% to {shi:+d}%")) if slo != shi else f"{shi}%"
        on = sorted({o for g in gs if g["has"] for o in g["onset"]})
        withp = [g for g in gs if g["has"]]
        if len(withp) == 1 and len(gs) > 2 and max(withp[0]["pen"]) <= 0.03:
            dag = "§"                                      # 只有一台，而且代價 ≤ 0.03 秒（表下的註腳）
        else:
            dag = "†" if any(not g["has"] for g in gs) and on else ""
        otxt = (f"{on[0]:g} s" if len(on) == 1 else f"{on[0]:g}–{on[-1]:g} s") + dag if on else "none"
        return f"{label} ; {count or len(gs)} ; {ptxt} ; {stxt} ; {otxt}"
    for label, sel in (("RTX 20 desktops", "2"), ("RTX 30 desktop", "3"), ("RTX 40 laptops", "4"), ("RTX 50", "5")):
        C("表 II", row(label, [g for g in nv if gen(g) == sel]), "重跑 summarize_all.py ＋ RTX 4050")
    C("表 II", row("Apple (Metal)", [g for g in gpus if gen(g) == "Apple"]), "同上 ＋ probe_m4（M4）")
    cmax = max(p for c in cpus for p in c["pen"])
    con = sorted({o for c in cpus if c["has"] for o in c["onset"]})     # 和顯示卡相同的標準
    cw = [c for c in cpus if c["has"]]
    ctxt = "none" if not con else f"{con[0]:g} s" + ("§" if len(cw) == 1 and max(cw[0]["pen"]) <= 0.03 else "")
    C("表 II", f"CPUs ; {sum(len(c['pen']) for c in cpus)} tests ; ≤ {neg(f'{cmax:+.2f}')} ; ≤ {100 * max(csh):.1f}% ; {ctxt}",
      "重跑 summarize_all.py（CPU）")
    C("表 II", f"Early checks‡ ; 2 ; {neg(f'{early[0][1]:+.2f}')}, {neg(f'{early[1][1]:+.2f}')} ; — ; "
               f"{'none' if all(lo <= 0 for _, _, lo, _ in early) else '?'}", "out/probe_wingpu、probe_wincpu.csv（95% CI 都包含 0）")
    C("表 II", "Idle penalty after 3 s or more, by device group", "summarize_all.py（代價＝閒置 ≥ 3 秒 − 一個接一個）", "紀錄")
    C("表 II", f"‡MX250 GPU and i7 CPU, measured over Wi-Fi with {t16[0]} output tokens", "out/probe_wingpu、probe_wincpu.csv", "紀錄")
    one = [g for g in gpus + cpus if g["has"] and max(g["pen"]) <= 0.03]
    which = "、".join("%s %.3f 秒" % (g["name"], max(g["pen"])) for g in one)
    C("表 II", "§Only one device in the group, by 0.03 s or less" if one and all(max(g["pen"]) <= 0.03 for g in one) else "（§ 那幾台的代價超過 0.03 秒）",
      f"同上（{which}）")


def probe_onset(probe):
    """閒置測試：第一個「CI 下限 > 0 而且平均 ≥ 20 毫秒」的閒置秒數（summarize_all.py 的定義；Welch 95% CI）"""
    r = [x for x in rc.probe(probe) if x["cond"] == "A"]
    s0 = np.array([float(x["service"]) for x in r if float(x["gap"]) == 0])
    out = []
    for g in sorted({float(x["gap"]) for x in r if float(x["gap"]) > 0}):
        s = np.array([float(x["service"]) for x in r if float(x["gap"]) == g])
        v1, v2 = s.var(ddof=1) / len(s), s0.var(ddof=1) / len(s0)
        se = np.sqrt(v1 + v2)
        t = stats.t.ppf(0.975, se ** 4 / (v1 ** 2 / (len(s) - 1) + v2 ** 2 / (len(s0) - 1)))
        d = s.mean() - s0.mean()
        if d - t * se > 0 and d >= 0.02:
            out.append(g)
            break
    return out


# ------------------------------------------------------------------ VI 保溫的取捨
KW = {1: {0: "power_default", 3: "kw_tau3", 6: "kw_tau6", 12: "kw_tau12", np.inf: "power_locked"},
      2: {t: f"kw5_tau{'inf' if t == np.inf else t}" for t in (0, 3, 6, 12, np.inf)},
      3: {t: f"kw5m_tau{'inf' if t == np.inf else t}" for t in (0, 3, 6, 12, np.inf)}}


def kw_point(plan_name):
    """一個保溫點：照 analyze_power.py 寫好的定義從原始紀錄重算（照做完的順序重建、丟前 1/3；功耗是第 60 個做完到最後一個做完之間
    nvidia-smi 讀數的平均；每個請求的能量 = 平均功耗 × 區間長度 ÷ 區間裡的請求數）"""
    plan = json.load(open(os.path.join(EXP, f"plan_{plan_name}.json"), encoding="utf-8"))
    run = plan["runs"][0]
    rows = sorted((r for r in csv.DictReader(open(os.path.join(OUT, f"{run['tag']}.csv"), encoding="utf-8")) if r["status"] == "ok"),
                  key=lambda r: float(r["t_send"]))
    a = np.array([float(r["t_send"]) for r in rows])
    c = np.array([float(r["t_done"]) for r in rows])
    prev, cp = np.empty(len(c)), -np.inf
    for j in np.argsort(c, kind="stable"):
        prev[j], cp = cp, c[j]
    start = np.maximum(a, prev)
    n, k = len(rows), run["n"] // 3
    log = open(os.path.join(OUT, f"{run['tag']}_runlog.txt"), encoding="utf-8").read()
    t0 = datetime.fromisoformat(re.search(r"實驗開始時間（UTC）：(\S+)", log).group(1)).timestamp()
    T0, T1 = t0 + c[k - 1], t0 + c[-1]
    ts, pw = [], []
    for p in csv.reader(open(os.path.join(OUT, f"{run['tag']}_gpu.csv"), encoding="utf-8")):
        try:
            t_, w_ = datetime.strptime(p[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(p[5])
            float(p[2])
        except (ValueError, IndexError):
            continue
        ts.append(t_)
        pw.append(w_)
    ts, pw = np.array(ts), np.array(pw)
    power = float(pw[(ts >= T0) & (ts <= T1)].mean())
    S = (c - start)[k:]
    return {"wait": float((start - a)[k:].mean()), "resp": float((c - a)[k:].mean()), "power": power,
            "energy": power * (T1 - T0) / (n - k), "plan": run, "n": run["n"], "rate": run["rate"],
            "rho_hat": (n - k - 1) / (a[-1] - a[k]) * S.mean()}


def cal_deltas(fn, part):
    """校正：保溫（K）放開後到達的代價（照放開了多久分四段）＝ 那段的平均服務時間 − 排隊中的服務時間；
    預設（D）閒置後的代價（照閒置多久分四段）＝ 那段的平均 − 閒置 0 秒的平均（和 run_keepwarm5.py 的參數相同）"""
    rows = [r for r in csv.DictReader(open(os.path.join(OUT, fn), encoding="utf-8")) if part is None or r.get("part") == part]
    edges = [1.0, 2.0, 5.0]
    if part in (None, "K"):
        s0 = np.mean([float(r["service"]) for r in rows if r["state"] == "queued"])
        bins = [[] for _ in range(4)]
        for r in rows:
            if r["state"] == "cold":
                bins[int(np.digitize(float(r["u_after_unlock"]), edges))].append(float(r["service"]))
    else:
        s0 = np.mean([float(r["service"]) for r in rows if float(r["gap"]) == 0])
        bins = [[] for _ in range(4)]
        for r in rows:
            if float(r["gap"]) > 0:
                bins[int(np.digitize(float(r["idle_before"]), edges))].append(float(r["service"]))
    return [np.mean(b) - s0 if b else None for b in bins]


def sec_keepwarm():
    P = {s: {t: kw_point(name) for t, name in m.items()} for s, m in KW.items()}
    taus = sorted(P[1], key=lambda t: (np.isinf(t), t))
    C("VI", "We measured five values of τ (" + ", ".join(f"{t:g}" for t in taus[:3]) + f" and {taus[3]:g} s, and ∞) on the RTX 4050 at "
            f"ρ ≈ {np.mean([p['rho_hat'] for s in P.values() for p in s.values()]):.2f}, with "
            f"{P[1][0]['n'] if all(p['n'] == P[1][0]['n'] for s in P.values() for p in s.values()) else '?'} requests per value",
      f"plan_power_*、plan_kw*_tau*.json（每點的 n、τ）；ρ = 量測區間實際的到達率 × 平均服務時間（15 點平均；"
      f"{min(p['rho_hat'] for s in P.values() for p in s.values()):.3f}–{max(p['rho_hat'] for s in P.values() for p in s.values()):.3f}）")
    tol = re.search(r"abs\(res\[\"power_W\"\] / run\[\"power_pred_W\"\] - 1\) <= ([\d.]+)", open(os.path.join(EXP, "analyze_power.py"), encoding="utf-8").read())
    C("VI", f"and the mean power within ±{100 * float(tol.group(1)):.0f}% of the predicted value", "analyze_power.py（功耗的判定）", "紀錄")
    hits, n_pred, power_hits, miss = 0, 0, 0, {1: [], 2: [], 3: []}
    for s, pts in P.items():
        for t, p in pts.items():
            r = p["plan"]
            for key, lo, hi, pred in (("wait", "lo", "hi", "pred"), ("resp", "resp_lo", "resp_hi", "resp_pred")):
                n_pred += 1
                if r[lo] <= p[key] <= r[hi]:
                    hits += 1
                else:
                    miss[s].append(p[key] - r[pred])
            n_pred += 1
            ok = abs(p["power"] / r["power_pred_W"] - 1) <= 0.10
            hits += ok
            power_hits += ok
    later = miss[2] + miss[3]
    C("VI", f"Of the {n_pred} predictions, {hits} held, including all {power_hits} for power" if power_hits == 15 else "（功耗沒有全中）",
      "15 點 × 等待、回應時間、功耗（事先寫在 plan_*.json 的預測和範圍）")
    C("VI", f"in the first session the measured values were faster than predicted, by at most {-min(miss[1]):.3f} s, and in the later two "
            f"they were slower, by at most {max(later):.3f} s" if max(miss[1]) < 0 and min(later) > 0 else "（沒中的方向不是第一次都快、後兩次都慢）",
      "同上（沒中的那幾項：實測 − 預測）")
    slopes, energy, gaps, adj = [], [], {}, {}
    for s, pts in P.items():
        d, z = pts[0], pts[np.inf]
        slopes.append((z["power"] - d["power"]) / (d["resp"] - z["resp"]) * 0.1)
        energy.append(z["energy"] / d["energy"] - 1)
        line = lambda p, t0=d["resp"]: t0 + (z["resp"] - t0) * (p - d["power"]) / (z["power"] - d["power"])
        gaps[s] = [pts[t]["resp"] - line(pts[t]["power"]) for t in (3, 6, 12)]
        adj[s] = [pts[t]["resp"] - line(pts[t]["power"], d["plan"]["resp_pred"]) for t in (3, 6, 12)]
    sys.path.insert(0, EXP)
    import keepwarm_model as km
    pd_, pl = km.params("probe_base_1011"), km.params("probe_locked_1011")
    p_warm = float(pl["profile"][2:].mean())
    sw = []
    for rho in (0.1, 0.2, 0.3, 0.5):
        lam = rho / pd_["b"]
        a_, z_ = km.closed_form(lam, 0.0, pd_, p_warm), km.closed_form(lam, np.inf, pl, p_warm, p_busy=pl["p_busy"])
        sw.append((z_["P"] - a_["P"]) / (10 * (a_["T"] - z_["T"])))
    C("VI", f"Each 0.1 s cut from the mean response time cost about {int(min(slopes))}–{int(max(slopes)) + 1} W of GPU power: "
            + ", ".join(f"{x:.1f}" for x in slopes[:-1]) + f" and {slopes[-1]:.1f} W in the three sessions, and {min(sw):.1f}–{max(sw):.1f} W in "
            f"the model for ρ = 0.1–0.5. Keeping the clocks locked all the time used {100 * min(energy):.0f}–{100 * max(energy):.0f}% more GPU energy "
            f"per request than the default", "三次量測的兩端（重算）；keepwarm_model.py 的 closed_form（ρ = 0.1、0.2、0.3、0.5）")
    tol2 = re.search(r"^SHAPE_TOL = (-?[\d.]+)", open(os.path.join(EXP, "analyze_keepwarm5.py"), encoding="utf-8").read(), re.M).group(1)
    C("VI", f"The criterion was that no interior point lies more than {-float(tol2):.2f} s below the line between the endpoints",
      "analyze_keepwarm5.py（SHAPE_TOL，10/21 事前登記）", "紀錄")
    s1_ok = all(g > 0 for g in gaps[1])
    C("VI", (f"It held in the first session, where the interior points lay {min(gaps[1]):.2f}–{max(gaps[1]):.2f} s above the line. "
             if s1_ok else "（第一次有點不在線上方）") +
      f"It failed in the second and third, where the three points lay {-max(gaps[2]):.3f}–{-min(gaps[2]):.3f} s and "
      f"{-max(gaps[3]):.3f}–{-min(gaps[3]):.3f} s below it",
      f"三次量測的中間三點 − 兩端實測連線（第一次 {'、'.join(f'{g:+.3f}' for g in gaps[1])} 秒）")
    C("VI", f"the driver keeps the clock up for about {V['drop']} s after the server becomes idle", "同 V-A（g = 8 秒的時脈紀錄）")
    k1 = [x for x in cal_deltas("kw_cal.csv", None) if x is not None]
    k2 = [x for x in cal_deltas("kw5_cal.csv", "K") if x is not None]
    k3 = [x for x in cal_deltas("kw5m_cal.csv", "K") if x is not None]
    d3 = [x for x in cal_deltas("kw5m_cal.csv", "D")[2:] if x is not None]
    k2m = np.mean(k2[1:])                                  # 第二次：放開 1 秒以後的三段（差不多），1 秒內那段另外寫
    C("VI", f"This penalty was {min(k1):.2f}–{max(k1):.2f} s in the first session, larger than the default one, and about {k2m:.2f} s in the "
            f"second ({k2[0]:.2f} s within 1 s of release), close to it; in the third it was {min(k3):.2f}–{max(k3):.2f} s, against "
            f"{min(d3):.2f}–{max(d3):.2f} s for the default" if max(k2[1:]) - min(k2[1:]) < 0.02 else "（第二次放開 1 秒以後的三段差太多，不能寫「約」）",
      f"out/kw_cal、kw5_cal、kw5m_cal.csv（第二次四段：{'、'.join(f'{x:.3f}' for x in k2)}；預設只看閒置 2 秒以上的兩段）")
    slow = [P[s][0]["resp"] - P[s][0]["plan"]["resp_pred"] for s in (2, 3)]
    adj_ok = all(g >= -0.005 for s in (2, 3) for g in adj[s])
    C("VI", f"the default point itself was slower than predicted, by {slow[0]:.3f} s and {slow[1]:.3f} s, and had it matched its prediction, "
            "the interior points would have lain on or above the line" if adj_ok else "（換成預測值後還有點在線下方）",
      f"同上（換成預測值後：第二次 {'、'.join(f'{g:+.3f}' for g in adj[2])}，第三次 {'、'.join(f'{g:+.3f}' for g in adj[3])} 秒）")
    worst = min((g, s, i) for s in (2, 3) for i, g in enumerate(gaps[s]))
    s_w, i_w = worst[1], worst[2]
    rel = -worst[0] / P[s_w][(3, 6, 12)[i_w]]["resp"]
    C("VI", f"Partial keep-warm thus gained little over switching: at most {-worst[0]:.2f} s ({100 * rel:.1f}%), and about "
            f"{-min(gaps[3]):.2f} s in the third session", "同上（線下方最多的那一點；% 是佔那一點實測的回應時間）")
    lw = [(P[s][np.inf]["plan"]["pred"], P[s][np.inf]["wait"]) for s in (1, 2, 3)]
    C("VI", "The predicted and measured mean waiting times were " + ", ".join(f"{p:.3f} and {m:.3f} s" for p, m in lw[:2]) +
            f", and {lw[2][0]:.3f} and {lw[2][1]:.3f} s in the three sessions", "plan_power_locked、plan_kw5_tauinf、plan_kw5m_tauinf.json ＋ 重算")
    # 模型：鎖住再放開的封閉解 vs 10 萬個請求的模擬（15 組設定）；中間點離兩端連線最多多少（ρ = 0.2）
    pk = km.lockunlock_params()
    b_k = float(pk["S0"].mean())
    dt, dp, n_set = 0.0, 0.0, 0
    for rho in (0.1, 0.2, 0.5):
        for tau in (0.0, 3.0, 6.0, 12.0, np.inf):
            cf, sm = km.closed_form_lockunlock(rho / b_k, tau, pk), km.simulate_lockunlock(rho / b_k, tau, pk)
            dt, dp, n_set = max(dt, abs(cf["T"] / sm["T"] - 1)), max(dp, abs(cf["P"] / sm["P"] - 1)), n_set + 1
    C("VI", f"Against simulations of 100,000 requests, the closed form was within {100 * dt:.2f}% for the response time and {100 * dp:.2f}% "
            f"for the power in {n_set} settings", "keepwarm_model.py 的 closed_form_lockunlock、simulate_lockunlock（參數從 out/kw_cal.csv）")
    lam = 0.2 / b_k
    a_, z_ = km.closed_form_lockunlock(lam, 0.0, pk), km.closed_form_lockunlock(lam, np.inf, pk)
    up = max(km.closed_form_lockunlock(lam, t, pk)["T"] - (a_["T"] + (z_["T"] - a_["T"]) * (km.closed_form_lockunlock(lam, t, pk)["P"] - a_["P"])
                                                          / (z_["P"] - a_["P"])) for t in np.arange(0.25, 40.01, 0.25))
    C("VI", f"the model puts the interior points slightly above the line, by at most {up:.3f} s at ρ = 0.2", "同上（τ = 0.25–40 秒，每 0.25 秒）")
    rho_fig = np.mean([p['rho_hat'] for s in P.values() for p in s.values()])
    C("Fig. 3", f"Mean response time against mean GPU power for five hold times τ in three sessions (RTX 4050, ρ ≈ {rho_fig:.2f})", "同上", "紀錄")
    C("Fig. 3", "hollow markers with bars are the predictions written down in advance, with their 95% simulation ranges",
      "run_keepwarm5.py（2.5、97.5 百分位）", "紀錄" if code_has(os.path.join(EXP, "run_keepwarm5.py"), r"np\.percentile\(x, \[2\.5, 97\.5\]\)") else "⚠️")
    V.update(energy=f"{100 * min(energy):.0f}–{100 * max(energy):.0f}%")


# ------------------------------------------------------------------ 摘要、引言、結論（和前面各節同一次重算的數字）
def sec_front():
    C("摘要", f"On an RTX 4050 laptop, a request arriving after 3 s or more of idleness took {V['pen']} s ({V['share']}) longer than one "
              "served back to back", "同 V-A")
    words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    C("摘要", f"{words[V['nv_has']].capitalize()} of the {V['nv']} NVIDIA RTX GPUs we tested showed a similar penalty of {V['sh_lo']} to {V['sh_hi']} "
              "of the service time", "同 V-D")
    C("摘要", f"In {V['n_pre']} blind predictions, the median error was {V['median']}, the model was closer to the measurement than "
              f"the P-K formula in {V['closer']}, and {V['hits']} measurements fell inside the narrow simulation ranges", "同 V-C")
    C("摘要", f"but costs {V['energy']} more GPU energy per request", "同 VI")
    C("I", f"requests after 3 s or more of idleness took {V['pen']} s longer on average than requests served back to back", "同 V-A")
    C("I", f"This extra delay appeared at about {V['drop']} s of idleness", "同 V-A")
    C("I", f"but locking cost {V['energy']} more GPU energy per request", "同 VI")
    C("I", f"Of the {V['nv']} NVIDIA RTX GPUs we tested, {words[V['nv_vol']]} of them on volunteers' computers, {words[V['nv_has']]} showed a "
           f"similar penalty of {V['sh_lo']} to {V['sh_hi']} of the service time", "同 V-D")
    C("VII", f"{V['pen']} s after 3 s or more on our RTX 4050, and {V['sh_lo'].rstrip('%')}–{V['sh_hi']} of the service time on "
             f"{V['nv_has']} of the {V['nv']} NVIDIA RTX GPUs we tested", "同 V-A、V-D")
    C("VII", f"predicted the mean wait of {V['n_pre']} runs blind, with a median error of {V['median']}, closer than the P-K formula in "
             f"{V['closer']} of them", "同 V-C")
    C("VII", f"Locking the clock nearly removes the penalty at {V['energy']} more GPU energy per request", "同 VI")
    C("VII", f"overstated the service time under load by {V['iso']} on the RTX 4050", "同 V-A（工作負載 A、C）")
    C("VII", f"calibrating with queued requests cut the error from {V['m2'][0]} to {V['m2'][1]} on {V['m2n']} volunteer runs", "同 V-E")


# ------------------------------------------------------------------ 志願者總表（重跑 summarize_all.py，輸出到暫存資料夾）
def rerun_summary():
    """把已揭曉的機器複製到暫存資料夾，重跑 summarize_all.py（不動正式的 朋友結果/彙整/）"""
    import shutil
    import subprocess
    import tempfile
    tmp = tempfile.mkdtemp(prefix="稽核_彙整_")
    for d in glob.glob(os.path.join(FRIENDS, "*")):
        if os.path.isdir(d) and os.path.basename(d) != "彙整":
            shutil.copytree(d, os.path.join(tmp, os.path.basename(d)))
    r = subprocess.run([sys.executable, "-I", os.path.join(ANALYSIS, "summarize_all.py"), tmp], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if r.returncode != 0:
        raise SystemExit("summarize_all.py 失敗：" + r.stderr[-2000:])
    return os.path.join(tmp, "彙整", "all_results.csv")


# ------------------------------------------------------------------ 執行、比對、涵蓋率
def paper_text():
    md = open(nl.DRAFT, encoding="utf-8").read()
    return "\n".join(t for _, t in nl.english_items(md))


def coverage(text, found_spans):
    masked = nl.NAME.sub(lambda m: "#" * len(m.group(0)), text)
    missing = []
    for m in nl.NUM.finditer(masked):
        if not any(a <= m.start() and m.end() <= b for a, b in found_spans):
            missing.append((m.group(0), text[max(0, m.start() - 50):m.end() + 30].replace("\n", " ")))
    return missing


SUMMARY = None
FAKE = None


def main():
    global SUMMARY, FAKE
    SUMMARY = rerun_summary()
    FAKE = fake_server()
    w = waiting()
    sec_testbed()
    sec_generator()
    sec_probes()
    sec_statistics(w)
    sec_idle()
    sec_waiting(w)
    sec_prereg()
    sec_slowdown()
    sec_hardware()
    sec_keepwarm()
    sec_front()
    text = paper_text()
    spans, lines, bad = [], [], 0
    for sec, sent, src, kind in CLAIMS:
        i = text.find(sent)
        ok = i >= 0 and kind != "⚠️"
        if ok:
            spans.append((i, i + len(sent)))
        else:
            bad += 1
        lines.append(f"| {sec} | {'✅' if ok else '⚠️'} {kind} | {sent} | {src} |")
    missing = coverage(text, spans)
    n_all = sum(1 for _ in nl.NUM.finditer(nl.NAME.sub(lambda m: "#" * len(m.group(0)), text)))
    with open(RESULT, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# 全文稽核結果（{datetime.now():%Y-%m-%d %H:%M}；稽核/全文稽核.py 產生）\n\n")
        f.write(f"> 檢查 {len(CLAIMS)} 條：✅ {len(CLAIMS) - bad}、⚠️ {bad}｜論文裡的數字 {n_all} 個，被 ✅ 的句子涵蓋 {n_all - len(missing)} 個\n\n")
        f.write("| 節 | 結果 | 重算後的句子（要和論文一字不差） | 來源 |\n|---|---|---|---|\n")
        f.write("\n".join(lines) + "\n\n## 還沒有被涵蓋的數字\n\n")
        f.write("\n".join(f"- {n}：…{c}…" for n, c in missing) + "\n")
    print(f"檢查 {len(CLAIMS)} 條：✅ {len(CLAIMS) - bad}、⚠️ {bad}｜數字 {n_all} 個，涵蓋 {n_all - len(missing)} 個 → {RESULT}")
    for line in lines:
        if "⚠️" in line:
            print("  ", line[:400])


if __name__ == "__main__":
    main()
