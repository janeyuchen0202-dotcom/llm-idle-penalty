"""一鍵驗證（Day 16–17）

自動做完：開假伺服器 → 跑兩個負載點 → 和 P-K 理論比較 → 關掉伺服器。
不需要開兩個終端機。

用法：
  python run_validation.py                  # 預設：ρ=0.5（300 筆）與 ρ=0.8（400 筆），約 9 分鐘
  python run_validation.py --quick          # 快速版：各 60 筆，約 1.5 分鐘（樣本少，只用來確認流程）
"""
import argparse
import os
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def wait_port(port, timeout=10):
    """等伺服器真的開始監聽"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.2)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mean", type=float, default=0.5)
    p.add_argument("--spread", type=float, default=0.25)
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--tag", default="my", help="輸出檔名前綴")
    a = p.parse_args()

    es = a.mean
    es2 = (2 * a.spread) ** 2 / 12 + a.mean ** 2
    loads = [(0.5, 60), (0.8, 60)] if a.quick else [(0.5, 300), (0.8, 400)]
    if a.quick:
        print("⚠️ --quick 只用來確認流程跑得動；樣本太少，判定出現「不吻合」是正常的。\n")

    print(f"E[S]={es:.4f}  E[S^2]={es2:.4f}")
    total = sum(n / (rho / es) for rho, n in loads)
    print(f"預計耗時約 {total / 60:.1f} 分鐘\n")

    # ---- 1. 開伺服器（子行程，之後由程式自己關）----
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    server = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "fake_server.py"),
         "--dist", "uniform", "--mean", str(a.mean), "--spread", str(a.spread),
         "--port", str(a.port), "--log", f"out/{a.tag}_server_log.csv"],
        cwd=HERE, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if not wait_port(a.port):
        server.kill()
        sys.exit(f"伺服器沒有在 port {a.port} 上啟動：{server.stderr.read().decode('utf-8', 'replace')[:500]}")
    print(f"假伺服器已啟動（port {a.port}）")

    try:
        # ---- 2. 跑各負載點 ----
        runs = []
        for rho, n in loads:
            lam = rho / es
            out = f"out/{a.tag}_rho{int(rho * 100)}.csv"
            print(f"\n=== ρ={rho}（λ={lam:.2f}/s，{n} 筆，約 {n / lam / 60:.1f} 分鐘）===")
            subprocess.run(
                [sys.executable, os.path.join(HERE, "load_gen.py"),
                 "--url", f"http://127.0.0.1:{a.port}/v1/chat/completions",
                 "--rate", str(lam), "--n", str(n), "--out", out],
                cwd=HERE, env=env, check=True)
            runs.append(f"{out}:{lam}")
    finally:
        # ---- 3. 請伺服器寫出 log 並關閉 ----
        try:
            import urllib.request
            urllib.request.urlopen(f"http://127.0.0.1:{a.port}/__shutdown__", timeout=5).read()
        except Exception:
            pass
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        print("\n假伺服器已關閉")

    warmup = "20" if not a.quick else "5"

    # ---- 4. 比較（用設定的分布）----
    print("\n【以設定的服務時間分布計算理論值】")
    subprocess.run([sys.executable, os.path.join(HERE, "compare_pk.py"),
                    "--es", f"{es}", "--es2", f"{es2}", "--warmup", warmup, *runs], cwd=HERE, env=env)

    # ---- 5. 比較（用伺服器實際占用時間）----
    # 實際占用 = 開始服務到結束，含 HTTP 傳輸等開銷。若它明顯大於設定值，
    # 代表真正的服務時間比設定的長，理論值要用這個重算才公平。
    log_path = os.path.join(HERE, "out", f"{a.tag}_server_log.csv")
    if os.path.exists(log_path):
        import csv
        occ = [float(r["t_end"]) - float(r["t_start"]) for r in csv.DictReader(open(log_path, encoding="utf-8"))]
        if occ:
            es_m = sum(occ) / len(occ)
            es2_m = sum(x * x for x in occ) / len(occ)
            print(f"\n【以伺服器實際占用時間計算理論值】設定 E[S]={es:.3f}，實測占用 E[S]={es_m:.3f}"
                  f"（多出 {es_m - es:+.3f} 秒／筆）")
            runs_m = [f"{r.rsplit(':', 1)[0]}:{r.rsplit(':', 1)[1]}" for r in runs]
            subprocess.run([sys.executable, os.path.join(HERE, "compare_pk.py"),
                            "--es", f"{es_m}", "--es2", f"{es2_m}", "--warmup", warmup, *runs_m],
                           cwd=HERE, env=env)


if __name__ == "__main__":
    main()
