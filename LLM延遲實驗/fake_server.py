"""假伺服器（Day 12）：服務時間分布已知的 OpenAI 相容端點

用途：驗證 load_gen.py 量得準不準。
因為服務時間的分布是我們自己決定的，E[S] 和 E[S²] 算得出來，
可以直接代進 Pollaczek–Khinchine 公式得到理論等待時間，再和實測比較。

行為
- **一次只處理一個請求**（單一服務台），其餘先進先出排隊 → M/G/1
- 輪到某個請求時，**立刻送出第一個 token**
  → 客戶端量到的 TTFT ≈ 排隊等待時間 W_q（方便和理論比較）
- 服務時間結束後送出 [DONE]
- 每個請求的到達、開始服務、結束時間都寫進 server_log.csv

只用 Python 標準函式庫。

範例：
  python fake_server.py --dist uniform --mean 2 --spread 1     # S ~ U(1, 3)
  python fake_server.py --dist exp --mean 2                    # S ~ Exp(平均 2 秒)
"""
import argparse
import csv
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None
RNG = random.Random(0)
START = time.perf_counter()
LOG, LOG_LOCK = [], threading.Lock()

# ---- 先進先出的單一服務台（用號碼牌，確保順序就是到達順序）----
TICKET_LOCK = threading.Lock()
NEXT_TICKET = 0          # 下一個要發出去的號碼
NOW_SERVING = 0          # 目前叫到的號碼
CV = threading.Condition(TICKET_LOCK)


def take_ticket():
    global NEXT_TICKET
    with TICKET_LOCK:
        t = NEXT_TICKET
        NEXT_TICKET += 1
        return t


def wait_turn(ticket):
    with CV:
        while NOW_SERVING != ticket:
            CV.wait()


def finish_turn():
    global NOW_SERVING
    with CV:
        NOW_SERVING += 1
        CV.notify_all()


def service_time():
    if ARGS.dist == "uniform":
        return RNG.uniform(ARGS.mean - ARGS.spread, ARGS.mean + ARGS.spread)
    if ARGS.dist == "exp":
        return RNG.expovariate(1.0 / ARGS.mean)
    if ARGS.dist == "const":
        return ARGS.mean
    raise ValueError(ARGS.dist)


def moments():
    """回傳 E[S]、E[S^2]（理論值，用來算 P-K 公式）"""
    m = ARGS.mean
    if ARGS.dist == "uniform":
        var = (2 * ARGS.spread) ** 2 / 12
    elif ARGS.dist == "exp":
        var = m ** 2
    else:
        var = 0.0
    return m, var + m ** 2


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True      # 關掉 Nagle：小封包不要等，否則每個回應會多出約 0.2 秒

    def log_message(self, *a):
        pass                                    # 不要印 HTTP access log

    def do_GET(self):
        """/__shutdown__：寫出 log 並關閉（run_validation.py 用，確保 log 一定寫得出來）
        /health：回 200（2026-10-21 加：試流程用，和 llama-server 一樣用它確認伺服器起來了）"""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
        elif self.path == "/__shutdown__":
            write_log()
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)                      # 讀掉 body，內容不重要

        t_arrive = time.perf_counter() - START
        ticket = take_ticket()

        wait_turn(ticket)                       # ← 排隊，等前面的人做完
        t_start = time.perf_counter() - START
        s = service_time()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            # 開始服務就送第一個 token → 客戶端的 TTFT ≈ 等待時間
            self._chunk("first ")
            time.sleep(s)
            self._chunk("done")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception:
            pass                                # 客戶端中途斷線就算了
        finally:
            t_end = time.perf_counter() - START
            finish_turn()                       # ← 叫下一位
            with LOG_LOCK:
                LOG.append({"ticket": ticket,
                            "t_arrive": round(t_arrive, 4),
                            "t_start": round(t_start, 4),
                            "t_end": round(t_end, 4),
                            "service": round(s, 4),
                            "wait": round(t_start - t_arrive, 4)})

    def _chunk(self, text):
        payload = json.dumps({"choices": [{"delta": {"content": text}}]})
        self.wfile.write(f"data: {payload}\n\n".encode())
        self.wfile.flush()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256            # 預設只有 5，佇列一長就會被拒絕連線


def write_log():
    import os
    os.makedirs(os.path.dirname(os.path.abspath(ARGS.log)), exist_ok=True)
    with LOG_LOCK, open(ARGS.log, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ticket", "t_arrive", "t_start", "t_end", "service", "wait"])
        w.writeheader()
        w.writerows(sorted(LOG, key=lambda r: r["ticket"]))
    print(f"已服務 {len(LOG)} 個請求，log → {ARGS.log}")


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--dist", choices=["uniform", "exp", "const"], default="uniform")
    p.add_argument("--mean", type=float, default=2.0, help="平均服務時間（秒）")
    p.add_argument("--spread", type=float, default=1.0, help="uniform 用：mean ± spread")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--log", default="out/server_log.csv")
    ARGS = p.parse_args()

    es, es2 = moments()
    print(f"服務時間分布：{ARGS.dist}，E[S]={es:.3f} 秒，E[S^2]={es2:.3f}")
    print("各負載對應的到達率與 P-K 理論等待時間：")
    print(f"  {'ρ':>5}{'λ (每秒)':>12}{'理論 W_q (秒)':>16}")
    for rho in (0.3, 0.5, 0.7, 0.8, 0.85):
        lam = rho / es
        wq = lam * es2 / (2 * (1 - rho))
        print(f"  {rho:>5.2f}{lam:>12.4f}{wq:>16.2f}")
    print(f"\n監聽 http://127.0.0.1:{ARGS.port}/v1/chat/completions   （Ctrl+C 結束並存 log）")

    srv = Server(("127.0.0.1", ARGS.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        write_log()


if __name__ == "__main__":
    main()
