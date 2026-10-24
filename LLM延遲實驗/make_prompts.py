"""產生指定 token 數的 prompt 檔（Day 18、Day 29 用）

llama-server 有 /tokenize 端點，可以問它「這段文字是幾個 token」。
本程式反覆加長文字，直到 token 數接近目標值，再存成檔案。

用法（llama-server 要先開著）：
  python make_prompts.py                       # 產生 short(100)、medium(800)、long(1500)
  python make_prompts.py --targets 800         # 只產生 medium
  python make_prompts.py --url http://localhost:8080

輸出：prompts/short.txt、prompts/medium.txt、prompts/long.txt
      以及 prompts/token_counts.txt（記錄實際 token 數，Day 18 的完成標準要用）
"""
import argparse
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "prompts")

# 填充用的技術文字；內容不重要，重要的是長度固定、每次一樣
FILLER = (
    "A queueing system consists of arrivals, a queue, and one or more servers. "
    "The arrival process describes when requests enter the system, while the service "
    "process describes how long each request occupies a server. Utilization is the "
    "fraction of time a server is busy, and waiting time grows sharply as utilization "
    "approaches one. Variability in service times makes waiting worse even when the "
    "average service time stays the same. "
)
INSTRUCTION = ("Read the notes below and then write a detailed technical summary of at least "
               "300 words about queueing systems, covering arrivals, service times, "
               "utilization and waiting time.\n\nNotes:\n")


def count_tokens(url, text):
    body = json.dumps({"content": text}).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/tokenize", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return len(json.loads(r.read().decode("utf-8"))["tokens"])


def build(url, target):
    """用二分逼近：一直加 FILLER 直到 token 數 ≥ target，再微調"""
    reps = max(1, target // 60)
    while True:
        text = INSTRUCTION + FILLER * reps
        n = count_tokens(url, text)
        if n >= target or reps > 10000:
            break
        reps += max(1, (target - n) // 60)
    while reps > 1:
        text2 = INSTRUCTION + FILLER * (reps - 1)
        if count_tokens(url, text2) < target * 0.97:
            break
        reps -= 1
    text = INSTRUCTION + FILLER * reps
    return text, count_tokens(url, text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--targets", default="100,800,1500", help="目標 token 數，逗號分隔")
    p.add_argument("--names", default="short,medium,long")
    a = p.parse_args()

    targets = [int(x) for x in a.targets.split(",")]
    names = a.names.split(",")[:len(targets)]
    os.makedirs(OUT_DIR, exist_ok=True)

    lines = []
    for name, target in zip(names, targets):
        text, n = build(a.url, target)
        path = os.path.join(OUT_DIR, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        line = f"{name}.txt  目標 {target} tokens  實際 {n} tokens  ({len(text)} 字元)"
        print(line)
        lines.append(line)

    with open(os.path.join(OUT_DIR, "token_counts.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n已寫入 {OUT_DIR}")


if __name__ == "__main__":
    main()
