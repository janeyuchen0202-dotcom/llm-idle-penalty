"""期中診斷 2：伺服器閒置後的第一個請求，服務時間是不是比較長？

  閒置後開始：到達時伺服器空著（W = 0），服務時間 = 完成 − 送出
  排隊後開始：到達時要等（W > 0），服務時間 = 完成 − 前一個完成
用全部資料（含暖機段），因為這裡比的是服務時間，不是穩態等待。
假伺服器（val_*）當對照組：它只是 sleep，沒有 CPU 喚醒的問題。
"""
import csv
import os

import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
NAMES = ["val_rho50", "val_rho80",
         "main2_rho30", "main2_rho50", "long70_rho70", "long85_rho85", "mix_A_rho50", "mix_A_rho75",
         "var_C_rho50", "var_C_rho70",
         "rep_rho70", "rep_rho85", "var2_C_rho70", "confirm_C_rho75"]


def load(name):
    rows = sorted([r for r in csv.DictReader(open(os.path.join(LAB, "out", name + ".csv"), encoding="utf-8"))
                   if r["status"] == "ok"], key=lambda r: float(r["t_send"]))
    a = np.array([float(r["t_send"]) for r in rows])
    c = np.array([float(r["t_done"]) for r in rows])
    tok = np.array([int(r["out_tokens"] or 0) for r in rows])
    prev = np.empty(len(c)); _cp = -1e9    # 10/22 改：照做完的順序重建（一次只處理一個請求，做完的順序就是處理的順序；原本照送出順序）
    for _j in np.argsort(c, kind="stable"):
        prev[_j] = _cp; _cp = c[_j]
    start = np.maximum(a, prev)
    return a, c, start - a, c - start, a - prev, tok


print(f"{'組別':<16}{'閒置後 n':>9}{'平均S':>8}{'排隊後 n':>9}{'平均S':>8}{'差':>8}"
      f"{'  閒置<2s':>9}{'2–10s':>8}{'>10s':>8}   (同輸出長度比較)")
for nm in NAMES:
    if not os.path.exists(os.path.join(LAB, "out", nm + ".csv")):
        continue
    a, c, W, S, gap, tok = load(nm)
    idle = W == 0
    idle[0] = False                               # 第一筆沒有前一個，跳過
    q = W > 0
    # C 組輸出長度有兩種，分開比較才公平：只看 224 tokens 的
    if len(set(tok)) > 1:
        sel = tok == tok.max()
        tag = f"(只看 {tok.max()} tokens)"
    else:
        sel = np.ones_like(idle)
        tag = ""
    i_s, q_s = S[idle & sel], S[q & sel]
    g = gap[idle & sel]
    bins = [i_s[g < 2], i_s[(g >= 2) & (g < 10)], i_s[g >= 10]]
    f = lambda x: f"{x.mean():.3f}" if len(x) else "  –  "
    print(f"{nm:<16}{len(i_s):>9}{f(i_s):>8}{len(q_s):>9}{f(q_s):>8}"
          f"{(i_s.mean() - q_s.mean()) if len(i_s) and len(q_s) else float('nan'):>+8.3f}"
          f"{f(bins[0]):>9}{f(bins[1]):>8}{f(bins[2]):>8}   {tag}")
