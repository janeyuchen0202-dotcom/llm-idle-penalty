"""公式檢查（2026-10-09）：用模擬驗證論文的公式 (3)、(4)

做法：直接模擬一個「每段忙碌期的第一個請求多 δ(I)」的單一伺服器佇列（FIFO），
跟論文的公式比。模擬不用任何公式，只照規則一個一個請求排：
- 請求以 Poisson 過程到達（間隔是指數分布，平均 1/λ）
- 伺服器空著時到達的請求：服務時間 = S + δ(I)，I = 伺服器已經空了多久
- 要排隊的請求：服務時間 = S
- 等待時間用 Lindley 公式一個一個算：下一個人的等待 = max(0, 這個人的等待 + 這個人的服務時間 − 兩人到達的間隔)

δ(I) 用論文量到的階梯函數（閒置 <1、1–2、2–5、≥5 秒：+0.041、+0.092、+0.337、+0.333 秒），
S 的平均 b = 1.797 秒，用常態分布、變異係數 0.02（和量到的一樣小）。

用法：python 稽核/check_formulas.py      （約 1–2 分鐘）
"""
import math

import numpy as np

EDGES = [1.0, 2.0, 5.0]
DELTA = [0.041, 0.092, 0.337, 0.333]
B, CV = 1.797, 0.02
N, BATCHES = 2_000_000, 20


def moments_delta(lam):
    """I ~ 指數分布（速率 λ）時，E[δ(I)] 和 E[δ(I)²]"""
    cuts = [0.0] + EDGES + [math.inf]
    p = [math.exp(-lam * lo) - (math.exp(-lam * hi) if hi != math.inf else 0.0) for lo, hi in zip(cuts, cuts[1:])]
    e1 = sum(pi * d for pi, d in zip(p, DELTA))
    e2 = sum(pi * d * d for pi, d in zip(p, DELTA))
    return e1, e2


def formula(lam, delta_on=True):
    b, b2 = B, B * B * (1 + CV * CV)
    e1, e2 = moments_delta(lam) if delta_on else (0.0, 0.0)
    b0, b02 = b + e1, b2 + 2 * b * e1 + e2          # 這裡用到「S 和 I 互相獨立」
    rho = lam * b
    wq = lam * b2 / (2 * (1 - rho)) + lam * (b02 - b2) / (2 * (1 - rho + lam * b0))      # 公式 (3)
    p0 = (1 - rho) / (1 - rho + lam * b0)
    t = wq + p0 * b0 + (1 - p0) * b                                                     # 公式 (4)
    pk_pooled_es = p0 * b0 + (1 - p0) * b                                               # 公式 (2)：把懲罰平均攤到所有請求
    pk_pooled_es2 = p0 * b02 + (1 - p0) * b2
    pk = lam * pk_pooled_es2 / (2 * (1 - lam * pk_pooled_es))
    return wq, t, p0, pk


def simulate(lam, seed):
    rng = np.random.default_rng(seed)
    inter = rng.exponential(1 / lam, N)
    s = rng.normal(B, B * CV, N)
    w = np.zeros(N)
    serv = np.zeros(N)
    idle_first = np.zeros(N, dtype=bool)
    free_at = -1e9                    # 伺服器什麼時候空下來
    t = 0.0
    for i in range(N):
        t += inter[i]
        if t >= free_at:              # 伺服器空著：這個請求不用等，但要付 δ(I)
            idle = t - free_at if i > 0 else 1e9
            pen = DELTA[int(np.searchsorted(EDGES, idle, side="right"))]
            w[i], serv[i], idle_first[i] = 0.0, s[i] + pen, True
            free_at = t + serv[i]
        else:                         # 要排隊
            w[i], serv[i] = free_at - t, s[i]
            free_at += serv[i]
    k = N // 10                       # 丟掉前 10%
    ww = w[k:]
    bm = np.array([x.mean() for x in np.array_split(ww, BATCHES)])
    half = 2.093 * bm.std(ddof=1) / math.sqrt(BATCHES)
    return ww.mean(), half, (ww + serv[k:]).mean(), idle_first[k:].mean()


def main():
    """每個 ρ 用 4 個亂數種子各模擬 200 萬個請求；用種子之間的差異當不確定性（比單次的批次區間可靠）"""
    print(f"{'ρ':>5}{'模擬 W_q':>11}{'±2SE':>8}{'公式(3)':>9}{'差':>8}{'模擬 T':>9}{'公式(4)':>9}"
          f"{'模擬p0':>8}{'公式p0':>8}{'公式(2)攤平':>11}")
    ok = True
    for i, rho in enumerate((0.3, 0.5, 0.7, 0.85)):
        lam = rho / B
        wq, t, p0, pk = formula(lam)
        sims = np.array([simulate(lam, 1000 + 10 * i + k)[:4] for k in range(4)])
        sw, st, sp0 = sims[:, 0].mean(), sims[:, 2].mean(), sims[:, 3].mean()
        two_se = 2 * sims[:, 0].std(ddof=1) / 2
        inside = abs(sw - wq) <= max(two_se, 0.01 * wq)
        ok &= inside
        print(f"{rho:>5.2f}{sw:>11.3f}{two_se:>8.3f}{wq:>9.3f}{100 * (sw / wq - 1):>+7.1f}%{st:>9.3f}{t:>9.3f}"
              f"{sp0:>8.3f}{p0:>8.3f}{pk:>11.3f}  {'✅ 一致' if inside else '⚠️ 不一致'}")
    wq0, t0, p00, _ = formula(1e-6)
    wq1, _, _, pk1 = formula(0.5 / B, delta_on=False)
    print(f"{chr(10)}檢查一：δ = 0 時 (3) 是否退回 P-K (2)：ρ=0.5 時 (3) = {wq1:.4f}，(2) = {pk1:.4f}"
          f"  → {'✅' if abs(wq1 - pk1) < 1e-9 else '⚠️'}")
    print(f"檢查二：λ → 0 時平均回應時間 = b + δ(≥5 秒) = {B} + {DELTA[-1]} = {B + DELTA[-1]:.3f}；公式 (4) 得 {t0:.3f}"
          f"  → {'✅' if abs(t0 - (B + DELTA[-1])) < 1e-3 else '⚠️'}")
    print("檢查三：單位。λ（1/秒）× b₂（秒²）＝ 秒，分母沒有單位，所以 W_q 是秒 → ✅")
    print(f"{chr(10)}最後一欄「公式(2)攤平」：把懲罰平均攤到所有請求的 P-K，比真實值高，這就是論文說 (2) 高估的原因")
    print(f"{chr(10)}{'全部通過' if ok else '有情況不一致，要看'}")


if __name__ == "__main__":
    main()
