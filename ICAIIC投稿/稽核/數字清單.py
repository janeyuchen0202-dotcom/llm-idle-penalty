"""改寫版論文的數字清單（2026-10-24）：列出 重寫/論文_目前版本_20261024.md 正文裡的每一個數字，給稽核擴充用

- 只看排進論文的英文：摘要、關鍵詞、各節段落、圖說、表格、致謝；不看參考文獻、> 開頭的中文說明、TODO 註記
- 不算數字的：節的編號、引用編號、型號和名稱裡的數字（RTX 4050、M1、3B、Qwen2.5、P-K…，見 NAME）、公式編號 (1)–(3)、
  1/(1 − ρ) 這種式子裡的 1
- 輸出：稽核/數字清單_20261024.md（每個數字一列：節、數字、前後文；之後在「稽核」欄填對應的重算項目）

用法：在 ICAIIC投稿 資料夾執行  python 稽核/數字清單.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DRAFT = os.path.join(os.path.dirname(HERE), "重寫", "論文_目前版本_20261024.md")
OUT = os.path.join(HERE, "數字清單_20261024.md")
# 名稱裡的數字（型號、版本、方法名稱），不算要稽核的數字
NAME = re.compile(r"RTX \d{4}( Ti| Laptop)?|RTX [2-5]0(-series)?|GTX \d+|MX250|M[14]( Air| Pro)?\b|i[579]-\d{4,5}[A-Z]*|i7\b|"
                  r"Qwen2\.5-Coder-3B-Instruct|LLaMA-2-7B|A100|\b[37]B\b|GPT-5\.6|Opus 5(\.5)?|Gemini 3\.1|P8\b|P0\b|"
                  r"Q4_K_M|b9549|SHA-256|M/G/1|Windows 10|IEEE \d+|SC24|\(\d\)|Section [IVX]+(-[A-F])?|Fig\. \d|Table [IVX]+|"
                  r"\[\w+\]|1/\(1 − ρ\)|e\^\{−λτ\}|2\.5th|97\.5th|first third|10⁻⁸|"
                  r"W_i = 0|τ = 0|τ → 0|2bE|Session 1")          # 10/24：記號和公式裡的常數（不是數據）
NUM = re.compile(r"[−+]?\d[\d,]*(?:\.\d+)?(?:–\d[\d,]*(?:\.\d+)?)?%?")


def english_items(md):
    """(節, 文字) 的清單"""
    sec, in_refs = "Front matter", False
    for line in md.splitlines():
        s = line.strip()
        if not s or s.startswith(">") or s.startswith("[[TODO") or s == "---" or s.startswith("# "):
            continue
        if s.startswith("## "):
            sec = s[3:]
            in_refs = sec == "References"
            continue
        if in_refs:
            continue
        if s.startswith("### "):
            sec = sec.split(" / ")[0] + " / " + s[4:]
            continue
        if s.startswith("|---"):
            continue
        if re.search(r"\(\d\)\s*$", s) and "=" in s:              # 公式那一行（排版時換成公式）
            continue
        if s.startswith("|"):
            s = " ; ".join(c.strip() for c in s.strip("|").split("|"))
        yield sec, re.sub(r"\[\[TODO[^\]]*\]\]", "", s)


def main():
    md = open(DRAFT, encoding="utf-8").read()
    rows = []
    for sec, text in english_items(md):
        masked = NAME.sub(lambda m: "#" * len(m.group(0)), text)
        for m in NUM.finditer(masked):
            ctx = text[max(0, m.start() - 45):m.end() + 30].replace("|", "/")
            rows.append((sec, m.group(0), ctx))
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("# 改寫版論文的數字清單（10/24；稽核/數字清單.py 產生）\n\n")
        f.write(f"> 共 {len(rows)} 個數字。「稽核」欄之後填：對應的重算項目（✅ 已核對、⚠️ 不一致、— 不用重算，例如設定值）\n\n")
        f.write("| # | 節 | 數字 | 前後文 | 稽核 |\n|---|---|---|---|---|\n")
        for i, (sec, num, ctx) in enumerate(rows, 1):
            f.write(f"| {i} | {sec} | {num} | …{ctx}… | |\n")
    by = {}
    for sec, _, _ in rows:
        top = sec.split(" / ")[0]
        by[top] = by.get(top, 0) + 1
    for k, v in by.items():
        print(f"  {v:4d}  {k}")
    print(f"  {len(rows):4d}  合計 → {OUT}")


if __name__ == "__main__":
    main()
