#!/bin/bash
# LLM 閒置測試（Mac 版，2026-10-19）
# 用法：打開「終端機」，輸入  bash ~/Downloads/mac_idle_test.sh  按 Enter
#       （檔案不在「下載」的話：輸入 bash 和一個空格，把這個檔案拖進終端機視窗，按 Enter）
# 會做的事：下載 llama.cpp b9549（和論文相同的版本）、免安裝的 Python、模型（約 2.1 GB），
# 全部放在家目錄的 llm_idle_test_mac 資料夾，下載後都核對 SHA-256；跑約 40 分鐘；結果存到桌面。
# 不需要系統管理員密碼，也不會改這台 Mac 的任何設定。
set -u
VER="2026-10-19b-mac"
W="$HOME/llm_idle_test_mac"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20260929/cpython-3.12.14%2B20260929-aarch64-apple-darwin-install_only.tar.gz"
PY_SHA="de6b8f94fa765639b423ea353ab340669704c7186f96ee3cab389dcfde770c3c"
LL_URL="https://github.com/ggml-org/llama.cpp/releases/download/b9549/llama-b9549-bin-macos-arm64.tar.gz"
LL_SHA="08ba1a42ca4356722552dde5ea7678e05e0bd2e983a44d1ebdcbab3df4dc0308"
MD_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
MD_SHA="724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7"

say() { printf '%s\n' "$*"; }
die() { say ""; say "【停止】$*"; say "請把這個視窗截圖，傳給請你幫忙測試的人。"; read -r -p "按 Enter 關閉…… " _ || true; exit 1; }
sha() { shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'; }

say "============================================================"
say "  LLM 閒置測試（Mac 版）"
say "============================================================"
say "這個測試會在這台 Mac 上跑一個小型語言模型，量「閒置幾秒之後，下一個請求會不會變慢」。"
say "全部大約 1 小時：先下載約 2.2 GB 的檔案，再跑約 40 分鐘。不會改這台 Mac 的任何設定。"
say "檔案放在家目錄的 llm_idle_test_mac 資料夾，結果會存到桌面。"
say "收集的只有時間紀錄、CPU 和 GPU 使用率、硬體型號；使用者名稱、電腦名稱和序號都會遮掉。"
say ""
say "跑的時候請："
say "  1. 插著電源，不要闔上螢幕（螢幕會一直亮著，可以把亮度調低）"
say "  2. 關掉其他程式（影片、遊戲、瀏覽器等）"
say "  3. 不要用這台電腦，也不要關掉這個視窗"
say "  4. 如果跳出「終端機想要取用……」的視窗，請按「好」或「允許」"
say ""
read -r -p "準備好了按 Enter 開始…… " _ || true

# 10/19 加：電池太低的話建議先充電（M1 Air 那次是 9% 邊充電邊跑，可能讓它更熱）
pct=$(pmset -g batt 2>/dev/null | grep -o '[0-9]*%' | head -1 | tr -d '%')
if [ -n "$pct" ] && [ "$pct" -lt 80 ] 2>/dev/null; then
  say ""
  say "電池現在 ${pct}%。建議先充到 80% 以上再跑：一邊充電一邊跑會比較熱，可能影響結果。"
  read -r -p "要先充電就關掉這個視窗，晚點再執行；要直接開始請按 Enter…… " _ || true
fi

# 10/19 加：用 powermetrics 記錄 GPU 頻率、耗電和過熱的程度（要這台 Mac 的登入密碼一次；不給也可以照常測）
PM=0
KEEP=""
PMPID=""
# 不管怎麼結束（跑完、出錯、按 Ctrl+C、關掉視窗），都把背景的權限續期和 powermetrics 關掉
trap '[ -n "$KEEP" ] && kill "$KEEP" 2>/dev/null; [ -n "$PMPID" ] && sudo -n kill -TERM "$PMPID" 2>/dev/null' EXIT
say ""
say "要不要順便記錄 GPU 頻率、耗電和過熱的程度？要輸入這台 Mac 的登入密碼一次（只用來讀這些狀態）。"
read -r -p "要記錄請輸入 y 再按 Enter；不要就直接按 Enter： " ans || ans=""
if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
  say "請輸入這台 Mac 的登入密碼（輸入時畫面不會顯示任何字，打完按 Enter）："
  if sudo -v; then
    PM=1
    ( while true; do sudo -n true 2>/dev/null; sleep 50; done ) &
    KEEP=$!
    say "好，會一起記錄。"
  else
    say "密碼沒有成功，這次不記錄這些狀態，測試照常進行。"
  fi
fi

[ "$(uname -s)" = "Darwin" ] || die "這個檔案只能在 Mac 上執行。"
[ "$(uname -m)" = "arm64" ] || die "這台 Mac 不是 Apple 晶片（M1、M2、M3、M4……），這個測試只支援 Apple 晶片。"
if ! pmset -g batt 2>/dev/null | grep -q "AC Power"; then
  say "偵測到沒有插電。請插上電源（每 3 秒檢查一次；要放棄請關掉這個視窗）。"
  while ! pmset -g batt 2>/dev/null | grep -q "AC Power"; do sleep 3; done
  say "已經插上電源。"
fi
free_gb=$(df -g "$HOME" 2>/dev/null | awk 'NR==2 {print $4}')
[ "${free_gb:-0}" -ge 5 ] 2>/dev/null || die "磁碟空間不夠（至少要 5 GB，現在剩 ${free_gb:-?} GB）。"
mkdir -p "$W/model" || die "無法建立資料夾 $W。"
cd "$W" || die "無法進入資料夾 $W。"

fetch() {   # 網址、SHA-256、存成的檔名、說明
  if [ -f "$3" ] && [ "$(sha "$3")" = "$2" ]; then say "  已經下載過：$4"; return 0; fi
  say "  下載 $4……"
  n=0
  while [ $n -lt 3 ]; do
    n=$((n + 1))
    curl -L --fail --retry 3 --connect-timeout 30 -C - -o "$3" "$1" && break
    say "  下載中斷，5 秒後重試（$n/3）……"
    sleep 5
  done
  if [ "$(sha "$3")" != "$2" ]; then
    rm -f "$3"
    die "$4 下載得不完整或不對。請確認網路正常，再執行一次（會重新下載）。"
  fi
  say "  確認無誤：$4"
}

say ""
say "下載需要的檔案（模型約 2.1 GB，視網路速度要幾分鐘到二十幾分鐘）……"
fetch "$PY_URL" "$PY_SHA" "$W/python.tar.gz" "免安裝的 Python（約 25 MB）"
fetch "$LL_URL" "$LL_SHA" "$W/llama.tar.gz" "llama.cpp b9549（約 11 MB）"
fetch "$MD_URL" "$MD_SHA" "$W/model/qwen2.5-coder-3b-instruct-q4_k_m.gguf" "模型 Qwen2.5-Coder-3B（約 2.1 GB）"
[ -x "$W/python/bin/python3.12" ] || tar -xzf "$W/python.tar.gz" -C "$W" || die "Python 解壓縮失敗。"
[ -x "$W/llama-b9549/llama-server" ] || tar -xzf "$W/llama.tar.gz" -C "$W" || die "llama.cpp 解壓縮失敗。"
xattr -dr com.apple.quarantine "$W/llama-b9549" "$W/python" 2>/dev/null || true
[ -x "$W/python/bin/python3.12" ] && [ -x "$W/llama-b9549/llama-server" ] || die "解壓縮後找不到需要的程式。"

cat > "$W/mac_idle_test.py" <<'PY_EOF'
"""LLM 閒置測試（Mac 版，2026-10-19）

和朋友測試包（Windows）的 GPU 測試流程相同：每輪重開伺服器、暖機 3 個，閒置 0、2、3、5、8 秒各 3 個請求為一個區塊，
每輪 4 個區塊、共 5 輪；不算的輪自動在最後補跑（最多 3 輪）。第 5 輪不記錄 GPU 狀態，當對照。
Mac 專用的部分：防睡眠用外層的 caffeinate、插電用 pmset、CPU 使用率用 iostat、GPU 使用率用 ioreg（都不需要系統管理員權限）。

用法（由 mac_idle_test.sh 呼叫）：python3 -I mac_idle_test.py <工作資料夾> <版本> [--quick] [--exe 路徑] [--model 路徑]
"""
import base64
import csv
import datetime
import json
import os
import random
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import zipfile


def arg_value(name, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


W = os.path.abspath(sys.argv[1])
KIT_VERSION = sys.argv[2]
QUICK = "--quick" in sys.argv                      # 只給作者試跑用：1 輪、15 個請求
EXE = arg_value("--exe", os.path.join(W, "llama-b9549", "llama-server"))
MODEL = arg_value("--model", os.path.join(W, "model", "qwen2.5-coder-3b-instruct-q4_k_m.gguf"))
WORK = os.path.join(W, "work")

GAPS = [0, 2, 3, 5, 8]                              # 閒置秒數（和 Windows 版相同）
REPS, BLOCKS, WARMUP, ROUNDS = 3, 4, 3, 5
if QUICK:
    BLOCKS, ROUNDS = 1, 1
WORKLOADS = {"main": ("medium.txt", 128),           # 838 token prompt、128 token 輸出（和 Windows 版 GPU 測試相同）
             "short": ("short.txt", 16)}            # 太慢的裝置改用這個
SLOW_GPU_S = 8.0
CPU_BUSY_MAX, GPU_BUSY_MAX, MIN_GPU_SAMPLES = 30.0, 10.0, 20
GUARD_START, GUARD_END = 1.2, 0.2                   # 算等待期間 GPU 使用率時，去掉每段等待的開頭和結尾
STALL_MAX, LONG_REQ_S = 2.0, 120.0
# 提示詞：和 Windows 版 _kit/prompts 裡的檔案用文字模式讀出來的內容相同
PROMPTS = {"medium.txt": base64.b64decode("UmVhZCB0aGUgbm90ZXMgYmVsb3cgYW5kIHRoZW4gd3JpdGUgYSBkZXRhaWxlZCB0ZWNobmljYWwgc3VtbWFyeSBvZiBhdCBsZWFzdCAzMDAgd29yZHMgYWJvdXQgcXVldWVpbmcgc3lzdGVtcywgY292ZXJpbmcgYXJyaXZhbHMsIHNlcnZpY2UgdGltZXMsIHV0aWxpemF0aW9uIGFuZCB3YWl0aW5nIHRpbWUuCgpOb3RlczoKQSBxdWV1ZWluZyBzeXN0ZW0gY29uc2lzdHMgb2YgYXJyaXZhbHMsIGEgcXVldWUsIGFuZCBvbmUgb3IgbW9yZSBzZXJ2ZXJzLiBUaGUgYXJyaXZhbCBwcm9jZXNzIGRlc2NyaWJlcyB3aGVuIHJlcXVlc3RzIGVudGVyIHRoZSBzeXN0ZW0sIHdoaWxlIHRoZSBzZXJ2aWNlIHByb2Nlc3MgZGVzY3JpYmVzIGhvdyBsb25nIGVhY2ggcmVxdWVzdCBvY2N1cGllcyBhIHNlcnZlci4gVXRpbGl6YXRpb24gaXMgdGhlIGZyYWN0aW9uIG9mIHRpbWUgYSBzZXJ2ZXIgaXMgYnVzeSwgYW5kIHdhaXRpbmcgdGltZSBncm93cyBzaGFycGx5IGFzIHV0aWxpemF0aW9uIGFwcHJvYWNoZXMgb25lLiBWYXJpYWJpbGl0eSBpbiBzZXJ2aWNlIHRpbWVzIG1ha2VzIHdhaXRpbmcgd29yc2UgZXZlbiB3aGVuIHRoZSBhdmVyYWdlIHNlcnZpY2UgdGltZSBzdGF5cyB0aGUgc2FtZS4gQSBxdWV1ZWluZyBzeXN0ZW0gY29uc2lzdHMgb2YgYXJyaXZhbHMsIGEgcXVldWUsIGFuZCBvbmUgb3IgbW9yZSBzZXJ2ZXJzLiBUaGUgYXJyaXZhbCBwcm9jZXNzIGRlc2NyaWJlcyB3aGVuIHJlcXVlc3RzIGVudGVyIHRoZSBzeXN0ZW0sIHdoaWxlIHRoZSBzZXJ2aWNlIHByb2Nlc3MgZGVzY3JpYmVzIGhvdyBsb25nIGVhY2ggcmVxdWVzdCBvY2N1cGllcyBhIHNlcnZlci4gVXRpbGl6YXRpb24gaXMgdGhlIGZyYWN0aW9uIG9mIHRpbWUgYSBzZXJ2ZXIgaXMgYnVzeSwgYW5kIHdhaXRpbmcgdGltZSBncm93cyBzaGFycGx5IGFzIHV0aWxpemF0aW9uIGFwcHJvYWNoZXMgb25lLiBWYXJpYWJpbGl0eSBpbiBzZXJ2aWNlIHRpbWVzIG1ha2VzIHdhaXRpbmcgd29yc2UgZXZlbiB3aGVuIHRoZSBhdmVyYWdlIHNlcnZpY2UgdGltZSBzdGF5cyB0aGUgc2FtZS4gQSBxdWV1ZWluZyBzeXN0ZW0gY29uc2lzdHMgb2YgYXJyaXZhbHMsIGEgcXVldWUsIGFuZCBvbmUgb3IgbW9yZSBzZXJ2ZXJzLiBUaGUgYXJyaXZhbCBwcm9jZXNzIGRlc2NyaWJlcyB3aGVuIHJlcXVlc3RzIGVudGVyIHRoZSBzeXN0ZW0sIHdoaWxlIHRoZSBzZXJ2aWNlIHByb2Nlc3MgZGVzY3JpYmVzIGhvdyBsb25nIGVhY2ggcmVxdWVzdCBvY2N1cGllcyBhIHNlcnZlci4gVXRpbGl6YXRpb24gaXMgdGhlIGZyYWN0aW9uIG9mIHRpbWUgYSBzZXJ2ZXIgaXMgYnVzeSwgYW5kIHdhaXRpbmcgdGltZSBncm93cyBzaGFycGx5IGFzIHV0aWxpemF0aW9uIGFwcHJvYWNoZXMgb25lLiBWYXJpYWJpbGl0eSBpbiBzZXJ2aWNlIHRpbWVzIG1ha2VzIHdhaXRpbmcgd29yc2UgZXZlbiB3aGVuIHRoZSBhdmVyYWdlIHNlcnZpY2UgdGltZSBzdGF5cyB0aGUgc2FtZS4gQSBxdWV1ZWluZyBzeXN0ZW0gY29uc2lzdHMgb2YgYXJyaXZhbHMsIGEgcXVldWUsIGFuZCBvbmUgb3IgbW9yZSBzZXJ2ZXJzLiBUaGUgYXJyaXZhbCBwcm9jZXNzIGRlc2NyaWJlcyB3aGVuIHJlcXVlc3RzIGVudGVyIHRoZSBzeXN0ZW0sIHdoaWxlIHRoZSBzZXJ2aWNlIHByb2Nlc3MgZGVzY3JpYmVzIGhvdyBsb25nIGVhY2ggcmVxdWVzdCBvY2N1cGllcyBhIHNlcnZlci4gVXRpbGl6YXRpb24gaXMgdGhlIGZyYWN0aW9uIG9mIHRpbWUgYSBzZXJ2ZXIgaXMgYnVzeSwgYW5kIHdhaXRpbmcgdGltZSBncm93cyBzaGFycGx5IGFzIHV0aWxpemF0aW9uIGFwcHJvYWNoZXMgb25lLiBWYXJpYWJpbGl0eSBpbiBzZXJ2aWNlIHRpbWVzIG1ha2VzIHdhaXRpbmcgd29yc2UgZXZlbiB3aGVuIHRoZSBhdmVyYWdlIHNlcnZpY2UgdGltZSBzdGF5cyB0aGUgc2FtZS4gQSBxdWV1ZWluZyBzeXN0ZW0gY29uc2lzdHMgb2YgYXJyaXZhbHMsIGEgcXVldWUsIGFuZCBvbmUgb3IgbW9yZSBzZXJ2ZXJzLiBUaGUgYXJyaXZhbCBwcm9jZXNzIGRlc2NyaWJlcyB3aGVuIHJlcXVlc3RzIGVudGVyIHRoZSBzeXN0ZW0sIHdoaWxlIHRoZSBzZXJ2aWNlIHByb2Nlc3MgZGVzY3JpYmVzIGhvdyBsb25nIGVhY2ggcmVxdWVzdCBvY2N1cGllcyBhIHNlcnZlci4gVXRpbGl6YXRpb24gaXMgdGhlIGZyYWN0aW9uIG9mIHRpbWUgYSBzZXJ2ZXIgaXMgYnVzeSwgYW5kIHdhaXRpbmcgdGltZSBncm93cyBzaGFycGx5IGFzIHV0aWxpemF0aW9uIGFwcHJvYWNoZXMgb25lLiBWYXJpYWJpbGl0eSBpbiBzZXJ2aWNlIHRpbWVzIG1ha2VzIHdhaXRpbmcgd29yc2UgZXZlbiB3aGVuIHRoZSBhdmVyYWdlIHNlcnZpY2UgdGltZSBzdGF5cyB0aGUgc2FtZS4gQSBxdWV1ZWluZyBzeXN0ZW0gY29uc2lzdHMgb2YgYXJyaXZhbHMsIGEgcXVldWUsIGFuZCBvbmUgb3IgbW9yZSBzZXJ2ZXJzLiBUaGUgYXJyaXZhbCBwcm9jZXNzIGRlc2NyaWJlcyB3aGVuIHJlcXVlc3RzIGVudGVyIHRoZSBzeXN0ZW0sIHdoaWxlIHRoZSBzZXJ2aWNlIHByb2Nlc3MgZGVzY3JpYmVzIGhvdyBsb25nIGVhY2ggcmVxdWVzdCBvY2N1cGllcyBhIHNlcnZlci4gVXRpbGl6YXRpb24gaXMgdGhlIGZyYWN0aW9uIG9mIHRpbWUgYSBzZXJ2ZXIgaXMgYnVzeSwgYW5kIHdhaXRpbmcgdGltZSBncm93cyBzaGFycGx5IGFzIHV0aWxpemF0aW9uIGFwcHJvYWNoZXMgb25lLiBWYXJpYWJpbGl0eSBpbiBzZXJ2aWNlIHRpbWVzIG1ha2VzIHdhaXRpbmcgd29yc2UgZXZlbiB3aGVuIHRoZSBhdmVyYWdlIHNlcnZpY2UgdGltZSBzdGF5cyB0aGUgc2FtZS4gQSBxdWV1ZWluZyBzeXN0ZW0gY29uc2lzdHMgb2YgYXJyaXZhbHMsIGEgcXVldWUsIGFuZCBvbmUgb3IgbW9yZSBzZXJ2ZXJzLiBUaGUgYXJyaXZhbCBwcm9jZXNzIGRlc2NyaWJlcyB3aGVuIHJlcXVlc3RzIGVudGVyIHRoZSBzeXN0ZW0sIHdoaWxlIHRoZSBzZXJ2aWNlIHByb2Nlc3MgZGVzY3JpYmVzIGhvdyBsb25nIGVhY2ggcmVxdWVzdCBvY2N1cGllcyBhIHNlcnZlci4gVXRpbGl6YXRpb24gaXMgdGhlIGZyYWN0aW9uIG9mIHRpbWUgYSBzZXJ2ZXIgaXMgYnVzeSwgYW5kIHdhaXRpbmcgdGltZSBncm93cyBzaGFycGx5IGFzIHV0aWxpemF0aW9uIGFwcHJvYWNoZXMgb25lLiBWYXJpYWJpbGl0eSBpbiBzZXJ2aWNlIHRpbWVzIG1ha2VzIHdhaXRpbmcgd29yc2UgZXZlbiB3aGVuIHRoZSBhdmVyYWdlIHNlcnZpY2UgdGltZSBzdGF5cyB0aGUgc2FtZS4gQSBxdWV1ZWluZyBzeXN0ZW0gY29uc2lzdHMgb2YgYXJyaXZhbHMsIGEgcXVldWUsIGFuZCBvbmUgb3IgbW9yZSBzZXJ2ZXJzLiBUaGUgYXJyaXZhbCBwcm9jZXNzIGRlc2NyaWJlcyB3aGVuIHJlcXVlc3RzIGVudGVyIHRoZSBzeXN0ZW0sIHdoaWxlIHRoZSBzZXJ2aWNlIHByb2Nlc3MgZGVzY3JpYmVzIGhvdyBsb25nIGVhY2ggcmVxdWVzdCBvY2N1cGllcyBhIHNlcnZlci4gVXRpbGl6YXRpb24gaXMgdGhlIGZyYWN0aW9uIG9mIHRpbWUgYSBzZXJ2ZXIgaXMgYnVzeSwgYW5kIHdhaXRpbmcgdGltZSBncm93cyBzaGFycGx5IGFzIHV0aWxpemF0aW9uIGFwcHJvYWNoZXMgb25lLiBWYXJpYWJpbGl0eSBpbiBzZXJ2aWNlIHRpbWVzIG1ha2VzIHdhaXRpbmcgd29yc2UgZXZlbiB3aGVuIHRoZSBhdmVyYWdlIHNlcnZpY2UgdGltZSBzdGF5cyB0aGUgc2FtZS4gQSBxdWV1ZWluZyBzeXN0ZW0gY29uc2lzdHMgb2YgYXJyaXZhbHMsIGEgcXVldWUsIGFuZCBvbmUgb3IgbW9yZSBzZXJ2ZXJzLiBUaGUgYXJyaXZhbCBwcm9jZXNzIGRlc2NyaWJlcyB3aGVuIHJlcXVlc3RzIGVudGVyIHRoZSBzeXN0ZW0sIHdoaWxlIHRoZSBzZXJ2aWNlIHByb2Nlc3MgZGVzY3JpYmVzIGhvdyBsb25nIGVhY2ggcmVxdWVzdCBvY2N1cGllcyBhIHNlcnZlci4gVXRpbGl6YXRpb24gaXMgdGhlIGZyYWN0aW9uIG9mIHRpbWUgYSBzZXJ2ZXIgaXMgYnVzeSwgYW5kIHdhaXRpbmcgdGltZSBncm93cyBzaGFycGx5IGFzIHV0aWxpemF0aW9uIGFwcHJvYWNoZXMgb25lLiBWYXJpYWJpbGl0eSBpbiBzZXJ2aWNlIHRpbWVzIG1ha2VzIHdhaXRpbmcgd29yc2UgZXZlbiB3aGVuIHRoZSBhdmVyYWdlIHNlcnZpY2UgdGltZSBzdGF5cyB0aGUgc2FtZS4gQSBxdWV1ZWluZyBzeXN0ZW0gY29uc2lzdHMgb2YgYXJyaXZhbHMsIGEgcXVldWUsIGFuZCBvbmUgb3IgbW9yZSBzZXJ2ZXJzLiBUaGUgYXJyaXZhbCBwcm9jZXNzIGRlc2NyaWJlcyB3aGVuIHJlcXVlc3RzIGVudGVyIHRoZSBzeXN0ZW0sIHdoaWxlIHRoZSBzZXJ2aWNlIHByb2Nlc3MgZGVzY3JpYmVzIGhvdyBsb25nIGVhY2ggcmVxdWVzdCBvY2N1cGllcyBhIHNlcnZlci4gVXRpbGl6YXRpb24gaXMgdGhlIGZyYWN0aW9uIG9mIHRpbWUgYSBzZXJ2ZXIgaXMgYnVzeSwgYW5kIHdhaXRpbmcgdGltZSBncm93cyBzaGFycGx5IGFzIHV0aWxpemF0aW9uIGFwcHJvYWNoZXMgb25lLiBWYXJpYWJpbGl0eSBpbiBzZXJ2aWNlIHRpbWVzIG1ha2VzIHdhaXRpbmcgd29yc2UgZXZlbiB3aGVuIHRoZSBhdmVyYWdlIHNlcnZpY2UgdGltZSBzdGF5cyB0aGUgc2FtZS4g").decode("utf-8"),
           "short.txt": base64.b64decode("UmVhZCB0aGUgbm90ZXMgYmVsb3cgYW5kIHRoZW4gd3JpdGUgYSBkZXRhaWxlZCB0ZWNobmljYWwgc3VtbWFyeSBvZiBhdCBsZWFzdCAzMDAgd29yZHMgYWJvdXQgcXVldWVpbmcgc3lzdGVtcywgY292ZXJpbmcgYXJyaXZhbHMsIHNlcnZpY2UgdGltZXMsIHV0aWxpemF0aW9uIGFuZCB3YWl0aW5nIHRpbWUuCgpOb3RlczoKQSBxdWV1ZWluZyBzeXN0ZW0gY29uc2lzdHMgb2YgYXJyaXZhbHMsIGEgcXVldWUsIGFuZCBvbmUgb3IgbW9yZSBzZXJ2ZXJzLiBUaGUgYXJyaXZhbCBwcm9jZXNzIGRlc2NyaWJlcyB3aGVuIHJlcXVlc3RzIGVudGVyIHRoZSBzeXN0ZW0sIHdoaWxlIHRoZSBzZXJ2aWNlIHByb2Nlc3MgZGVzY3JpYmVzIGhvdyBsb25nIGVhY2ggcmVxdWVzdCBvY2N1cGllcyBhIHNlcnZlci4gVXRpbGl6YXRpb24gaXMgdGhlIGZyYWN0aW9uIG9mIHRpbWUgYSBzZXJ2ZXIgaXMgYnVzeSwgYW5kIHdhaXRpbmcgdGltZSBncm93cyBzaGFycGx5IGFzIHV0aWxpemF0aW9uIGFwcHJvYWNoZXMgb25lLiBWYXJpYWJpbGl0eSBpbiBzZXJ2aWNlIHRpbWVzIG1ha2VzIHdhaXRpbmcgd29yc2UgZXZlbiB3aGVuIHRoZSBhdmVyYWdlIHNlcnZpY2UgdGltZSBzdGF5cyB0aGUgc2FtZS4g").decode("utf-8")}
REASON_ZH = {"placement": "模型沒有完全放進 GPU", "request_failed": "有請求失敗或伺服器中斷", "battery": "沒有插電",
             "cpu_busy": "等待時 CPU 太忙", "gpu_busy": "等待時 GPU 被其他程式使用", "sleep": "電腦中途睡著或卡住"}


def say(msg=""):
    print(msg, flush=True)


def pause(msg="按 Enter 繼續……"):
    try:
        input(msg)
    except EOFError:
        pass


def run_text(cmd, timeout=60):
    """執行 Mac 內建指令，回傳輸出文字（失敗回傳空字串）"""
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return r.stdout.decode("utf-8", "replace")
    except Exception:
        return ""


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def on_ac_power():
    """回傳 (有沒有電池, 有沒有插電)；讀不到就當作插著電（例如桌上型 Mac）"""
    out = run_text(["pmset", "-g", "batt"], timeout=10)
    if not out:
        return False, True
    return "InternalBattery" in out, "AC Power" in out


def parse_top_cpu(line):
    """top 的一行「CPU usage: 3.12% user, 5.43% sys, 91.44% idle」→ CPU 使用率 %；其他行回傳 None"""
    m = re.search(r"CPU usage:\s*([\d.]+)% user,\s*([\d.]+)% sys,\s*([\d.]+)% idle", line)
    return round(100.0 - float(m.group(3)), 1) if m else None


def parse_gpu_util(text):
    """ioreg 的 IOAccelerator 裡的 "Device Utilization %"=12；有好幾個就取最大"""
    vals = [float(v) for v in re.findall(r'"Device Utilization %"\s*=\s*(\d+)', text)]
    return max(vals) if vals else None


def gpu_util_now():
    return parse_gpu_util(run_text(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], timeout=5))


REDACT = re.compile(r"^(\s*)([^:\n]*(?:Serial|UUID|UDID)[^:\n]*?)\s*:.*$", re.M | re.I)    # 鍵名有 Serial、UUID、UDID 的都遮


def collect_hw():
    sp = run_text(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"], timeout=180)
    sp = REDACT.sub(lambda m: f"{m.group(1)}{m.group(2)}: <removed>", sp)

    def sysctl(k):
        return run_text(["sysctl", "-n", k], timeout=10).strip()

    hw = {"chip": sysctl("machdep.cpu.brand_string"), "model": sysctl("hw.model"), "memsize_bytes": sysctl("hw.memsize"),
          "ncpu": sysctl("hw.ncpu"), "perf_cores": sysctl("hw.perflevel0.physicalcpu"),
          "eff_cores": sysctl("hw.perflevel1.physicalcpu"), "sw_vers": run_text(["sw_vers"], timeout=10).strip(),
          "uname": run_text(["uname", "-mrsv"], timeout=10).strip(), "pmset": run_text(["pmset", "-g"], timeout=10).strip(),
          "pmset_batt": run_text(["pmset", "-g", "batt"], timeout=10).strip(),
          "pmset_therm": run_text(["pmset", "-g", "therm"], timeout=10).strip(), "system_profiler": sp.strip()}
    disp = sp.split("Graphics/Displays:")[-1] if "Graphics/Displays:" in sp else ""
    m = re.search(r"Total Number of Cores:\s*(\d+)", disp)
    hw["gpu_cores"] = int(m.group(1)) if m else None
    hw["displays"] = len(re.findall(r"^\s+Resolution:", disp, flags=re.M)) if disp else None
    m = re.search(r"lowpowermode\s+(\d)", hw["pmset"])
    hw["low_power_mode"] = int(m.group(1)) if m else None
    return hw


# ---------------------------------------------------------------- 伺服器與請求（和 Windows 版相同）
class Server:
    def __init__(self, exe, args, log_path, cwd=None):
        self.exe, self.args, self.log_path, self.cwd = exe, args, log_path, cwd
        self.proc, self.port = None, free_port()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout=240):
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        cmd = [self.exe] + self.args + ["--host", "127.0.0.1", "--port", str(self.port)]
        self.log.write("COMMAND: " + " ".join(cmd) + "\n")
        self.log.flush()
        self.proc = subprocess.Popen(cmd, cwd=self.cwd or os.path.dirname(self.exe), stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                if urllib.request.urlopen(self.base + "/health", timeout=3).status == 200:
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()
        if getattr(self, "log", None):
            self.log.close()

    def request(self, prompt, max_tokens):
        """和論文的閒置測試完全相同的請求；回傳 (送出, 第一個 token, 完成) 的 perf_counter、牆上時間、token 數"""
        payload = {"model": "local", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0.0, "stream": True, "ignore_eos": True}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
                                     method="POST", headers={"Content-Type": "application/json",
                                                             "Authorization": "Bearer sk-no-key-required"})
        wall, t_send, t_first, n = time.time(), time.perf_counter(), None, 0
        with urllib.request.urlopen(req, timeout=900) as resp:
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


def offload_info(log_path):
    try:
        txt = open(log_path, encoding="utf-8", errors="replace").read()
    except Exception:
        return {}
    m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", txt)
    lines = [l.strip() for l in txt.splitlines() if re.search(r"Metal|MTL|offload|device|CPU_Mapped|buffer size", l)]
    return {"offloaded": m.group(1) if m else None, "layers": m.group(2) if m else None, "log_lines": lines[:40]}


# ---------------------------------------------------------------- 狀態記錄
class CpuMonitor:
    """top 每秒一筆的 CPU 使用率（整台電腦）。每一輪都開，用來算等待期間 CPU 忙不忙（Windows 版用 GetSystemTimes）"""
    def __init__(self, path):
        self.samples, self.path, self.proc = [], path, None
        try:
            self.proc = subprocess.Popen(["top", "-l", "0", "-s", "1", "-n", "0"], stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL)
            threading.Thread(target=self._read, daemon=True).start()
        except Exception:
            self.proc = None

    def _read(self):
        first = True
        for raw in self.proc.stdout:
            t = time.time()
            u = parse_top_cpu(raw.decode("utf-8", "replace"))
            if u is None:
                continue
            if first:                       # 第一筆沒有前一次可以比，數值不準，不用
                first = False
                continue
            self.samples.append((t, u))     # 這一筆代表 t 之前約 1 秒的平均

    def stop(self):
        if self.proc:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("time,cpu_percent\n")
            for t, u in list(self.samples):
                f.write(f"{t:.3f},{u}\n")

    def mean_between(self, a, b):
        """等待期間 [a, b] 的 CPU 使用率：只用整個落在裡面的取樣（每筆代表前 1 秒）"""
        v = [u for t, u in list(self.samples) if a + 1.0 <= t <= b + 0.1]
        return round(sum(v) / len(v), 1) if v else None


class GpuMonitor:
    """ioreg 每 0.5 秒讀一次 GPU 使用率。只在記錄狀態的輪次開（第 5 輪不開，當對照）"""
    def __init__(self, path):
        self.samples, self.path, self.ev = [], path, threading.Event()
        self.th = threading.Thread(target=self._poll, daemon=True)
        self.th.start()

    def _poll(self):
        while not self.ev.is_set():
            t = time.time()
            u = gpu_util_now()
            if u is not None:
                self.samples.append((t, u))
            self.ev.wait(0.5)

    def stop(self):
        self.ev.set()
        self.th.join(timeout=10)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("time,gpu_device_util_percent\n")
            for t, u in list(self.samples):
                f.write(f"{t:.3f},{u}\n")

    def util_in_waits(self, waits):
        wins = [(a + GUARD_START, b - GUARD_END) for a, b in waits if b - GUARD_END > a + GUARD_START]
        v = [u for t, u in list(self.samples) if any(a <= t <= b for a, b in wins)]
        return (sum(v) / len(v) if v else None), len(v)


def busy_check():
    """開跑前 3 秒：CPU 和 GPU 使用率"""
    cpu = CpuMonitor(os.devnull)
    g = []
    for _ in range(6):
        u = gpu_util_now()
        if u is not None:
            g.append(u)
        time.sleep(0.5)
    time.sleep(1.5)
    cpu.stop()
    c = [u for _, u in cpu.samples]
    return (round(sum(c) / len(c), 1) if c else None), (round(sum(g) / len(g), 1) if g else None)


# ---------------------------------------------------------------- 個資和打包
def secrets():
    """要遮的字：(文字, 標籤, 是不是路徑)。路徑直接換；名稱要整個字相同才換，免得把紀錄裡剛好相同的字母也換掉"""
    items = []
    for v, tag in ((W, "<TESTFOLDER>"), (os.path.expanduser("~"), "<HOME>")):
        items += [(v, tag, True), (v.replace("\\", "/"), tag, True)]            # 兩種斜線的寫法都遮
    for k in ("USER", "LOGNAME", "USERNAME"):
        items.append((os.environ.get(k, ""), "<USER>", False))
    items.append((socket.gethostname(), "<COMPUTERNAME>", False))
    for k in ("ComputerName", "LocalHostName", "HostName"):
        items.append((run_text(["scutil", "--get", k], timeout=10).strip(), "<COMPUTERNAME>", False))
    items = list({x[0]: x for x in items if x[0] and len(x[0]) >= 2}.values())
    return sorted(items, key=lambda x: -len(x[0]))        # 先換最長的（路徑裡包含使用者名稱）


def sanitize_text(txt, items):
    for v, tag, is_path in items:
        pat = re.escape(v) if is_path else r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])"
        txt = re.sub(pat, lambda m: tag, txt, flags=re.I)
    return txt


def desktop_dir():
    d = os.path.join(os.path.expanduser("~"), "Desktop")
    return d if os.path.isdir(d) else W


def package(run_dir, label, unfinished):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    name = f"閒置測試結果_{label}_{stamp}{'_未完成' if unfinished else ''}.zip"
    items = secrets()
    for folder in (desktop_dir(), W):          # 桌面不能寫（例如沒有按「允許」）就存在測試資料夾
        dest = os.path.join(folder, name)
        try:
            with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
                for fn in sorted(os.listdir(run_dir)):
                    p = os.path.join(run_dir, fn)
                    if fn.endswith((".log", ".json", ".txt")):          # 數據檔（.csv）只有數字，不用遮
                        z.writestr(fn, sanitize_text(open(p, encoding="utf-8", errors="replace").read(), items))
                    else:
                        z.write(p, fn)
            return dest
        except OSError:
            continue
    raise OSError("桌面和測試資料夾都不能寫入")


def short_label(chip):
    s = re.sub(r"[^A-Za-z0-9]+", "-", (chip or "Mac").strip()).strip("-")
    return "GPU_" + (s[:40] or "Mac")


# ---------------------------------------------------------------- 主流程
FIELDS = ["round", "req_id", "block", "gap", "idle_before", "t_send", "wall_send", "ttft", "service",
          "out_tokens", "workload", "monitor", "cpu_wait", "error", "stall"]


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def run():
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(WORK, f"gpu_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    info = {"kit_version": KIT_VERSION, "platform": "macOS", "mode": "gpu", "start": datetime.datetime.now().isoformat(),
            "gaps": GAPS, "reps": REPS, "blocks": BLOCKS, "rounds": ROUNDS, "warmup": WARMUP, "rounds_done": 0,
            "no_monitor_rounds": [r for r in range(1, ROUNDS + 1) if r % 5 == 0], "notes": [], "quick": QUICK}
    label, server, unfinished, stopped = "GPU_Mac", None, True, False
    cpu_mon = gpu_mon = None
    all_rows = []
    req_path = os.path.join(run_dir, "requests.csv")
    try:
        has_batt, ac = on_ac_power()
        info["laptop_battery"] = has_batt
        while has_batt and not ac:
            say("偵測到沒有插電。請插上電源（每 3 秒檢查一次；要放棄請關掉這個視窗）。")
            time.sleep(3)
            has_batt, ac = on_ac_power()

        say("記錄硬體資訊……")
        hw = collect_hw()
        json.dump(hw, open(os.path.join(run_dir, "hardware.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        chip = hw.get("chip") or "Apple"
        label = short_label(chip)
        info.update(device="Metal", device_name=f"{chip} GPU（{hw.get('gpu_cores') or '?'} 核，Metal）", label=label,
                    integrated=True, threads=4)
        say(f"使用的晶片：{chip}（GPU {hw.get('gpu_cores') or '?'} 核）")
        if hw.get("low_power_mode"):
            say("注意：這台 Mac 開著「低耗電模式」。測試照樣跑，結果會記下這個設定。")
            info["notes"].append("low power mode on")

        args = ["-m", MODEL, "-c", "4096", "--parallel", "1", "-t", "4", "--cache-ram", "0", "-ngl", "all"]
        args += arg_value("--server-args", "").split()                  # 只給作者試跑用
        say("確認模型能不能全部放進 GPU……")
        probe_log = os.path.join(run_dir, "placement_probe.log")
        probe = Server(EXE, args + ["-lv", "4"], probe_log, cwd=os.path.dirname(EXE))
        probe_ok = probe.start()
        probe.stop()
        pr = offload_info(probe_log)
        pr.update(started=probe_ok)
        info["placement_probe"] = pr
        if not probe_ok:
            raise RuntimeError("llama-server 沒有啟動成功（見 placement_probe.log）")
        probe_full = bool(pr.get("offloaded")) and pr["offloaded"] == pr.get("layers")
        version = ""
        try:
            r = subprocess.run([EXE, "--version"], capture_output=True, timeout=60, cwd=os.path.dirname(EXE))
            version = (r.stdout + r.stderr).decode("utf-8", "replace")
        except Exception:
            pass
        m = re.search(r"^version:.*$", version, flags=re.M)
        info["llama_version"] = m.group(0).strip() if m else (version.strip()[-200:] or None)
        info["python_version"] = sys.version

        cpu_u, gpu_u = busy_check()
        info["busy_before"] = {"cpu_percent": cpu_u, "gpu_percent": gpu_u}
        if (cpu_u is not None and cpu_u > 30) or (gpu_u is not None and gpu_u > 15):
            say(f"電腦好像還在忙（CPU {cpu_u}%、GPU {gpu_u}%）。請關掉其他程式。")
            pause("關好之後按 Enter 繼續……")

        workload, reps = "main", REPS
        write_rows(req_path, all_rows)
        info["rounds_info"] = []
        t_begin = time.time()
        plan = [r % 5 != 0 for r in range(1, ROUNDS + 1)]
        cap = (ROUNDS + 1) // 2
        makeups, rnd = 0, 0
        while rnd < len(plan):
            rnd += 1
            monitor_on = plan[rnd - 1]
            tag = f"第 {rnd} 輪" + ("（補跑）" if rnd > ROUNDS else "")
            say()
            say(f"GPU 測試{tag}：啟動伺服器……")
            log_path = os.path.join(run_dir, f"round{rnd}_server.log")
            server = Server(EXE, args, log_path, cwd=os.path.dirname(EXE))
            if not server.start():
                info["notes"].append(f"round {rnd}: server failed to start")
                say("伺服器沒有啟動成功，停止這個測試（已經跑完的部分還是會打包）。")
                stopped = True
                break
            rinfo = {"round": rnd, "makeup": rnd > ROUNDS, "monitor": int(monitor_on), "gpu_mem_gb": None,
                     "monitors_active_start": None, "monitors_active_end": None,
                     "placement": "partial" if (pr.get("offloaded") and not probe_full) else "unknown"}
            placement_ok = rinfo["placement"] != "partial"
            errors, on_batt, waits, stalls = [], False, [], []
            prompt_file, max_tokens = WORKLOADS[workload]
            prompt = PROMPTS[prompt_file]
            try:
                warm = []
                for i in range(WARMUP):                                 # 暖機，不記錄
                    ts, tf, td, wall, n = server.request(f"[warmup {rnd}-{i}] " + prompt, max_tokens)
                    warm.append(td - ts)
                rinfo["warmup_service"] = [round(x, 3) for x in warm]
                if rnd == 1 and workload == "main" and statistics.median(warm) > SLOW_GPU_S:
                    workload = "short"
                    info["notes"].append(f"slow device ({statistics.median(warm):.1f} s per request): switched to the short workload")
                    say("這台跑得比較慢，改用比較短的請求。")
                    prompt_file, max_tokens = WORKLOADS[workload]
                    prompt = PROMPTS[prompt_file]
                    for i in range(WARMUP):
                        server.request(f"[warmup {rnd}-s{i}] " + prompt, max_tokens)
            except Exception as e:
                errors.append(f"warmup: {type(e).__name__}: {e}"[:200])
            info["workload"], info["reps_used"] = workload, reps

            cpu_mon = CpuMonitor(os.path.join(run_dir, f"round{rnd}_cpu.csv"))
            gpu_mon = GpuMonitor(os.path.join(run_dir, f"round{rnd}_gpu.csv")) if monitor_on and not errors else None
            rng = random.Random(2026 + rnd)
            n_req, done = BLOCKS * len(GAPS) * reps, 0
            round_rows = []
            last_end, last_wall, t_round = time.perf_counter(), time.time(), time.time()
            with open(req_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDS)
                for b in range(1, BLOCKS + 1):
                    if errors:
                        break
                    has_batt, ac = on_ac_power()
                    on_batt = on_batt or (has_batt and not ac)
                    order = GAPS * reps
                    rng.shuffle(order)
                    for g in order:
                        wait_until(last_end + g)
                        rid = (rnd - 1) * 1000 + done
                        row = dict(round=rnd, req_id=rid, block=b, gap=g, workload=workload, monitor=int(monitor_on),
                                   cpu_wait="", error="")
                        try:
                            ts, tf, td, wall, n = server.request(f"[request {rid:05d}] " + prompt, max_tokens)
                        except Exception as e:
                            row["error"] = f"{type(e).__name__}: {e}"[:200]
                            errors.append(f"request {rid}: {row['error']}")
                            round_rows.append(row)
                            writer.writerow({k: row.get(k, "") for k in FIELDS})
                            fh.flush()
                            break
                        wall_end = time.time()
                        stall = max((ts - last_end) - g, (wall - last_wall) - g, (wall_end - wall) - (td - ts))
                        if stall > STALL_MAX or td - ts > LONG_REQ_S:
                            stalls.append(rid)
                        row.update(idle_before=round(ts - last_end, 4), t_send=round(ts, 4), wall_send=round(wall, 4),
                                   ttft=round(tf - ts, 4) if tf else "", service=round(td - ts, 4), out_tokens=n,
                                   stall=round(stall, 3))
                        if g >= 2:
                            row["_wait"] = (wall - (ts - last_end), wall)
                            waits.append(row["_wait"])
                        round_rows.append(row)
                        writer.writerow({k: row.get(k, "") for k in FIELDS})
                        fh.flush()
                        if n == 0:
                            errors.append(f"request {rid}: no output tokens")
                        last_end, last_wall = td, wall_end
                        done += 1
                        if done % 10 == 0 or done == n_req:
                            per = (time.time() - t_round) / done
                            left = per * (n_req - done) + (len(plan) - rnd) * (per * n_req + 30)
                            say(f"  {tag}：{done}/{n_req}，大約還要 {left / 60:.0f} 分鐘")
            time.sleep(1.2)                                             # 讓 iostat 交出最後一段等待的取樣
            if gpu_mon:
                gpu_mon.stop()
            cpu_mon.stop()
            has_batt, ac = on_ac_power()
            on_batt = on_batt or (has_batt and not ac)
            server.stop()
            server = None

            # 每段等待的 CPU 使用率，寫回 requests.csv（Windows 版是每段等待前後讀 GetSystemTimes）
            num = den = 0.0
            for r in round_rows:
                if "_wait" in r:
                    c = cpu_mon.mean_between(*r["_wait"])
                    if c is not None:
                        r["cpu_wait"] = c
                        num += c * r["idle_before"]
                        den += r["idle_before"]
            all_rows.extend(round_rows)
            write_rows(req_path, all_rows)

            reasons = []
            if not placement_ok:
                reasons.append("placement")
            if errors or done < n_req:
                reasons.append("request_failed")
            if on_batt:
                reasons.append("battery")
            if stalls:
                reasons.append("sleep")
                rinfo["stalled_requests"] = stalls[:20]
            rinfo["cpu_wait_percent"] = round(num / den, 1) if den else None
            if rinfo["cpu_wait_percent"] is not None and rinfo["cpu_wait_percent"] > CPU_BUSY_MAX:
                reasons.append("cpu_busy")
            if gpu_mon:
                u, k = gpu_mon.util_in_waits(waits)
                rinfo["gpu_wait_percent"] = round(u, 1) if u is not None else None
                rinfo["gpu_wait_samples"] = k
                if u is not None and u > GPU_BUSY_MAX:
                    reasons.append("gpu_busy")
                elif k < MIN_GPU_SAMPLES:
                    rinfo["gpu_check"] = "insufficient"
            cpu_mon = gpu_mon = None
            rinfo.update(valid=not reasons, invalid_reasons=reasons, errors=errors[:10], requests_done=done)
            if reasons:
                why = "、".join(REASON_ZH.get(r, r) for r in reasons)
                structural = "placement" in reasons
                if makeups < cap and not structural:
                    plan.append(monitor_on)
                    makeups += 1
                    say(f"  這一輪不算（{why}），最後會多跑一輪補上。")
                else:
                    say(f"  這一輪不算（{why}）。" + ("補跑的次數已經用完。" if makeups >= cap else ""))
            info["rounds_info"].append(rinfo)
            info.update(rounds_done=rnd, rounds_planned=ROUNDS, makeup_cap=cap, makeup_rounds=makeups,
                        valid_rounds=sum(1 for x in info["rounds_info"] if x.get("valid")),
                        no_monitor_rounds=[i + 1 for i, mo in enumerate(plan) if not mo])
            if rnd < len(plan):
                time.sleep(10)
        unfinished = stopped
        info["minutes"] = round((time.time() - t_begin) / 60, 1)
    except KeyboardInterrupt:
        info["notes"].append("interrupted by user")
    except Exception:
        info["notes"].append("error: " + traceback.format_exc())
        say("發生錯誤，已經跑完的部分還是會打包。")
    finally:
        for mon in (gpu_mon, cpu_mon):
            if mon:
                try:
                    mon.stop()
                except Exception:
                    pass
        if server:
            server.stop()
        info["end"] = datetime.datetime.now().isoformat()
        att = arg_value("--attach", None)                  # 10/19 加：powermetrics 的紀錄（外層用 sudo 開的），一起打包
        info["powermetrics"] = False
        if att and os.path.exists(att):
            try:
                shutil.copy(att, os.path.join(run_dir, "powermetrics.txt"))
                info["powermetrics"] = True
            except OSError:
                info["notes"].append("powermetrics file could not be copied")
        try:
            json.dump(info, open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            dest = package(run_dir, label, unfinished)
            say()
            say("=" * 60)
            where = "桌面" if os.path.dirname(dest) == desktop_dir() else "這個資料夾（" + os.path.dirname(dest) + "）"
            say(f"完成！結果已經存到{where}：" if not unfinished else f"測試沒有全部跑完，已經跑完的部分存到{where}：")
            say("  " + os.path.basename(dest))
            say("請把這個檔案用 LINE 傳給請你幫忙測試的人。")
            say("傳完之後，可以把家目錄裡的 llm_idle_test_mac 資料夾丟到垃圾桶（約 2.3 GB）。")
            say("=" * 60)
        except Exception:
            say("打包結果時出了問題，請把這個視窗截圖傳給請你幫忙測試的人。")
            say(traceback.format_exc())


if __name__ == "__main__":
    run()
PY_EOF

say ""
say "開始測試（約 40 分鐘到 1 小時多）。這段時間 Mac 不會睡著，螢幕也會一直亮著。"
if [ $PM -eq 1 ]; then
  rm -f "$W/powermetrics.txt"
  sudo -n powermetrics --samplers cpu_power,gpu_power,thermal -i 500 -o "$W/powermetrics.txt" >/dev/null 2>&1 &
  PMPID=$!
  sleep 3
  if ! ps -p "$PMPID" >/dev/null 2>&1; then
    say "（記錄 GPU 頻率的程式沒有啟動，這次不記錄，測試照常進行）"
    PM=0
  fi
fi
if [ $PM -eq 1 ]; then
  caffeinate -dims "$W/python/bin/python3.12" -I "$W/mac_idle_test.py" "$W" "$VER" --attach "$W/powermetrics.txt"
else
  caffeinate -dims "$W/python/bin/python3.12" -I "$W/mac_idle_test.py" "$W" "$VER"
fi
st=$?
[ -n "$PMPID" ] && sudo -n kill -TERM "$PMPID" 2>/dev/null
[ -n "$KEEP" ] && kill "$KEEP" 2>/dev/null
say ""
[ $st -eq 0 ] || say "（程式結束代碼 $st）"
read -r -p "按 Enter 關閉…… " _ || true
