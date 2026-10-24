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
__PYTHON_CLIENT__
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
