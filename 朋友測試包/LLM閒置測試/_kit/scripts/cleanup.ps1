# 清理：只刪測試包自己帶來的檔案（MANIFEST.txt 列出的）和測試時產生的中間檔（_kit\work）。
# 桌面上的結果 zip 不在這個資料夾裡，不會被刪；資料夾裡如果有別的檔案，也不會動。
# 10/16 加強（外部審閱建議）：
#   - 清單裡每個路徑都先確認在測試資料夾裡面；只要有一行不是，整個停下來，什麼都不刪
#   - 遇到連結（symlink、junction）不跟進去，也不刪
#   - 只刪測試包本來就有的資料夾，而且要是空的才刪
$ErrorActionPreference = "Stop"
$ReparsePoint = [System.IO.FileAttributes]::ReparsePoint

function Stop-Clean($msg) {
    Write-Host $msg
    Read-Host "按 Enter 關閉"
    exit 1
}
function Test-Link($item) { return [bool]($item.Attributes -band $ReparsePoint) }

if (-not $env:KITDIR) { Stop-Clean "找不到測試資料夾，什麼都沒有刪。" }
$root = [System.IO.Path]::GetFullPath($env:KITDIR).TrimEnd("\", "/")
$driveRoot = [System.IO.Path]::GetPathRoot($root).TrimEnd("\", "/")
if ($root -eq $driveRoot -or $root.StartsWith("\\")) { Stop-Clean "測試資料夾的位置不對（磁碟根目錄或網路位置），為了安全，什麼都沒有刪。" }
$prefix = $root + "\"
function Test-Inside($path) {
    return [System.IO.Path]::GetFullPath($path).StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)
}

$kit = Join-Path $root "_kit"
$id = Join-Path $kit "KIT_ID.txt"
$manifestPath = Join-Path $kit "MANIFEST.txt"
if (-not (Test-Path -LiteralPath $id) -or -not ((Get-Content -LiteralPath $id -Raw) -match "LLM-IDLE-TEST-KIT-2026") -or -not (Test-Path -LiteralPath $manifestPath)) {
    Stop-Clean "這個資料夾看起來不是測試資料夾，為了安全，什麼都沒有刪。"
}
foreach ($p in @($root, $kit)) {
    if (Test-Link (Get-Item -LiteralPath $p -Force)) { Stop-Clean "測試資料夾是連結，為了安全，什麼都沒有刪。" }
}
$list = @(Get-Content -LiteralPath $manifestPath -Encoding UTF8 | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
foreach ($rel in $list) {
    if ([System.IO.Path]::IsPathRooted($rel) -or $rel -match "(^|[\/])\.\.([\/]|$)" -or -not (Test-Inside (Join-Path $root $rel))) {
        Stop-Clean "清單裡有不在測試資料夾裡的路徑，為了安全，什麼都沒有刪：$rel"
    }
}

Set-Location -LiteralPath $env:TEMP
Write-Host "這會刪除測試資料夾：$root"
Write-Host ("（只刪測試包帶來的 {0} 個檔案和測試時產生的中間檔；桌面上的結果 zip 會保留。）" -f $list.Count)
$ans = Read-Host "結果 zip 已經傳出去了嗎？確定要刪除請輸入 Y 再按 Enter"
if ($ans -notmatch "^[Yy]") { Write-Host "沒有刪除任何東西。"; Start-Sleep -Seconds 2; exit 0 }
Start-Sleep -Seconds 2

# 刪一整個中間檔資料夾：不跟進連結，檔案一個一個刪，資料夾空了才刪
function Remove-Tree($dir) {
    foreach ($c in @(Get-ChildItem -LiteralPath $dir -Force)) {
        if (Test-Link $c) { continue }
        if ($c.PSIsContainer) { Remove-Tree $c.FullName } else { Remove-Item -LiteralPath $c.FullName -Force }
    }
    if (-not (Get-ChildItem -LiteralPath $dir -Force)) { Remove-Item -LiteralPath $dir -Force }
}
# 找某個名字的資料夾（不跟進連結）
function Find-Dirs($dir, $name) {
    foreach ($c in @(Get-ChildItem -LiteralPath $dir -Directory -Force -ErrorAction SilentlyContinue)) {
        if (Test-Link $c) { continue }
        if ($c.Name -eq $name) { $c.FullName } else { Find-Dirs $c.FullName $name }
    }
}

# 1. 清單上的檔案（識別檔和清單本身最後才刪）
$last = @("_kit\KIT_ID.txt", "_kit\MANIFEST.txt")
$dirs = New-Object System.Collections.Generic.HashSet[string]
foreach ($rel in $list) {
    $d = Split-Path $rel -Parent
    while ($d) { [void]$dirs.Add($d); $d = Split-Path $d -Parent }
    if ($last -contains $rel) { continue }
    $p = Join-Path $root $rel
    $item = Get-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue
    if ($item -and -not $item.PSIsContainer -and -not (Test-Link $item)) { Remove-Item -LiteralPath $p -Force }
}
# 2. 測試時產生的中間檔
$work = Join-Path $kit "work"
$extra = @()
if (Test-Path -LiteralPath $work) { $extra += $work }
$extra += @(Find-Dirs $kit "__pycache__")
foreach ($w in $extra) {
    $item = Get-Item -LiteralPath $w -Force
    if ((Test-Inside $w) -and -not (Test-Link $item)) { Remove-Tree $w }
}
# 3. 識別檔和清單
foreach ($rel in $last) {
    $p = Join-Path $root $rel
    if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Force }
}
# 4. 測試包本來就有的資料夾，空了才刪（從最深的開始）
foreach ($d in ($dirs | Sort-Object { $_.Length } -Descending)) {
    $p = Join-Path $root $d
    $item = Get-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue
    if ($item -and $item.PSIsContainer -and -not (Test-Link $item) -and -not (Get-ChildItem -LiteralPath $p -Force)) {
        Remove-Item -LiteralPath $p -Force
    }
}
if (-not (Get-ChildItem -LiteralPath $root -Force)) { Remove-Item -LiteralPath $root -Force }

if (Test-Path -LiteralPath $root) {
    Write-Host "清理完成。資料夾裡還有不是測試包帶來的東西，沒有動它們：$root"
} else {
    Write-Host "清理完成。桌面上的結果 zip 都還在。"
}
Start-Sleep -Seconds 3
