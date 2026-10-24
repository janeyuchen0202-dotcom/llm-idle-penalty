@echo off
setlocal
title LLM idle test - clean up
set "KITDIR=%~dp0"
if not exist "%KITDIR%_kit\KIT_ID.txt" goto notkit
findstr /c:"LLM-IDLE-TEST-KIT-2026" "%KITDIR%_kit\KIT_ID.txt" >nul
if errorlevel 1 goto notkit
if not exist "%KITDIR%_kit\scripts\cleanup.ps1" goto notkit
cd /d "%TEMP%"
start "LLM idle test - clean up" powershell -NoProfile -ExecutionPolicy Bypass -File "%KITDIR%_kit\scripts\cleanup.ps1"
exit /b 0
:notkit
echo This file must stay inside the test folder. Nothing was deleted.
pause
exit /b 1
