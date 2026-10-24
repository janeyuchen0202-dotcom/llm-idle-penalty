@echo off
setlocal
title LLM queue test (GPU)
cd /d "%~dp0"
if not exist "%~dp0_kit\python\python.exe" goto notready
"%~dp0_kit\python\python.exe" "%~dp0_kit\scripts\queue_test.py"
echo.
pause
exit /b 0
:notready
echo Please extract the whole zip file first,
echo then run this file inside the extracted folder.
pause
exit /b 1
