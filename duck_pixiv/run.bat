@echo off
title Duck Pixiv Assistant
cd /d "%~dp0"

echo ===================================================
echo   Duck Pixiv Assistant を起動しています...
echo   ポート: 9123 (競合回避モード)
echo ===================================================

:: Use the virtual environment Python if available, otherwise fallback to system python
if exist "..\venv\Scripts\python.exe" (
    "..\venv\Scripts\python.exe" app\main.py
) else if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" app\main.py
) else (
    python app\main.py
)

if %errorlevel% neq 0 (
    echo.
    echo 起動中にエラーが発生しました。
    pause
)
