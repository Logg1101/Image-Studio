@echo off
title Get Pixiv Refresh Token
cd /d "%~dp0"

echo ===================================================
echo   Pixiv OAuth Refresh Token Helper
echo ===================================================

if exist "..\venv\Scripts\python.exe" (
    "..\venv\Scripts\python.exe" scripts\get_pixiv_token.py
) else if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" scripts\get_pixiv_token.py
) else (
    python scripts\get_pixiv_token.py
)

pause
