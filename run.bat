@echo off
title ImageStudio AI Workstation
setlocal enabledelayedexpansion

echo =======================================================
echo           ImageStudio AI Workstation Launcher
echo =======================================================
echo.

cd /d "%~dp0"

:: 1. Check Python Virtual Environment
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at venv\Scripts\python.exe!
    pause
    exit /b 1
)

:: 2. Free ports and ensure clean GPU VRAM
echo [1/3] Freeing ports 8188, 1420, 8400 and cleaning stale GPU processes...
taskkill /f /im llama-server.exe 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8188') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :1420') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8400') do taskkill /f /pid %%a 2>nul

:: 3. Start Python GPU AI Engine on port 8188
echo [2/3] Starting ImageStudio GPU AI Engine on port 8188...
cd /d "%~dp0ImageStudio_Tauri"
start "ImageStudio GPU Backend" /MIN "..\venv\Scripts\python.exe" backend_bridge.py

:: 4. Ensure frontend dependencies and launch Workstation
if not exist "node_modules" (
    echo [ImageStudio] Installing frontend dependencies...
    call npm install
)

echo [3/3] Waiting for AI Engine and launching Workstation at http://localhost:1420 ...
timeout /t 3 /nobreak >nul

call npm run dev -- --open

:: Cleanup backend on exit
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8188') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :1420') do taskkill /f /pid %%a 2>nul
pause