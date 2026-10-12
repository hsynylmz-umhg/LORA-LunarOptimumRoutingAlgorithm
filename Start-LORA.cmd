@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Once kurulum yapin: powershell -ExecutionPolicy Bypass -File setup.ps1
  pause
  exit /b 1
)
".venv\Scripts\python.exe" lora_master.py %*
if errorlevel 1 pause
