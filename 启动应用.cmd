@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0启动应用.ps1"
if errorlevel 1 pause
