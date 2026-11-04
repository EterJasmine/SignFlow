@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo 请先按照 README.md 创建本项目的 .venv 并安装依赖。
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
pause
