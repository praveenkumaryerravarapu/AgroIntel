@echo off
cd /d "%~dp0"
echo Starting AgroIntel...
call venv\Scripts\activate
python app\app.py
pause
