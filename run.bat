@echo off
rem Market Intel — double-click to launch the dashboard.
cd /d "%~dp0"
echo Starting Market Intel... your browser will open shortly.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
