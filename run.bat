@echo off
title Stock Market Analysis

cd /d "%~dp0"

echo Starting Stock Market Analysis...
echo.

"C:\Program Files\Python312\python.exe" -m streamlit run app.py

pause