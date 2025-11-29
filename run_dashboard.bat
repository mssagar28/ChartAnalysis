@echo off
echo Starting Candlestick Analysis Dashboard...
cd /d "%~dp0"
python -m streamlit run dashboard/app.py
pause
