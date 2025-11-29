@echo off
echo Running Data Loader...
cd /d "%~dp0"
python src/data_loader.py
echo.
echo Data collection complete.
pause
