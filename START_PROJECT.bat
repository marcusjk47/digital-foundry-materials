@echo off
REM ML Alloy Project Launcher
REM Double-click this file to start working!

title ML Alloy Development Project

echo ========================================
echo   ML Alloy Development Project
echo ========================================
echo.
echo Starting environment...
echo.

REM Change to project directory
cd /d "%~dp0"

REM Activate virtual environment and keep window open
cmd /k "mp-alloy-env\Scripts\activate.bat && echo. && echo ========================================== && echo   Environment Ready! && echo ========================================== && echo. && echo Your API key is loaded from .env && echo. && echo Try these commands: && echo   python test_mp_connection.py && echo   python download_first_dataset.py && echo   python && echo. && echo =========================================="
