@echo off
REM Launch ML Alloy Development Web Dashboard

title ML Alloy Development - Web Dashboard

echo ========================================
echo   ML Alloy Development Web Dashboard
echo ========================================
echo.
echo Starting web application...
echo.
echo The dashboard will open in your web browser.
echo Press Ctrl+C to stop the server.
echo.
echo ========================================

cd /d "%~dp0"
call mp-alloy-env\Scripts\activate.bat
streamlit run app.py
