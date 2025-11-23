@echo off
REM Launch Experimental Data Manager

title Experimental Data Manager

echo ========================================
echo   Experimental Data Manager
echo ========================================
echo.
echo Starting application...
echo.
echo Upload and visualize temperature-dependent
echo alloy properties (strength, hardness, etc.)
echo.
echo Press Ctrl+C to stop the server.
echo.
echo ========================================

cd /d "%~dp0"
call mp-alloy-env\Scripts\activate.bat
streamlit run experimental_data_manager.py --server.port 8502
