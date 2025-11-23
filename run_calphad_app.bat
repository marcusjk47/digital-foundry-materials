@echo off
REM Launch CALPHAD Phase Diagram Calculator (Streamlit App)

title CALPHAD Phase Diagrams

echo ========================================
echo   CALPHAD Phase Diagram Calculator
echo ========================================
echo.
echo Starting Streamlit app on port 8503...
echo.
echo Once loaded, your browser will open automatically.
echo.
echo To stop the app: Close this window or press Ctrl+C
echo.
echo ========================================
echo.

cd /d "%~dp0"

REM Activate virtual environment
call mp-alloy-env\Scripts\activate.bat

REM Run Streamlit app on port 8503 (8501=main app, 8502=experimental app)
streamlit run calphad_app.py --server.port 8503

pause
