@echo off
REM Activation script for ML Alloy Project
echo ========================================
echo   ML Alloy Development Project
echo ========================================
echo.
echo Activating virtual environment...
call mp-alloy-env\Scripts\activate.bat
echo.
echo Environment activated!
echo Python:
python --version
echo.
echo To deactivate, type: deactivate
echo ========================================
