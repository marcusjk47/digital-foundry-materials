@echo off
REM Install PyCalphad and CALPHAD tools

title Installing PyCalphad

echo ========================================
echo   Installing CALPHAD Tools
echo ========================================
echo.
echo This will install:
echo   - PyCalphad (phase diagram calculations)
echo   - ESPEI (parameter optimization)
echo   - Scheil (solidification simulations)
echo.
echo Using conda-forge (recommended for Windows)
echo.
echo ========================================
echo.

cd /d "%~dp0"

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Conda found, using conda for installation...
    echo.
    call mp-alloy-env\Scripts\activate.bat
    conda install -c conda-forge pycalphad espei scheil -y
) else (
    echo Conda not found, using pip for installation...
    echo (Note: conda is recommended for Windows)
    echo.
    call mp-alloy-env\Scripts\activate.bat
    python -m pip install -r requirements_calphad.txt
)

echo.
echo ========================================
echo Testing installation...
echo ========================================
echo.

REM Test installation
python test_calphad.py

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Next steps:
echo 1. Download TDB files from NIMS: https://cpddb.nims.go.jp/
echo 2. Place them in: calphad_databases\
echo 3. Run examples from CALPHAD_SETUP_GUIDE.md
echo.
pause
