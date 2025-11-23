@echo off
echo ============================================================
echo Installing GNN Dependencies for Digital Foundry
echo ============================================================
echo.

echo Step 1: Activating environment...
call mp-alloy-env\Scripts\activate.bat

echo.
echo Step 2: Installing PyTorch (CPU version for Streamlit Cloud)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Step 3: Installing PyTorch Geometric dependencies...
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

echo.
echo Step 4: Installing PyTorch Geometric...
pip install torch-geometric

echo.
echo Step 5: Installing Pymatgen...
pip install pymatgen

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo Running verification tests...
python test_gnn_setup.py

echo.
echo If all tests passed, you're ready to build the GNN!
echo.
pause
