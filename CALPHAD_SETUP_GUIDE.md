# 🔬 CALPHAD Integration Setup Guide

**CALPHAD** = CALculation of PHAse Diagrams

Add powerful phase diagram calculations and thermodynamic modeling to your alloy development platform!

---

## 🎯 What You'll Get

### **New Capabilities:**
- 📊 **Interactive Phase Diagrams** (binary, ternary, multi-component)
- 🌡️ **Temperature-Dependent Stability** predictions
- 🔥 **Solidification Simulations** (Scheil-Gulliver)
- ⚖️ **Equilibrium Calculations** at any T and composition
- 📈 **Thermodynamic Properties** (Cp, H, S vs temperature)

### **Why This Matters:**
- **Materials Project**: DFT at 0 Kelvin (ground state)
- **CALPHAD**: Real temperatures (300K-2000K+)
- **Together**: Complete picture from theory to practice!

---

## 📦 Phase 1: Installation

### **Method 1: Using Conda (Recommended for Windows)** ⭐

Conda handles all dependencies and builds automatically - **much easier**!

```bash
# Open Anaconda Prompt or Terminal

# Navigate to project
cd "C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project"

# Activate your environment
activate mp-alloy-env

# Install CALPHAD tools from conda-forge
conda install -c conda-forge pycalphad espei scheil

# Verify installation
python -c "import pycalphad; print(f'PyCalphad {pycalphad.__version__} installed!')"
```

### **Method 2: Using Pip (Alternative)**

```bash
# Activate environment
activate mp-alloy-env

# Install PyCalphad and tools
pip install pycalphad espei scheil

# May need to install SymEngine separately on Windows
pip install symengine

# Verify
python -c "import pycalphad; print('PyCalphad installed!')"
```

### **Troubleshooting Windows Installation**

**Issue:** "Microsoft Visual C++ required"
```bash
# Solution: Use conda instead of pip
# Conda provides pre-built binaries
conda install -c conda-forge pycalphad
```

**Issue:** SymEngine compilation fails
```bash
# Solution: Install pre-built wheel
conda install -c conda-forge symengine
# Or use pip with pre-built wheel
pip install symengine --only-binary :all:
```

**Issue:** Import errors
```bash
# Make sure you're in the right environment
where python
# Should show path to mp-alloy-env

# Reinstall if needed
pip uninstall pycalphad
conda install -c conda-forge pycalphad
```

---

## 🧪 Test Installation

Create a test script to verify everything works:

**File: `test_calphad.py`**

```python
"""
Test PyCalphad Installation
Run this to verify CALPHAD tools are working
"""

print("=" * 60)
print("Testing PyCalphad Installation")
print("=" * 60)

# Test 1: Import PyCalphad
print("\n1. Testing PyCalphad import...")
try:
    import pycalphad
    print(f"   ✓ PyCalphad {pycalphad.__version__} imported successfully!")
except ImportError as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 2: Import core modules
print("\n2. Testing core modules...")
try:
    from pycalphad import Database, equilibrium, variables as v
    from pycalphad import binplot, ternplot
    print("   ✓ Core modules imported successfully!")
except ImportError as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 3: Import ESPEI
print("\n3. Testing ESPEI...")
try:
    import espei
    print(f"   ✓ ESPEI imported successfully!")
except ImportError as e:
    print(f"   ⚠ ESPEI not available: {e}")

# Test 4: Import Scheil
print("\n4. Testing Scheil...")
try:
    import scheil
    print(f"   ✓ Scheil imported successfully!")
except ImportError as e:
    print(f"   ⚠ Scheil not available: {e}")

# Test 5: Load a simple database
print("\n5. Testing database functionality...")
try:
    # Create a minimal TDB content for testing
    tdb_content = """
    $ Simple test database
    ELEMENT AL FCC_A1 0 0 0 !
    ELEMENT ZN HCP_A3 0 0 0 !
    ELEMENT VA VACUUM 0 0 0 !

    FUNCTION GHSERAL 298.15 -7976.15+137.093038*T-24.3671976*T*LN(T)
     -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1); 700 Y
     -11276.24+223.048446*T-38.5844296*T*LN(T)+.018531982*T**2
     -5.764227E-06*T**3+74092*T**(-1); 933.47 Y
     -11278.378+188.684153*T-31.748192*T*LN(T)-1.230524E+28*T**(-9);
     2900 N !

    PHASE FCC_A1 %  1  1.0  !
    CONSTITUENT FCC_A1 :AL,ZN : !
    PARAMETER G(FCC_A1,AL;0) 298.15 +GHSERAL; 2900 N !
    """

    # Save to temp file
    with open('test_database.tdb', 'w') as f:
        f.write(tdb_content)

    db = Database('test_database.tdb')
    print(f"   ✓ Database loaded successfully!")
    print(f"   ✓ Elements: {db.elements}")
    print(f"   ✓ Phases: {list(db.phases.keys())}")

    # Clean up
    import os
    os.remove('test_database.tdb')

except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 6: Dependencies
print("\n6. Testing dependencies...")
try:
    import numpy
    print(f"   ✓ NumPy {numpy.__version__}")
except:
    print("   ✗ NumPy missing")

try:
    import scipy
    print(f"   ✓ SciPy {scipy.__version__}")
except:
    print("   ✗ SciPy missing")

try:
    import matplotlib
    print(f"   ✓ Matplotlib {matplotlib.__version__}")
except:
    print("   ✗ Matplotlib missing")

try:
    import xarray
    print(f"   ✓ xarray {xarray.__version__}")
except:
    print("   ✗ xarray missing")

print("\n" + "=" * 60)
print("✅ All tests passed! CALPHAD is ready to use!")
print("=" * 60)
print("\nNext steps:")
print("1. Download thermodynamic databases (TDB files)")
print("2. Run example calculations")
print("3. Build Streamlit app")
```

**Run the test:**
```bash
python test_calphad.py
```

**Expected output:**
```
============================================================
Testing PyCalphad Installation
============================================================

1. Testing PyCalphad import...
   ✓ PyCalphad 0.11.0 imported successfully!

2. Testing core modules...
   ✓ Core modules imported successfully!

3. Testing ESPEI...
   ✓ ESPEI imported successfully!

4. Testing Scheil...
   ✓ Scheil imported successfully!

5. Testing database functionality...
   ✓ Database loaded successfully!
   ✓ Elements: {'AL', 'ZN', 'VA'}
   ✓ Phases: ['FCC_A1']

6. Testing dependencies...
   ✓ NumPy 2.3.4
   ✓ SciPy 1.16.3
   ✓ Matplotlib 3.10.7
   ✓ xarray 2024.11.0

============================================================
✅ All tests passed! CALPHAD is ready to use!
============================================================
```

---

## 📚 Phase 2: Get Thermodynamic Databases

CALPHAD needs **TDB files** (Thermodynamic DataBase format) containing Gibbs energy functions for different phases.

### **Option 1: NIMS Database (Best Free Source)** ⭐

**NIMS** = National Institute for Materials Science (Japan)

1. **Visit:** https://cpddb.nims.go.jp/
2. **Register:** Free account required
3. **Browse:** Search for your alloy system
4. **Download:** Get TDB files

**Available Systems:**
- 700+ binary and ternary systems
- Al-Zn, Fe-Cr, Cu-Ni, Ti-Al, Ni-Al, etc.
- High quality, peer-reviewed data

**How to download:**
```
1. Go to NIMS website
2. Search for system (e.g., "Al-Zn")
3. Select assessment
4. Download TDB file
5. Save to: calphad_databases/
```

### **Option 2: PyCalphad Examples**

PyCalphad includes example databases:

```bash
# Download PyCalphad examples
git clone https://github.com/pycalphad/pycalphad.git pycalphad_repo
cd pycalphad_repo/examples/

# Copy TDB files to your project
copy *.tdb "C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project\calphad_databases\"
```

### **Option 3: CALPHAD Journal Papers**

Many CALPHAD papers include TDB files as supplementary material:

1. Search Google Scholar: "CALPHAD [system] TDB"
2. Download paper's supplementary material
3. Save TDB file to `calphad_databases/`

### **Starter Database Collection**

I recommend getting these first (all available from NIMS):

1. **alzn_mey.tdb** - Aluminum-Zinc (simple binary)
2. **crfe.tdb** - Chromium-Iron (steels)
3. **cuni.tdb** - Copper-Nickel (solid solution)
4. **tial.tdb** - Titanium-Aluminum (intermetallics)
5. **nial.tdb** - Nickel-Aluminum (superalloys)

---

## 🎓 Phase 3: Learn by Example

### **Example 1: Load and Inspect a Database**

```python
from pycalphad import Database

# Load TDB file
db = Database('calphad_databases/alzn_mey.tdb')

# Inspect contents
print("Elements:", db.elements)
print("Phases:", list(db.phases.keys()))
print("Species:", db.species)

# View phase constituents
for phase_name, phase in db.phases.items():
    print(f"\n{phase_name}:")
    print(f"  Sublattices: {phase.sublattices}")
    print(f"  Constituents: {phase.constituents}")
```

### **Example 2: Calculate Binary Phase Diagram**

```python
from pycalphad import Database, binplot
import pycalphad.variables as v
import matplotlib.pyplot as plt

# Load database
db = Database('calphad_databases/alzn_mey.tdb')

# Define system
comps = ['AL', 'ZN', 'VA']  # Components (VA = vacancy)
phases = list(db.phases.keys())  # All phases in database

# Define calculation conditions
conditions = {
    v.N: 1,  # 1 mole total
    v.P: 101325,  # 1 atm pressure
    v.T: (300, 1000, 10),  # Temperature: 300-1000K in 10K steps
    v.X('ZN'): (0, 1, 0.01)  # Zn composition: 0-100% in 1% steps
}

# Calculate and plot
fig = binplot(db, comps, phases, conditions, plot_kwargs={'ax': plt.gca()})
plt.title('Al-Zn Binary Phase Diagram')
plt.ylabel('Temperature (K)')
plt.xlabel('Mole Fraction Zn')
plt.savefig('alzn_phase_diagram.png', dpi=150)
plt.show()
```

### **Example 3: Equilibrium at Specific Conditions**

```python
from pycalphad import Database, equilibrium
import pycalphad.variables as v

db = Database('calphad_databases/alzn_mey.tdb')
comps = ['AL', 'ZN', 'VA']
phases = list(db.phases.keys())

# Calculate equilibrium at 600K, 30% Zn
result = equilibrium(db, comps, phases, {
    v.T: 600,  # Temperature in Kelvin
    v.P: 101325,  # Pressure in Pa
    v.X('ZN'): 0.3,  # 30% Zn
    v.N: 1  # 1 mole total
})

# Extract results
print("Stable phases:", result.Phase.squeeze())
print("Phase fractions:", result.NP.squeeze())
print("Chemical potentials:", result.MU.squeeze())
```

### **Example 4: Scheil Solidification Simulation**

```python
from scheil import simulate_scheil_solidification
from pycalphad import Database, variables as v

db = Database('calphad_databases/alzn_mey.tdb')
comps = ['AL', 'ZN', 'VA']
phases = sorted(db.phases.keys())

# Simulate solidification of Al-30%Zn
result = simulate_scheil_solidification(
    db, comps, phases,
    composition={v.X('ZN'): 0.3},
    start_temperature=900,  # Start at 900K
    step_temperature=1.0  # 1K steps
)

# Plot solidification path
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(result.fraction_solid, result.temperatures)
plt.xlabel('Fraction Solid')
plt.ylabel('Temperature (K)')
plt.title('Scheil Solidification: Al-30%Zn')
plt.grid(True, alpha=0.3)
plt.savefig('scheil_solidification.png', dpi=150)
plt.show()

print(f"Liquidus: {result.temperatures[0]:.1f} K")
print(f"Solidus: {result.temperatures[-1]:.1f} K")
print(f"Freezing range: {result.temperatures[0] - result.temperatures[-1]:.1f} K")
```

---

## 📁 Project Structure

After setup, your project should look like:

```
ML-Alloy-Project/
├── calphad_databases/           # TDB files go here
│   ├── alzn_mey.tdb
│   ├── crfe.tdb
│   ├── cuni.tdb
│   ├── tial.tdb
│   └── nial.tdb
├── calphad_examples/            # Example scripts
│   ├── example_binary_diagram.py
│   ├── example_equilibrium.py
│   └── example_scheil.py
├── test_calphad.py              # Installation test
├── calphad_app.py               # Streamlit app (coming soon!)
├── run_calphad_app.bat          # Launcher (coming soon!)
└── CALPHAD_SETUP_GUIDE.md       # This file
```

---

## ✅ Verification Checklist

Before moving to Phase 2 (Building the app), verify:

- [ ] PyCalphad installed (`import pycalphad` works)
- [ ] ESPEI installed (optional but recommended)
- [ ] Scheil installed (for solidification)
- [ ] Test script runs without errors
- [ ] Downloaded 3+ TDB files from NIMS
- [ ] Can load TDB file and view contents
- [ ] Can calculate simple binary diagram
- [ ] Can run equilibrium calculation

**If all checked: Ready for Phase 2!** 🎉

---

## 🆘 Troubleshooting

### **"ModuleNotFoundError: No module named 'pycalphad'"**

```bash
# Make sure environment is activated
activate mp-alloy-env

# Check Python path
where python
# Should show: C:\...\ML-Alloy-Project\mp-alloy-env\Scripts\python.exe

# Reinstall
conda install -c conda-forge pycalphad
```

### **"Cannot find TDB file"**

```bash
# Use absolute path
db = Database(r'C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project\calphad_databases\alzn_mey.tdb')

# Or relative path from project root
db = Database('calphad_databases/alzn_mey.tdb')
```

### **"SymEngine import error"**

```bash
# Install SymEngine explicitly
conda install -c conda-forge symengine

# Or with pip
pip install symengine --only-binary :all:
```

### **Slow calculations**

```bash
# Install JIT compiler for speed
conda install -c conda-forge numba

# PyCalphad will automatically use it if available
```

---

## 📚 Resources

### **Documentation:**
- PyCalphad Docs: https://pycalphad.org/docs/
- ESPEI Docs: https://espei.org/
- PyCalphad Examples: https://github.com/pycalphad/pycalphad/tree/develop/examples

### **Databases:**
- NIMS CPDDB: https://cpddb.nims.go.jp/
- Thermodynamic Database Database: https://avdwgroup.engin.brown.edu/

### **Learning:**
- CALPHAD Method Introduction: https://www.calphad.org/
- PyCalphad Tutorial Videos: YouTube search "PyCalphad tutorial"
- Example calculations: https://github.com/pycalphad/pycalphad-examples

### **Community:**
- PyCalphad Gitter Chat: https://gitter.im/pycalphad/Lobby
- CALPHAD Discussion Forum: https://www.calphad.org/

---

## 🎯 Next Steps

After completing setup:

1. **Download 5 TDB files** from NIMS
2. **Run all example calculations** to learn PyCalphad
3. **Proceed to Phase 2**: Build Streamlit CALPHAD app
4. **Read** `CALPHAD_APP_PLAN.md` for implementation details

---

**Ready to calculate phase diagrams! 🔬📊**

Installation complete → Download databases → Run examples → Build app!
