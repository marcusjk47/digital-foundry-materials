# 🚀 CALPHAD Quick Start

**Get phase diagram calculations running in 10 minutes!**

---

## ⚡ Step 1: Install (Choose One Method)

### **Method A: Easy Install Script** ⭐

Just double-click:
```
install_calphad.bat
```

This automatically installs PyCalphad, ESPEI, and Scheil!

### **Method B: Manual Install (Conda)**

```bash
# Open command prompt
cd "C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project"
activate mp-alloy-env

# Install from conda-forge
conda install -c conda-forge pycalphad espei scheil
```

### **Method C: Manual Install (Pip)**

```bash
# Activate environment
activate mp-alloy-env

# Install from requirements file
pip install -r requirements_calphad.txt
```

---

## ✅ Step 2: Test Installation

```bash
python test_calphad.py
```

**Expected output:**
```
============================================================
✅ All critical tests passed! CALPHAD is ready to use!
============================================================
```

---

## 📥 Step 3: Get Thermodynamic Databases

### **Quick Start: Use My Example TDB**

I'll create a simple Al-Zn database for you to test with immediately!

### **For More Systems:**

1. Visit **NIMS Database**: https://cpddb.nims.go.jp/
2. Register (free)
3. Search for systems: Al-Zn, Fe-Cr, Cu-Ni, etc.
4. Download TDB files
5. Save to: `calphad_databases/`

**Recommended starter databases:**
- **alzn_mey.tdb** - Aluminum-Zinc
- **crfe.tdb** - Chromium-Iron
- **cuni.tdb** - Copper-Nickel
- **tial.tdb** - Titanium-Aluminum

---

## 🎓 Step 4: Run Your First Calculation

Create a file `my_first_phase_diagram.py`:

```python
from pycalphad import Database, binplot
import pycalphad.variables as v
import matplotlib.pyplot as plt

# Load database (use your TDB file)
db = Database('calphad_databases/alzn_mey.tdb')

# Define system
comps = ['AL', 'ZN', 'VA']
phases = list(db.phases.keys())

# Calculation conditions
conditions = {
    v.N: 1,  # 1 mole
    v.P: 101325,  # 1 atm
    v.T: (300, 1000, 10),  # 300-1000K
    v.X('ZN'): (0, 1, 0.01)  # 0-100% Zn
}

# Calculate and plot!
binplot(db, comps, phases, conditions)
plt.title('Al-Zn Binary Phase Diagram')
plt.ylabel('Temperature (K)')
plt.xlabel('Mole Fraction Zn')
plt.savefig('my_first_phase_diagram.png', dpi=150)
plt.show()

print("✓ Phase diagram created!")
```

Run it:
```bash
python my_first_phase_diagram.py
```

You just calculated a phase diagram! 🎉

---

## 📚 Step 5: Learn More

- **Detailed Guide:** `CALPHAD_SETUP_GUIDE.md`
- **PyCalphad Docs:** https://pycalphad.org/docs/
- **Examples:** `calphad_examples/` folder (coming soon!)

---

## 🎯 What's Next?

**Phase 2: Build Streamlit App**

Next, we'll create an interactive web app where you can:
- Upload TDB files
- Select elements
- Generate phase diagrams interactively
- Calculate equilibrium at any T and composition
- Compare with Materials Project data

**Coming up:**
- `calphad_app.py` - Interactive phase diagram calculator
- Integration with your existing Materials Project data
- Combine CALPHAD + DFT features for ML models

---

## 🆘 Need Help?

**Installation issues?**
→ Check `CALPHAD_SETUP_GUIDE.md` troubleshooting section

**Can't find TDB files?**
→ I can help you create a simple database for testing

**Want to learn more?**
→ PyCalphad has excellent documentation and examples

---

## ✅ Quick Checklist

- [ ] Installed PyCalphad (`test_calphad.py` passes)
- [ ] Have at least 1 TDB file
- [ ] Can load database in Python
- [ ] Calculated a simple phase diagram
- [ ] Ready for Phase 2 (Streamlit app)!

---

**🔬 Welcome to computational thermodynamics!**

You can now calculate phase diagrams, predict phase stability, and simulate solidification - all with Python!
